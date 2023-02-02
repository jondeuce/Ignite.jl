using Ignite
using Test

using Aqua: test_all
using DataStructures: OrderedDict
using Logging: NullLogger

@testset "Ignite.jl" begin
    function dummy_process_function(engine, batch)
        engine.state.batch = batch
        return Dict("loss" => sum(map(sum, batch)))
    end

    function dummy_engine(f = dummy_process_function; logger = NullLogger(), kwargs...)
        Engine(f; kwargs..., logger)
    end

    function dummy_trainer_and_loader(args...; max_epochs = 10, epoch_length = 10, kwargs...)
        engine = dummy_engine(args...; kwargs...)
        dataloader = (rand(3) for _ in 1:max_epochs * epoch_length)
        return engine, dataloader
    end

    function fire_and_check_triggered(engine, event, prim_event)
        fire_event!(engine, prim_event)
        return Ignite.is_triggered_by(engine, event, prim_event)
    end

    @testset "EPOCH_STARTED" begin
        @testset "every <Int>" begin
            trainer = dummy_engine()
            event = EPOCH_STARTED(; every = 2)
            for i in 1:5
                @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED())
                @test !fire_and_check_triggered(trainer, event, ITERATION_COMPLETED())
                t = fire_and_check_triggered(trainer, event, EPOCH_STARTED())
                @test !t || (t && mod1(i, 2) == 2)
            end
        end

        @testset "every <Int list>" begin
            trainer = dummy_engine()
            every = [2, 5, 7]
            event = EPOCH_STARTED(; every = every)
            for i in 1:25
                @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED())
                @test !fire_and_check_triggered(trainer, event, ITERATION_COMPLETED())
                t = fire_and_check_triggered(trainer, event, EPOCH_STARTED())
                @test !t || (t && any(mod1.(i, every) .== every))
            end
        end

        @testset "once <Int list>" begin
            trainer = dummy_engine()
            once = [2, 5, 7]
            event = EPOCH_STARTED(; once = once)
            for i in 1:25
                @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED())
                @test !fire_and_check_triggered(trainer, event, ITERATION_COMPLETED())
                t = fire_and_check_triggered(trainer, event, EPOCH_STARTED())
                @test !t || (t && any(i .== once))
            end
        end
    end

    @testset "ITERATION_COMPLETED" begin
        @testset "once <Int>" begin
            trainer = dummy_engine()
            event = ITERATION_COMPLETED(; once = 4)
            for i in 1:7
                @test !fire_and_check_triggered(trainer, event, EPOCH_STARTED())
                @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED())
                t = fire_and_check_triggered(trainer, event, ITERATION_COMPLETED())
                @test !t || (t && i == 4)
            end
        end
    end

    @testset "throttle filter" begin
        engine, event = dummy_engine(), EPOCH_COMPLETED() # dummy arguments
        event_filter = throttle_filter(1.0)
        @test event_filter(engine, event) # first call is unthrottled
        sleep(0.6); @test !event_filter(engine, event)
        sleep(0.6); @test event_filter(engine, event) # throttle resets
        sleep(0.3); @test !event_filter(engine, event)
        sleep(0.3); @test !event_filter(engine, event)
        sleep(0.6); @test event_filter(engine, event) # throttle resets
    end

    @testset "timeout filter" begin
        engine, event = dummy_engine(), EPOCH_COMPLETED() # dummy arguments
        event_filter = timeout_filter(1.0)
        t₀ = time()
        while (Δt = time() - t₀) < 2.0
            @test event_filter(engine, event) == (Δt >= 1.0)
            sleep(0.3)
        end
    end

    @testset "OrEvent" begin
        trainer = dummy_engine()
        event_filter = (_engine, _event) -> _engine.state.new_field !== nothing
        event = EPOCH_COMPLETED(; every = 3) | EPOCH_COMPLETED(; event_filter)
        @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED())
        @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED())
        @test fire_and_check_triggered(trainer, event, EPOCH_COMPLETED())

        Ignite.reset!(trainer)
        @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED())
        trainer.state.new_field = 1
        @test fire_and_check_triggered(trainer, event, EPOCH_COMPLETED())
        @test fire_and_check_triggered(trainer, event, EPOCH_COMPLETED())
    end

    @testset "AndEvent" begin
        trainer = dummy_engine()
        event_filter = (_engine, _event) -> _engine.state.new_field !== nothing
        event = EPOCH_COMPLETED(; every = 3) & EPOCH_COMPLETED(; event_filter)
        @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED())
        @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED())
        @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED())
        trainer.state.new_field = 1
        @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED())
        @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED())
        @test fire_and_check_triggered(trainer, event, EPOCH_COMPLETED())
    end

    @testset "custom show methods" begin
        @test startswith(sprint(show, dummy_engine()), "Engine")
    end

    @testset "State methods" begin
        s = State()
        s[:iteration] = 0
        s.epoch = 0
        s[:new_field_1] = :one
        s.new_field_2 = "two"
        @test s[:iteration] == s.iteration == 0
        @test s[:epoch] == s.epoch == 0
        @test s[:new_field_1] == s.new_field_1 == :one
        @test s[:new_field_2] == s.new_field_2 == "two"
        @test get!(s, :new_field_3, 3) == 3
        @test s[:new_field_3] == s.new_field_3 == 3
        @test get(s, :new_field_4, 4.0) == 4.0
        @test !haskey(s, :new_field_4)
        @test length(s) == length(State()) + 3 == 11
        @test OrderedDict(k => v for (k, v) in s) == Ignite.state(s)
    end

    @testset "DataCycler" begin
        dl = Ignite.DataCycler(1:5)
        @test_throws MethodError length(dl)
        @test Ignite.default_epoch_length(dl) == 5
        @test Base.IteratorSize(dl) == Base.IsInfinite()
        @test Base.IteratorEltype(dl) == Base.HasEltype()
        @test Base.eltype(dl) == Int

        dl = Ignite.DataCycler(Iterators.cycle(1.0:5.0))
        @test_throws MethodError length(dl)
        @test_throws Ignite.DataLoaderUnknownLengthException Ignite.default_epoch_length(dl)
        @test Base.IteratorSize(dl) == Base.IsInfinite()
        @test Base.IteratorEltype(dl) == Base.HasEltype()
        @test Base.eltype(dl) == Float64
    end

    @testset "run!" begin
        @testset "early termination" begin
            max_epochs, epoch_length = 3, 5
            trainer, dl = dummy_trainer_and_loader(; max_epochs, epoch_length)

            add_event_handler!(trainer, EPOCH_COMPLETED()) do engine
                @test engine.state.output isa Dict
                @test engine.state.output["loss"] isa Float64
                engine.should_terminate = true
            end

            add_event_handler!(trainer, TERMINATE()) do engine
                engine.state.terminate_event_fired = true
            end

            @test Ignite.run!(trainer, dl; max_epochs, epoch_length).exception isa Ignite.TerminationException

            @test trainer.should_terminate
            @test trainer.state.iteration == epoch_length
            @test trainer.state.epoch == 1
            @test trainer.state.terminate_event_fired == true
        end

        @testset "all default events" begin
            max_epochs, epoch_length = 3, 5
            trainer, dl = dummy_trainer_and_loader(; max_epochs, epoch_length)
            add_event_handler!(trainer, STARTED()) do engine
                @test engine.state.iteration == 0
                @test engine.state.epoch == 0
            end
            add_event_handler!(trainer, EPOCH_STARTED()) do engine
                @test mod(engine.state.iteration, epoch_length) == 0
            end
            add_event_handler!(trainer, ITERATION_STARTED()) do engine
                #TODO
            end
            add_event_handler!(trainer, ITERATION_COMPLETED()) do engine
                #TODO
            end
            add_event_handler!(trainer, EPOCH_COMPLETED()) do engine
                @test mod1(engine.state.iteration, epoch_length) == epoch_length
            end
            add_event_handler!(trainer, COMPLETED()) do engine
                @test engine.state.epoch == max_epochs
                @test engine.state.iteration == max_epochs * epoch_length
            end
            Ignite.run!(trainer, dl; max_epochs, epoch_length)
        end

        @testset "event ordering" begin
            max_epochs, epoch_length = 7, 3
            final_iter = epoch_length * (max_epochs - 1) + 1
            trainer, dl = dummy_trainer_and_loader(; max_epochs, epoch_length)

            event_list = Any[]
            add_event_handler!(trainer, EPOCH_COMPLETED(; every = 3) | INTERRUPT() | TERMINATE()) do engine
                push!(event_list, engine.state.last_event)
            end
            add_event_handler!(trainer, ITERATION_COMPLETED(; once = final_iter)) do engine
                @test engine.state.iteration == final_iter
                throw(InterruptException())
            end

            @test Ignite.run!(trainer, dl; max_epochs, epoch_length).exception isa InterruptException

            @test event_list == Any[EPOCH_COMPLETED(), EPOCH_COMPLETED(), INTERRUPT(), TERMINATE()]
        end

        @testset "default epoch length" begin
            max_epochs, epoch_length = 7, 3
            trainer = dummy_engine()
            dl = 1:5

            # `length` fails for infinite data loader
            @test Ignite.run!(trainer, Iterators.cycle(dl)).exception isa Ignite.DataLoaderUnknownLengthException

            dl = collect(dl)
            add_event_handler!(trainer, ITERATION_COMPLETED()) do engine
                @test engine.state.batch == dl[mod1(engine.state.iteration, length(dl))]
            end

            Ignite.run!(trainer, dl)
        end

        @testset "empty data loader" begin
            max_epochs, epoch_length = 7, 3
            trainer = dummy_engine()
            dl = []

            add_event_handler!(trainer, DATALOADER_STOP_ITERATION()) do engine
                engine.state.dataloader_stop_event_fired = true
            end

            @test Ignite.run!(trainer, dl; max_epochs, epoch_length).exception isa Ignite.DataLoaderEmptyException
            @test trainer.state.dataloader_stop_event_fired == true
        end

        @testset "exception raised" begin
            max_epochs, epoch_length = 7, 3
            trainer, dl = dummy_trainer_and_loader(; max_epochs, epoch_length) do engine, batch
                return engine.state.iteration >= 10 ? x : nothing # x is undefined
            end

            add_event_handler!(trainer, EXCEPTION_RAISED()) do engine
                engine.state.exception_raised_event_fired = true
            end

            @test Ignite.run!(trainer, dl; max_epochs, epoch_length).exception isa UndefVarError
            @test trainer.state.exception_raised_event_fired == true
        end

        @testset "many handlers" begin
            max_epochs, epoch_length = 7, 3
            trainer, dl = dummy_trainer_and_loader(; max_epochs, epoch_length)

            num_handlers = 1000
            events = [STARTED(), EPOCH_STARTED(), ITERATION_STARTED(), ITERATION_COMPLETED(), EPOCH_COMPLETED(), COMPLETED()]
            buffer = zeros(Int, num_handlers)

            for (i, event) in zip(1:num_handlers, Iterators.cycle(events))
                add_event_handler!(trainer, event) do engine
                    return buffer[i] = i
                end
            end

            Ignite.run!(trainer, dl)
            @test buffer == 1:num_handlers
        end

        @testset "additional handler args" begin
            max_epochs, epoch_length = 7, 3
            trainer, dl = dummy_trainer_and_loader(; max_epochs, epoch_length)

            data = Ref(0)
            add_event_handler!(trainer, ITERATION_COMPLETED(), data) do engine, _data
                _data[] += engine.state.iteration
            end
            @test trainer.event_handlers[1].args[1] === data

            Ignite.run!(trainer, dl)
            @test data[] == sum(1:max_epochs * epoch_length)
        end

        @testset "custom events" begin
            struct NEW_LOOP_EVENT <: AbstractLoopEvent end
            struct NEW_FIRING_EVENT <: AbstractFiringEvent end

            @test NEW_LOOP_EVENT(every = 3) isa FilteredEvent
            @test_throws MethodError NEW_FIRING_EVENT(every = 3)

            trainer = dummy_engine() do engine, batch
                fire_event!(engine, NEW_LOOP_EVENT())
                fire_event!(engine, NEW_FIRING_EVENT())
                return nothing
            end

            loop_event_counter, firing_event_counter = Ref(0), Ref(0)
            add_event_handler!(_ -> loop_event_counter[] += 1, trainer, NEW_LOOP_EVENT(every = 3))
            add_event_handler!(_ -> firing_event_counter[] += 1, trainer, NEW_FIRING_EVENT())

            Ignite.run!(trainer, [1]; epoch_length = 12)

            @test loop_event_counter[] == 4
            @test firing_event_counter[] == 12
        end
    end
end

# (A)uto (QU)ality (A)ssurance tests
@testset "Aqua.jl" begin
    test_all(Ignite)
end
