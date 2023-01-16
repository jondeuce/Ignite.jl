using Ignite
using Test

using Logging: NullLogger

@testset "Ignite.jl" begin
    function dummy_trainer_and_loader(; max_epochs = 10, epoch_length = 10)
        process_function = (_engine, _batch) -> Dict{String, Any}("loss" => sum(map(sum, _batch)))
        engine = Engine(process_function; logger = NullLogger())
        dataloader = (rand(3) for _ in 1:max_epochs * epoch_length)
        return engine, dataloader
    end

    function fire_and_check_triggered(engine, event, prim_event)
        fire_event!(engine, prim_event)
        return Ignite.is_triggered_by(engine, event, prim_event)
    end

    @testset "EPOCH_STARTED" begin
        @testset "every <Int>" begin
            trainer, _ = dummy_trainer_and_loader()
            event = EPOCH_STARTED(; every = 2)
            for i in 1:5
                @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED())
                @test !fire_and_check_triggered(trainer, event, ITERATION_COMPLETED())
                t = fire_and_check_triggered(trainer, event, EPOCH_STARTED())
                @test !t || (t && mod1(i, 2) == 2)
            end
        end

        @testset "every <Int list>" begin
            trainer, _ = dummy_trainer_and_loader()
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
            trainer, _ = dummy_trainer_and_loader()
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
            trainer, _ = dummy_trainer_and_loader()
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
        engine, event = Engine(nothing), EPOCH_COMPLETED() # dummy arguments
        filter = throttle_filter(0.1)
        @test !filter(engine, event)
        sleep(0.05); @test !filter(engine, event)
        sleep(0.05); @test filter(engine, event)
        sleep(0.025); @test !filter(engine, event)
        sleep(0.025); @test !filter(engine, event)
        sleep(0.05); @test filter(engine, event)
    end

    @testset "OrEvent" begin
        trainer, _ = dummy_trainer_and_loader()
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
        trainer, _ = dummy_trainer_and_loader()
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
        trainer, _ = dummy_trainer_and_loader()
        @test sprint(show, trainer.state) == "Engine state"
        @test startswith(sprint(show, MIME"text/plain"(), trainer.state), "Engine state:\n")
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

            termination_fired = Ref(false)
            add_event_handler!(trainer, TERMINATE()) do engine
                termination_fired[] = true
            end

            @test_throws Ignite.TerminationException Ignite.run!(trainer, dl; max_epochs, epoch_length)

            @test trainer.should_terminate
            @test trainer.state.iteration == epoch_length
            @test trainer.state.epoch == 1
            @test termination_fired[]
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

            @test_throws InterruptException Ignite.run!(trainer, dl; max_epochs, epoch_length)

            @test event_list == Any[EPOCH_COMPLETED(), EPOCH_COMPLETED(), INTERRUPT(), TERMINATE()]
        end

        @testset "user exception" begin
            max_epochs, epoch_length = 7, 3
            trainer, dl = dummy_trainer_and_loader(; max_epochs, epoch_length)

            # `length` fails for infinite data loader
            @test_throws MethodError Ignite.run!(trainer, Iterators.cycle(dl))

            dl = collect(dl)
            add_event_handler!(trainer, ITERATION_COMPLETED()) do engine
                @test engine.state.batch == dl[mod1(engine.state.iteration, length(dl))]
            end

            Ignite.run!(trainer, dl)
        end

        @testset "default epoch length" begin
            max_epochs, epoch_length = 7, 3
            trainer, dl = dummy_trainer_and_loader(; max_epochs, epoch_length)

            # `length` fails for infinite data loader
            @test_throws MethodError Ignite.run!(trainer, Iterators.cycle(dl))

            dl = collect(dl)
            add_event_handler!(trainer, ITERATION_COMPLETED()) do engine
                @test engine.state.batch == dl[mod1(engine.state.iteration, length(dl))]
            end

            Ignite.run!(trainer, dl)
        end

        @testset "empty data loader" begin
            max_epochs, epoch_length = 7, 3
            trainer, _ = dummy_trainer_and_loader(; max_epochs, epoch_length)
            dl = []

            fired = Ref(false)
            add_event_handler!(trainer, DATALOADER_STOP_ITERATION()) do engine
                fired[] = true
            end

            @test_throws Ignite.DataLoaderEmptyException Ignite.run!(trainer, dl; max_epochs, epoch_length)
            @test fired[]
        end
    end

    @testset "README examples" begin
        @testset "Quickstart" begin
            #TODO
        end
    end
end
