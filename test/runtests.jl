using Ignite
using Test

using Ignite: fire_event!, is_triggered
using Logging: NullLogger

@testset "Ignite.jl" begin
    function dummy_trainer_and_loader(; max_epochs = 10, epoch_length = 10)
        process_function = (_engine, _batch) -> Dict{String, Any}("loss" => sum(map(sum, _batch)))
        engine = Engine(; process_function, logger = NullLogger())
        dataloader = (rand(3) for _ in 1:max_epochs * epoch_length)
        return engine, dataloader
    end

    function fire_and_check_triggered(engine, event, E)
        fire_event!(engine, E)
        return is_triggered(engine, event, E)
    end

    @testset "EPOCH_STARTED" begin
        @testset "every <Int>" begin
            trainer, _ = dummy_trainer_and_loader()
            event = EPOCH_STARTED(; every = 2)
            for i in 1:5
                @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED)
                @test !fire_and_check_triggered(trainer, event, ITERATION_COMPLETED)
                t = fire_and_check_triggered(trainer, event, EPOCH_STARTED)
                @test !t || (t && mod1(i, 2) == 2)
            end
        end

        @testset "every <Int list>" begin
            trainer, _ = dummy_trainer_and_loader()
            every = [2, 5, 7]
            event = EPOCH_STARTED(; every = every)
            for i in 1:25
                @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED)
                @test !fire_and_check_triggered(trainer, event, ITERATION_COMPLETED)
                t = fire_and_check_triggered(trainer, event, EPOCH_STARTED)
                @test !t || (t && any(mod1.(i, every) .== every))
            end
        end

        @testset "once <Int list>" begin
            trainer, _ = dummy_trainer_and_loader()
            once = [2, 5, 7]
            event = EPOCH_STARTED(; once = once)
            for i in 1:25
                @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED)
                @test !fire_and_check_triggered(trainer, event, ITERATION_COMPLETED)
                t = fire_and_check_triggered(trainer, event, EPOCH_STARTED)
                @test !t || (t && any(i .== once))
            end
        end
    end

    @testset "ITERATION_COMPLETED" begin
        @testset "once <Int>" begin
            trainer, _ = dummy_trainer_and_loader()
            event = ITERATION_COMPLETED(; once = 4)
            for i in 1:7
                @test !fire_and_check_triggered(trainer, event, EPOCH_STARTED)
                @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED)
                t = fire_and_check_triggered(trainer, event, ITERATION_COMPLETED)
                @test !t || (t && i == 4)
            end
        end
    end

    @testset "AndEvent" begin
        trainer, _ = dummy_trainer_and_loader()
        event_filter = (_engine, _event) -> _engine.state.new_field !== nothing
        event = EPOCH_COMPLETED(; every = 3) & EPOCH_COMPLETED(; event_filter)
        @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED)
        @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED)
        @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED)
        trainer.state.new_field = 1
        @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED)
        @test !fire_and_check_triggered(trainer, event, EPOCH_COMPLETED)
        @test fire_and_check_triggered(trainer, event, EPOCH_COMPLETED)
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
            run!(trainer, dl; max_epochs, epoch_length)

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
            run!(trainer, dl; max_epochs, epoch_length)
        end

        @testset "event ordering" begin
            max_epochs, epoch_length = 7, 3
            trainer, dl = dummy_trainer_and_loader(; max_epochs, epoch_length)

            event_list = Any[]
            add_event_handler!(trainer, TERMINATE() | EPOCH_COMPLETED(; every = 3)) do engine
                push!(event_list, engine.state.last_event)
            end
            add_event_handler!(trainer, ITERATION_COMPLETED(; once = epoch_length * (max_epochs - 1) + 1)) do engine
                push!(event_list, engine.state.last_event)
                terminate!(engine)
            end

            run!(trainer, dl; max_epochs, epoch_length)

            @test event_list == Any[EPOCH_COMPLETED, EPOCH_COMPLETED, ITERATION_COMPLETED, TERMINATE]
        end

        @testset "data loader" begin
            max_epochs, epoch_length = 7, 3
            trainer, _ = dummy_trainer_and_loader(; max_epochs, epoch_length)
            dl = [rand(3) for _ in 1:11]

            fired = Ref(false)
            add_event_handler!(trainer, DATALOADER_STOP_ITERATION()) do engine
                @test engine.state.iteration == length(dl)
                fired[] = true
            end
            run!(trainer, dl; max_epochs, epoch_length)

            @test fired[]
        end
    end
end
