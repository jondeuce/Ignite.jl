using Pkg
Pkg.activate(@__DIR__)
Pkg.update()

using Ignite
using BenchmarkTools

function sleep_expensive(dt)
    t = time_ns()
    while time_ns() - t < 1e9 * dt
        # More accurate than `sleep(dt)`, but is blocking and simulates a workload
    end
    return nothing
end

function dummy_process_function(dt)
    process_function(engine, batch) = Ignite.@timeit engine.timer "sleep_expensive" sleep_expensive(dt)
    return process_function
end

function run_trainer(;
        max_epochs = 5,
        iter_time = 500e-3,
        epoch_length = round(Int, 1 / iter_time),
        add_event_handlers = true,
    )

    dl = [rand(1) for _ in 1:100]
    trainer = Engine(dummy_process_function(iter_time))

    if add_event_handlers
        add_event_handler!(trainer, STARTED()) do engine
            @assert engine.state.iteration == 0
            @assert engine.state.epoch == 0
        end
        add_event_handler!(trainer, EPOCH_STARTED()) do engine
            @assert mod(engine.state.iteration, epoch_length) == 0
        end
        add_event_handler!(trainer, ITERATION_STARTED()) do engine
            Ignite.@timeit engine.timer "dummy timing" 1+1
        end
        add_event_handler!(trainer, ITERATION_COMPLETED()) do engine
            Ignite.@timeit engine.timer "dummy timing" 1+1
        end
        add_event_handler!(trainer, EPOCH_COMPLETED()) do engine
            @assert mod1(engine.state.iteration, epoch_length) == epoch_length
        end
        add_event_handler!(trainer, COMPLETED()) do engine
            @assert engine.state.epoch == max_epochs
            @assert engine.state.iteration == max_epochs * epoch_length
        end
    end

    @time Ignite.run!(trainer, dl; max_epochs, epoch_length)
end

trainer = run_trainer(max_epochs = 1, iter_time = 100e-3);
trainer = run_trainer(iter_time = 1e-3, add_event_handlers = false); Ignite.print_timer(trainer.timer)
trainer = run_trainer(iter_time = 1e-3, add_event_handlers = true); Ignite.print_timer(trainer.timer)
