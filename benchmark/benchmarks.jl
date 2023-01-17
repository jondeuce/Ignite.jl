using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))
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

function dummy_event_handlers!(engine)
    add_event_handler!(engine, STARTED()) do engine
        @assert engine.state.iteration == 0
        @assert engine.state.epoch == 0
    end
    add_event_handler!(engine, EPOCH_STARTED()) do engine
        @assert mod(engine.state.iteration, engine.state.epoch_length) == 0
    end
    add_event_handler!(engine, ITERATION_STARTED()) do engine
        Ignite.@timeit engine.timer "dummy timing" 1+1
    end
    add_event_handler!(engine, ITERATION_COMPLETED()) do engine
        Ignite.@timeit engine.timer "dummy timing" 1+1
    end
    add_event_handler!(engine, EPOCH_COMPLETED()) do engine
        @assert mod1(engine.state.iteration, engine.state.epoch_length) == engine.state.epoch_length
    end
    add_event_handler!(engine, COMPLETED()) do engine
        @assert engine.state.epoch == engine.state.max_epochs
        @assert engine.state.iteration == engine.state.max_epochs * engine.state.epoch_length
    end
    return engine
end

function run_trainer(;
        dt = 1e-3,
        max_epochs = 5,
        epoch_length = round(Int, 1 / dt),
        add_event_handlers = true,
    )
    dl = [rand(1) for _ in 1:100]
    trainer = Engine(dummy_process_function(dt))
    add_event_handlers && dummy_event_handlers!(trainer)
    @time Ignite.run!(trainer, dl; max_epochs, epoch_length)
end

trainer = run_trainer(max_epochs = 1, dt = 1e-3, add_event_handlers = false);
trainer = run_trainer(max_epochs = 5, dt = 1e-3, add_event_handlers = false); Ignite.print_timer(trainer.timer)

trainer = run_trainer(max_epochs = 1, dt = 1e-3, add_event_handlers = true);
trainer = run_trainer(max_epochs = 5, dt = 1e-3, add_event_handlers = true); Ignite.print_timer(trainer.timer)

function fire_all_events!(engine)
    fire_event!(engine, STARTED())
    fire_event!(engine, EPOCH_STARTED())
    fire_event!(engine, ITERATION_STARTED())
    fire_event!(engine, ITERATION_COMPLETED())
    fire_event!(engine, EPOCH_COMPLETED())
    fire_event!(engine, COMPLETED())
    return nothing
end

function set_event_handlers!(engine, num_handlers::Int)
    events = [STARTED(), EPOCH_STARTED(), ITERATION_STARTED(), ITERATION_COMPLETED(), EPOCH_COMPLETED(), COMPLETED()]
    empty!(engine.event_handlers)
    while length(engine.event_handlers) < num_handlers
        for event in events
            add_event_handler!(engine, event) do engine
                return Ignite.curr_time() # dummy work
            end
            length(engine.event_handlers) >= num_handlers && break
        end
    end
    return engine
end

function bench_fire_events!(engine)
    for num_handlers in [0, 16, 64, 256, 257]
        set_event_handlers!(engine, num_handlers)
        @info "num. handlers: $(length(trainer.event_handlers))"
        @info "fire_all_events!:"; @time fire_all_events!(engine); @btime fire_all_events!($engine)
    end
    return engine
end

bench_fire_events!(trainer);
