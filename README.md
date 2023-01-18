# Ignite.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jondeuce.github.io/Ignite.jl/dev/)
[![Build Status](https://github.com/jondeuce/Ignite.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/jondeuce/Ignite.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/jondeuce/Ignite.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jondeuce/Ignite.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Welcome to `Ignite.jl`, a Julia port of the Python library [`ignite`](https://github.com/pytorch/ignite) for simplifying neural network training and validation loops using events and handlers.

`Ignite.jl` provides a simple yet flexible engine and event system, allowing for the easy composition of training pipelines with various events such as artifact saving, metric logging, and model validation. Event-based training abstracts away the training loop, replacing it with:
1. An *engine* which wraps a *process function* that consumes a single batch of data,
2. An iterable data loader which produces said batches of data, and
3. Events and corresponding event handlers which are attached to the engine, configured to fire at specific points during training.

Event handlers are much more flexibile compared to other approaches like callbacks: handlers can be any callable; multiple handlers can be attached to a single event; multiple events can trigger the same handler; and custom events can be defined to fire at user-specified points during training. This makes adding functionality to your training pipeline easy, minimizing the need to modify existing code.

## Quick Start

The example below demonstrates how to use `Ignite.jl` to train a simple neural network. Key features to note:
* The training step is factored out of the training loop: the `train_step` process function takes a batch of training data and computes the training loss, gradients, and updates the model parameters.
* Data loaders can be any iterable collection. Here, we use a [`DataLoader`](https://juliaml.github.io/MLUtils.jl/stable/api/#MLUtils.DataLoader) from [`MLUtils.jl`](https://github.com/JuliaML/MLUtils.jl)

````julia
using Ignite
using Flux, Zygote, Optimisers, MLUtils # for training a neural network

# Build simple neural network and initialize Adam optimizer
model = Chain(Dense(1 => 32, tanh), Dense(32 => 1))
optim = Optimisers.setup(Optimisers.Adam(1f-3), model)

# Create mock data and data loaders
f(x) = 2x-x^3
xtrain, xtest = 2 * rand(1, 10_000) .- 1, reshape(range(-1, 1, length = 100), 1, :)
ytrain, ytest = f.(xtrain), f.(xtest)
train_data_loader = DataLoader((; x = xtrain, y = ytrain); batchsize = 64, shuffle = true, partial = false)
eval_data_loader = DataLoader((; x = xtest, y = ytest); batchsize = 10, shuffle = false)

# Create training engine:
#   - `engine` is a reference to the parent `trainer` engine, created below
#   - `batch` is a batch of training data, retrieved by iterating `train_data_loader`
#   - (optional) return value is stored in `trainer.state.output`
function train_step(engine, batch)
    x, y = batch
    l, gs = Zygote.withgradient(m -> sum(abs2, m(x) .- y), model)
    global optim, model = Optimisers.update!(optim, model, gs[1])
    return Dict("loss" => l)
end
trainer = Engine(train_step)

# Start the training
Ignite.run!(trainer, train_data_loader; max_epochs = 25, epoch_length = 100)
````

### Periodically evaluate model

The real power of `Ignite.jl` comes when adding *events* to our training engine.

Let's evaluate our model after every 5th training epoch. This can be easily incorporated without needing to modify any of the above training code:
1. Create an `evaluator` engine which consumes batches of evaluation data
2. Add *event handlers* to the `evaluator` engine which accumulate a running average of evaluation metrics over batches of evaluation data; we use [`OnlineStats.jl`](https://github.com/joshday/OnlineStats.jl) to make this easy.
3. Add an event handler to the `trainer` which runs the `evaluator` on the evaluation data loader every 5 training epochs.

````julia
using OnlineStats: Mean, fit! # for tracking evaluation metrics

# Create an evaluation engine using `do` syntax:
evaluator = Engine() do engine, batch
    x, y = batch
    ypred = model(x) # evaluate model on a single batch of validation data
    return Dict("ytrue" => y, "ypred" => ypred) # result is stored in `evaluator.state.output`
end

# Add events to the evaluation engine to track metrics:
add_event_handler!(evaluator, STARTED()) do engine
    # When `evaluator` starts, initialize the running mean
    engine.state.metrics = Dict("abs_err" => Mean()) # new fields can be added to `engine.state` dynamically
end

add_event_handler!(evaluator, ITERATION_COMPLETED()) do engine
    # Each iteration, compute eval metrics from predictions
    o = engine.state.output
    m = engine.state.metrics["abs_err"]
    fit!(m, abs.(o["ytrue"] .- o["ypred"]) |> vec)
end

# Add an event to `trainer` which runs `evaluator` every 5 epochs:
add_event_handler!(trainer, EPOCH_COMPLETED(every = 5)) do engine
    Ignite.run!(evaluator, eval_data_loader)
    @info "Evaluation metrics: abs_err = $(evaluator.state.metrics["abs_err"])"
end

# Run the trainer with periodic evaluation
Ignite.run!(trainer, train_data_loader; max_epochs = 25, epoch_length = 100)
````

### Artifact saving

Logging artifacts can be easily added to the trainer, again without modifying the above code. For example, save the current model and optimizer state to disk every 10 epochs using [`BSON.jl`](https://github.com/JuliaIO/BSON.jl):

````julia
using BSON: @save

# Save model and optimizer state every 10 epochs
add_event_handler!(trainer, EPOCH_COMPLETED(every = 10)) do engine
    @save "model_and_optim.bson" model optim
    @info "Saved model and optimizer state to disk"
end
````

### Trigger multiple functions per event

Multiple event handlers can be added to the same event:

````julia
add_event_handler!(trainer, COMPLETED()) do engine
    @info "Training is ended"
end
add_event_handler!(trainer, COMPLETED()) do engine
    Ignite.print_timer(engine.timer)
end
````

### Attach the same handler to multiple events

The boolean operators `|` and `&` can be used to combine events together:

````julia
add_event_handler!(trainer, EPOCH_COMPLETED(every = 10) | COMPLETED()) do engine
    # Runs at the end of every 10th epoch, or when training is completed
end

throttled_event = EPOCH_COMPLETED(; every = 3) & EPOCH_COMPLETED(; event_filter = throttle_filter(30.0))
add_event_handler!(trainer, throttled_event) do engine
    # Runs at the end of every 3rd epoch if at least 30s has passed since the last firing
end
````

### Define custom events

Custom events can be created and fired at user-defined stages in the training process.

For example, suppose we want to define events that fire at the start and finish of both the backward pass and the optimizer step. All we need to do is define new event types that subtype `AbstractLoopEvent`, and then fire them at appropriate points in the `train_step` process function using `fire_event!`:

````julia
struct BACKWARD_STARTED <: AbstractLoopEvent end
struct BACKWARD_COMPLETED <: AbstractLoopEvent end
struct OPTIM_STEP_STARTED <: AbstractLoopEvent end
struct OPTIM_STEP_COMPLETED <: AbstractLoopEvent end

function train_step(engine, batch)
    x, y = batch

    # Compute the gradients of the loss with respect to the model
    fire_event!(engine, BACKWARD_STARTED())
    l, gs = Zygote.withgradient(m -> sum(abs2, m(x) .- y), model)
    engine.state.gradients = gs # the engine state can be accessed by event handlers
    fire_event!(engine, BACKWARD_COMPLETED())

    # Update the model's parameters
    fire_event!(engine, OPTIM_STEP_STARTED())
    global optim, model = Optimisers.update!(optim, model, gs[1])
    fire_event!(engine, OPTIM_STEP_COMPLETED())

    return Dict("loss" => l)
end
trainer = Engine(train_step)
````

Then, add event handlers for these custom events as usual:

````julia
add_event_handler!(trainer, BACKWARD_COMPLETED(every = 10)) do engine
    # This code runs after every 10th backward pass is completed
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

