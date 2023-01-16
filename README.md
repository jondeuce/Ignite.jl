# Ignite.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jondeuce.github.io/Ignite.jl/dev/)
[![Build Status](https://github.com/jondeuce/Ignite.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/jondeuce/Ignite.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/jondeuce/Ignite.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jondeuce/Ignite.jl)

Welcome to `Ignite.jl`, a Julia port of the Python library [`ignite`](https://github.com/pytorch/ignite) for simplifying neural network training and validation loops using events and handlers.

`Ignite.jl` provides a simple yet flexible engine and event system, allowing for the easy composition of training pipelines with various events such as artifact saving, metric logging, and model validation. Event-based training abstracts away the training loop, replacing it with 1) a training engine which executes a single training step, 2) a data loader for the engine to iterate over, and 3) events and corresponding handlers which are attached to the engine, configured to fire at specific points during training.

Event handlers much more flexibile compared to other approaches like callbacks: they can be any callable, multiple handlers can be attached to a single event, multiple events can trigger the same handler, and custom events can be defined to fire at user-specified points during training. This makes adding functionality to your training pipeline easy, minimizing the need to modify existing code.

## Quick Start

The example below demonstrates how to use `Ignite.jl` to train a simple neural network while keeping track of evaluation metrics and displaying them every 5 epochs. Key features to note:

* The training step is factored out of the training loop: the `train_step` function takes a batch of training data and computes the loss, gradients, and updates the model parameters.
* Events are flexibly added to the training and evaluation engines to customize training: the `evaluator` logs metrics, and the `trainer` runs the evaluator every 5 epochs.
* Data loaders can be any iterable collection. Here we use a [`DataLoader`](https://juliaml.github.io/MLUtils.jl/stable/api/#MLUtils.DataLoader) from [`MLUtils.jl`](https://github.com/JuliaML/MLUtils.jl)

````julia
using Ignite
using Flux, Zygote, Optimisers, MLUtils # for training a neural network
using OnlineStats: Mean, fit! # for tracking evaluation metrics

# Build simple neural network and initialize Adam optimizer
model = Chain(Dense(1 => 32, tanh), Dense(32 => 1))
optim = Optimisers.setup(Optimisers.Adam(1f-3), model)

# Creat mock data and data loaders
f(x) = 2x-x^3
xtrain, xtest = randn(1, 10_000), randn(1, 100)
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

# Create evaluation engine using `do` syntax:
evaluator = Engine() do engine, batch
    x, y = batch
    ypred = model(x) # evaluate model on a single batch of validation data
    return Dict("ytrue" => y, "ypred" => ypred) # result is stored in `evaluator.state.output`
end

# Add events to the evaluation engine to track metrics:
#   - when `evaluator` starts, initialize the running mean
#   - after each iteration, compute eval metrics from predictions and update the running average
add_event_handler!(evaluator, STARTED()) do engine
    engine.state.metrics = Dict("abs_err" => Mean()) # new fields can be dynamically added to `engine.state`
end
add_event_handler!(evaluator, ITERATION_COMPLETED()) do engine
    o = engine.state.output
    m = engine.state.metrics["abs_err"]
    fit!(m, abs.(o["ytrue"] .- o["ypred"]) |> vec)
end

# Add an event to the training engine to run `evaluator` every 5 epochs:
add_event_handler!(trainer, EPOCH_COMPLETED(every = 5)) do engine
    Ignite.run!(evaluator, eval_data_loader)
    @info "Evaluation metrics: abs_err = $(evaluator.state.metrics["abs_err"])"
end

# Start the training
Ignite.run!(trainer, train_data_loader; max_epochs = 25, epoch_length = 1_000)
````

### Periodically save model

Easily add custom functionality to your training process without modifying existing code by incorporating new events. For example, saving the current model and optimizer state to disk every 10 epochs using [`BSON.jl`](https://github.com/JuliaIO/BSON.jl):

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
add_event_handler!(engine -> display(engine.state.times), trainer, COMPLETED())
````

### Attach the same handler to multiple events

The boolean operators `|` and `&` can be used to combine events:

````julia
add_event_handler!(trainer, COMPLETED() | EPOCH_COMPLETED(every = 10)) do engine
    # Runs at the end of every 10th epoch, or when training is completed
end

throttled_event = EPOCH_COMPLETED(; every = 3) & EPOCH_COMPLETED(; event_filter = throttle_filter(30.0))
add_event_handler!(trainer, throttled_event) do engine
    # Runs at the end of every 3rd epoch if at least 30s has passed since the last firing
end
````

### Define custom events

Custom events can be created to track different stages in the training process.

For example, suppose we want to define events that fire at the start and finish of the backward pass and the optimizer step. All we need to do is define new event types that subtype `AbstractPrimitiveEvent`, and then fire them at appropriate points in the `train_step` process function using `fire_event!`:

````julia
struct BACKWARD_STARTED <: AbstractPrimitiveEvent end
struct BACKWARD_COMPLETED <: AbstractPrimitiveEvent end
struct OPTIM_STEP_STARTED <: AbstractPrimitiveEvent end
struct OPTIM_STEP_COMPLETED <: AbstractPrimitiveEvent end

function train_step(engine, batch)
    x, y = batch

    # Compute the gradients of the loss with respect to the model
    fire_event!(engine, BACKWARD_STARTED())
    l, gs = Zygote.withgradient(m -> sum(abs2, m(x) .- y), model)
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

