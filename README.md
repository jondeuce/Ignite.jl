# Ignite.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jondeuce.github.io/Ignite.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jondeuce.github.io/Ignite.jl/dev/)
[![Build Status](https://github.com/jondeuce/Ignite.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jondeuce/Ignite.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jondeuce/Ignite.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jondeuce/Ignite.jl)

Welcome to `Ignite.jl`, a Julia port of the Python library [`ignite`](https://github.com/pytorch/ignite) for simplifying neural network training and validation loops using events and handlers.

`Ignite.jl` provides a simple engine and event system. This allows a user to easily compose training pipelines with events such artifact saving, metric logging, and model validation. Event handlers can be any Julia function, and they can be easily configured to run at specific times during training, offering unparalleled flexibility compared to other approaches like callbacks.

Additionally, `Ignite.jl` allows users to define custom events and stack events together to enable multiple calls, giving users even more control over their training process.

## Quick Start

```julia
using Ignite
using Flux, Zygote, Optimisers # for training a neural network
using OnlineStats: Mean, fit! # for tracking evaluation metrics

model = Chain(Dense(1 => 32, tanh), Dense(32 => 1))
optim = Optimisers.setup(Optimisers.Adam(1f-3), model)

# Data loaders can be any iterable
dummy_data(x) = (x, @. 2x-x^3)
train_data_loader = Iterators.cycle([dummy_data(randn(Float32, 1, 10)) for _ in 1:1000]) # iterator can be infinite
eval_data_loader = [dummy_data(randn(Float32, 1, 10)) for _ in 1:10]

# Create training engine:
#   - `engine` is a reference to the parent `trainer` engine, created below
#   - `batch` is a batch of training data, retrieved by iterating `train_data_loader`
#   - (Optional) return value is stored in `trainer.state.output`
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
    @show engine.state
    fit!(m, abs.(o["ytrue"] .- o["ypred"]) |> vec)
end

# Add an event to the training engine to run `evaluator` every 5 epochs:
add_event_handler!(trainer, EPOCH_COMPLETED(every = 5)) do engine
    Ignite.run!(evaluator, eval_data_loader)
    @info "Evaluation metrics: abs_err = $(evaluator.state.metrics["abs_err"])"
end

# Start the training
Ignite.run!(trainer, train_data_loader; max_epochs = 25, epoch_length = 1_000)
```
