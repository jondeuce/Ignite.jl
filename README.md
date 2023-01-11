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
optim = Optimisers.setup(Optimisers.Adam(), model)

# Data loaders can be any iterable
dummy_data(x) = (x, @. 2x-x^3)
train_data_loader = Iterators.cycle([dummy_data(randn(Float32, 1, 10)) for _ in 1:1000]) # iterator can be infinite
eval_data_loader = [dummy_data(randn(Float32, 1, 10)) for _ in 1:10]

# Create training engine:
function train_step(engine, batch)
    # Process function for training engine:
    #   - This is the training loop body: do forward/backward pass + update models here
    #   - `engine` is a reference to the parent `trainer` engine, created below
    #   - `batch` is a batch of training data, retrieved by iterating `train_data_loader`
    #   - (Optional) return value is stored in `trainer.state.output`
    x, y = batch
    l, gs = Zygote.withgradient(m -> sum(abs, m(x) .- y), model)
    global optim, model = Optimisers.update!(optim, model, gs[1])
    return Dict("loss" => l)
end
trainer = Engine(train_step)

# Create evaluation engine with one call using `do` syntax:
evaluator = Engine() do engine, batch
    x, y = batch # batch of validation input data and corresponding labels
    ypred = model(x) # evaluate model on a single batch of validation data
    return Dict("ytrue" => y, "ypred" => ypred)
end

# Add an event to compute running averages of metrics
add_event_handler!(evaluator, STARTED()) do engine
    engine.state.metrics = Dict("l1" => Mean()) # when the evaluator starts, initialize the running mean
end

add_event_handler!(evaluator, ITERATION_COMPLETED()) do engine
    # Each iteration, compute the l1 losses and update the running average
    o = engine.state.output
    m = engine.state.metrics["l1"]
    fit!(m, abs.(o["ytrue"] .- o["ypred"]) |> vec)
end

# Add an event to the training engine to evaluate the model every 5 epochs:
add_event_handler!(trainer, EPOCH_COMPLETED(every = 5)) do engine
    Ignite.run!(evaluator, eval_data_loader; max_epochs = 1, epoch_length = 10)
    @info "Evaluation metrics: l1 = $(evaluator.state.metrics["l1"])"
end

# Start the training
Ignite.run!(trainer, train_data_loader; max_epochs = 100, epoch_length = 1_000)
```
