var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = Ignite","category":"page"},{"location":"#Ignite.jl","page":"Home","title":"Ignite.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for Ignite.jl.","category":"page"},{"location":"#docstrings","page":"Home","title":"Docstrings","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [Ignite]","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [Ignite]","category":"page"},{"location":"#Ignite.Ignite","page":"Home","title":"Ignite.Ignite","text":"Ignite.jl\n\n(Image: Dev) (Image: Build Status) (Image: Coverage) (Image: Aqua QA)\n\nWelcome to Ignite.jl, a Julia port of the Python library ignite for simplifying neural network training and validation loops using events and handlers.\n\nIgnite.jl provides a simple yet flexible engine and event system, allowing for the easy composition of training pipelines with various events such as artifact saving, metric logging, and model validation. Event-based training abstracts away the training loop, replacing it with:\n\nAn engine which wraps a process function that consumes a single batch of data,\nAn iterable data loader which produces said batches of data, and\nEvents and corresponding event handlers which are attached to the engine, configured to fire at specific points during training.\n\nEvent handlers are much more flexibile compared to other approaches like callbacks: handlers can be any callable; multiple handlers can be attached to a single event; multiple events can trigger the same handler; and custom events can be defined to fire at user-specified points during training. This makes adding functionality to your training pipeline easy, minimizing the need to modify existing code.\n\nQuick Start\n\nThe example below demonstrates how to use Ignite.jl to train a simple neural network. Key features to note:\n\nThe training step is factored out of the training loop: the train_step process function takes a batch of training data and computes the training loss, gradients, and updates the model parameters.\nData loaders can be any iterable collection. Here, we use a DataLoader from MLUtils.jl\n\nusing Ignite\nusing Flux, Zygote, Optimisers, MLUtils # for training a neural network\n\n# Build simple neural network and initialize Adam optimizer\nmodel = Chain(Dense(1 => 32, tanh), Dense(32 => 1))\noptim = Optimisers.setup(Optimisers.Adam(1f-3), model)\n\n# Create mock data and data loaders\nf(x) = 2x-x^3\nxtrain, xtest = 2 * rand(Float32, 1, 10_000) .- 1, collect(reshape(range(-1f0, 1f0, length = 100), 1, :))\nytrain, ytest = f.(xtrain), f.(xtest)\ntrain_data_loader = DataLoader((; x = xtrain, y = ytrain); batchsize = 64, shuffle = true, partial = false)\neval_data_loader = DataLoader((; x = xtest, y = ytest); batchsize = 10, shuffle = false)\n\n# Create training engine:\n#   - `engine` is a reference to the parent `trainer` engine, created below\n#   - `batch` is a batch of training data, retrieved by iterating `train_data_loader`\n#   - (optional) return value is stored in `trainer.state.output`\nfunction train_step(engine, batch)\n    x, y = batch\n    l, gs = Zygote.withgradient(m -> sum(abs2, m(x) .- y), model)\n    global optim, model = Optimisers.update!(optim, model, gs[1])\n    return Dict(\"loss\" => l)\nend\ntrainer = Engine(train_step)\n\n# Start the training\nIgnite.run!(trainer, train_data_loader; max_epochs = 25, epoch_length = 100)\n\nPeriodically evaluate model\n\nThe real power of Ignite.jl comes when adding event handlers to our training engine.\n\nLet's evaluate our model after every 5th training epoch. This can be easily incorporated without needing to modify any of the above training code:\n\nCreate an evaluator engine which consumes batches of evaluation data\nAdd event handlers to the evaluator engine which accumulate a running average of evaluation metrics over batches of evaluation data; we use OnlineStats.jl to make this easy.\nAdd an event handler to the trainer which runs the evaluator on the evaluation data loader every 5 training epochs.\n\nusing OnlineStats: Mean, fit!, value # for tracking evaluation metrics\n\n# Create an evaluation engine using `do` syntax:\nevaluator = Engine() do engine, batch\n    x, y = batch\n    ypred = model(x) # evaluate model on a single batch of validation data\n    return Dict(\"ytrue\" => y, \"ypred\" => ypred) # result is stored in `evaluator.state.output`\nend\n\n# Add event handlers to the evaluation engine to track metrics:\nadd_event_handler!(evaluator, STARTED()) do engine\n    # When `evaluator` starts, initialize the running mean\n    engine.state.metrics = Dict(\"abs_err\" => Mean()) # new fields can be added to `engine.state` dynamically\nend\n\nadd_event_handler!(evaluator, ITERATION_COMPLETED()) do engine\n    # Each iteration, compute eval metrics from predictions\n    o = engine.state.output\n    m = engine.state.metrics[\"abs_err\"]\n    fit!(m, abs.(o[\"ytrue\"] .- o[\"ypred\"]) |> vec)\nend\n\n# Add an event handler to `trainer` which runs `evaluator` every 5 epochs:\nadd_event_handler!(trainer, EPOCH_COMPLETED(every = 5)) do engine\n    Ignite.run!(evaluator, eval_data_loader)\n    @info \"Evaluation metrics: abs_err = $(evaluator.state.metrics[\"abs_err\"])\"\nend\n\n# Run the trainer with periodic evaluation\nIgnite.run!(trainer, train_data_loader; max_epochs = 25, epoch_length = 100)\n\nTerminating a run\n\nThere are several ways to stop a training run before it has completed:\n\nThrow an exception as usual. This will immediately stop training. An EXCEPTION_RAISED() event will be subsequently be fired.\nSimilarly, use a terminal interrupt (CTRL+C, i.e. throw an InterruptException). Training will halt, and an INTERRUPT() event will be fired.\nGracefully terminate via Ignite.terminate!(trainer), or equivalently, trainer.should_terminate = true. This will allow the current iteration to finish, but no further iterations will begin. Then, a TERMINATE() event will be fired followed by a COMPLETED() event.\n\nEarly stopping\n\nTo implement early stopping, we can add an event handler to trainer which checks the evaluation metrics and gracefully terminates trainer if the metrics fail to improve. To do so, we first define a training termination trigger using Flux.early_stopping:\n\n# Callback which returns `true` if the eval loss fails to decrease by\n# at least `min_dist` for two consecutive evaluations\nearly_stop_trigger = Flux.early_stopping(2; init_score = Inf32, min_dist = 5f-3) do\n    return value(evaluator.state.metrics[\"abs_err\"])\nend\n\nThen, we add an event handler to trainer which checks the early stopping trigger and terminates training if the trigger returns true:\n\n# This handler must fire every 5th epoch, the same as the evaluation event handler,\n# to ensure new evaluation metrics are available\nadd_event_handler!(trainer, EPOCH_COMPLETED(every = 5)) do engine\n    if early_stop_trigger()\n        @info \"Stopping early\"\n        Ignite.terminate!(trainer)\n    end\nend\n\n# Run the trainer with periodic evaluation and early stopping\nIgnite.run!(trainer, train_data_loader; max_epochs = 25, epoch_length = 100)\n\nNote: instead of adding a new event, the evaluation event handler from the previous section could have been modified to check early_stop_trigger() immediately after evaluator is run.\n\nArtifact saving\n\nLogging artifacts can be easily added to the trainer, again without modifying the above code. For example, save the current model and optimizer state to disk every 10 epochs using BSON.jl:\n\nusing BSON: @save\n\n# Save model and optimizer state every 10 epochs\nadd_event_handler!(trainer, EPOCH_COMPLETED(every = 10)) do engine\n    @save \"model_and_optim.bson\" model optim\n    @info \"Saved model and optimizer state to disk\"\nend\n\nTrigger multiple functions per event\n\nMultiple event handlers can be added to the same event:\n\nadd_event_handler!(trainer, COMPLETED()) do engine\n    # Runs after training has completed\nend\nadd_event_handler!(trainer, COMPLETED()) do engine\n    # Also runs after training has completed, after the above function runs\nend\n\nAttach the same handler to multiple events\n\nThe boolean operators | and & can be used to combine events together:\n\nadd_event_handler!(trainer, EPOCH_COMPLETED(every = 10) | COMPLETED()) do engine\n    # Runs at the end of every 10th epoch, or when training is completed\nend\n\nthrottled_event = EPOCH_COMPLETED(; every = 3) & EPOCH_COMPLETED(; event_filter = throttle_filter(30.0))\nadd_event_handler!(trainer, throttled_event) do engine\n    # Runs at the end of every 3rd epoch if at least 30s has passed since the last firing\nend\n\nDefine custom events\n\nCustom events can be created and fired at user-defined stages in the training process.\n\nFor example, suppose we want to define events that fire at the start and finish of both the backward pass and the optimizer step. All we need to do is define new event types that subtype AbstractLoopEvent, and then fire them at appropriate points in the train_step process function using fire_event!:\n\nstruct BACKWARD_STARTED <: AbstractLoopEvent end\nstruct BACKWARD_COMPLETED <: AbstractLoopEvent end\nstruct OPTIM_STEP_STARTED <: AbstractLoopEvent end\nstruct OPTIM_STEP_COMPLETED <: AbstractLoopEvent end\n\nfunction train_step(engine, batch)\n    x, y = batch\n\n    # Compute the gradients of the loss with respect to the model\n    fire_event!(engine, BACKWARD_STARTED())\n    l, gs = Zygote.withgradient(m -> sum(abs2, m(x) .- y), model)\n    engine.state.gradients = gs # the engine state can be accessed by event handlers\n    fire_event!(engine, BACKWARD_COMPLETED())\n\n    # Update the model's parameters\n    fire_event!(engine, OPTIM_STEP_STARTED())\n    global optim, model = Optimisers.update!(optim, model, gs[1])\n    fire_event!(engine, OPTIM_STEP_COMPLETED())\n\n    return Dict(\"loss\" => l)\nend\ntrainer = Engine(train_step)\n\nThen, add event handlers for these custom events as usual:\n\nadd_event_handler!(trainer, BACKWARD_COMPLETED(every = 10)) do engine\n    # This code runs after every 10th backward pass is completed\nend\n\n\n\nThis page was generated using Literate.jl.\n\n\n\n\n\n","category":"module"},{"location":"#Ignite.AbstractEvent","page":"Home","title":"Ignite.AbstractEvent","text":"abstract type AbstractEvent\n\nAbstract supertype for all events.\n\n\n\n\n\n","category":"type"},{"location":"#Ignite.AbstractFiringEvent","page":"Home","title":"Ignite.AbstractFiringEvent","text":"abstract type AbstractFiringEvent <: AbstractEvent\n\nAbstract supertype for all events which can trigger event handlers via fire_event!.\n\n\n\n\n\n","category":"type"},{"location":"#Ignite.AbstractLoopEvent","page":"Home","title":"Ignite.AbstractLoopEvent","text":"abstract type AbstractLoopEvent <: AbstractFiringEvent\n\nAbstract supertype for events fired during the normal execution of Ignite.run!.\n\nA default convenience constructor (EVENT::Type{<:AbstractLoopEvent})(; kwargs...) is provided to allow for easy filtering of AbstractLoopEvents. For example, EPOCH_COMPLETED(every = 3) will build a FilteredEvent which is triggered every third epoch. See filter_event for allowed keywords.\n\nBy inheriting from AbstractLoopEvent, custom events will inherit these convenience constructors, too. If this is undesired, one can instead inherit from the supertype AbstractFiringEvent.\n\n\n\n\n\n","category":"type"},{"location":"#Ignite.AndEvent","page":"Home","title":"Ignite.AndEvent","text":"struct AndEvent{E1<:AbstractEvent, E2<:AbstractEvent} <: AbstractEvent\n\nAndEvent(event1, event2) wraps two events and triggers if and only if both wrapped events are triggered by the same firing event firing.\n\nAndEvents can be constructed via the & operator: event1 & event2.\n\nFields: \n\nevent1::AbstractEvent: The first wrapped event that will be considered for triggering.\nevent2::AbstractEvent: The second wrapped event that will be considered for triggering.\n\n\n\n\n\n","category":"type"},{"location":"#Ignite.Engine","page":"Home","title":"Ignite.Engine","text":"mutable struct Engine{P}\n\nAn Engine struct to be run using Ignite.run!. Can be constructed via engine = Engine(process_function; kwargs...), where the process function takes two arguments: the parent engine, and a batch of data.\n\nFields: \n\nprocess_function::Any: A function that processes a single batch of data and returns an output.\nstate::State: An object that holds the current state of the engine.\nevent_handlers::Vector{EventHandler}: A list of event handlers that are called at specific points when the engine is running.\nlogger::Union{Nothing, Base.CoreLogging.AbstractLogger}: An optional logger; if nothing, then current_logger() will be used.\ntimer::TimerOutputs.TimerOutput: Internal timer. Can be used with TimerOutputs to record event timings\nshould_terminate::Bool: A flag that indicates whether the engine should stop running.\nexception::Union{Nothing, Exception}: Exception thrown during training\n\n\n\n\n\n","category":"type"},{"location":"#Ignite.EventHandler","page":"Home","title":"Ignite.EventHandler","text":"struct EventHandler{E<:AbstractEvent, H, A<:Tuple}\n\nEventHandlers wrap an event and a corresponding handler!. The handler! is executed when event is triggered by a call to fire_event!. The output from handler! is ignored. Additional args for handler! may be stored in EventHandler at construction; see add_event_handler!.\n\nWhen h::EventHandler is triggered, the event handler is called as h.handler!(engine::Engine, h.args...).\n\nFields: \n\nevent::AbstractEvent: Event which triggers handler\nhandler!::Any: Event handler which executes when triggered by event\nargs::Tuple: Additional arguments passed to the event handler\n\n\n\n\n\n","category":"type"},{"location":"#Ignite.FilteredEvent","page":"Home","title":"Ignite.FilteredEvent","text":"struct FilteredEvent{E<:AbstractEvent, F} <: AbstractEvent\n\nFilteredEvent(event::E, event_filter::F) wraps an event and a event_filter function.\n\nWhen a firing event e is fired, if event_filter(engine, e) returns true then the filtered event will be fired too.\n\nFields: \n\nevent::AbstractEvent: The wrapped event that will be fired if the filter function returns true when applied to a firing event.\nevent_filter::Any: The filter function (::Engine, ::AbstractFiringEvent) -> Bool returns true if the filtered event should be fired.\n\n\n\n\n\n","category":"type"},{"location":"#Ignite.OrEvent","page":"Home","title":"Ignite.OrEvent","text":"struct OrEvent{E1<:AbstractEvent, E2<:AbstractEvent} <: AbstractEvent\n\nOrEvent(event1, event2) wraps two events and triggers if either of the wrapped events are triggered by a firing event firing.\n\nOrEvents can be constructed via the | operator: event1 | event2.\n\nFields: \n\nevent1::AbstractEvent: The first wrapped event that will be checked if it should be fired.\nevent2::AbstractEvent: The second wrapped event that will be checked if it should be fired.\n\n\n\n\n\n","category":"type"},{"location":"#Ignite.State","page":"Home","title":"Ignite.State","text":"struct State <: AbstractDict{Symbol, Any}\n\nCurrent state of the engine.\n\nState is a light wrapper around a DefaultOrderedDict{Symbol, Any, Nothing} with the following keys:\n\n:iteration: the current iteration, beginning with 1.\n:epoch: the current epoch, beginning with 1.\n:max_epochs: The number of epochs to run.\n:epoch_length: The number of batches processed per epoch.\n:output: The output of process_function after a single iteration.\n:last_event: The last event fired.\n:counters: A DefaultOrderedDict{AbstractFiringEvent, Int, Int}(0) with firing event firing counters.\n:times: An OrderedDict{AbstractFiringEvent, Float64}() with total and per-epoch times fetched on firing event keys.\n\nFields can be accessed and modified using getproperty and setproperty!. For example, engine.state.iteration can be used to access the current iteration, and engine.state.new_field = value can be used to store value for later use e.g. by an event handler.\n\n\n\n\n\n","category":"type"},{"location":"#Ignite.add_event_handler!-Tuple{Any, Engine, AbstractEvent, Vararg{Any, N} where N}","page":"Home","title":"Ignite.add_event_handler!","text":"add_event_handler!(\n    handler!,\n    engine::Engine,\n    event::AbstractEvent,\n    handler_args...\n) -> Engine\n\n\nAdd an event handler to an engine which is fired when event is triggered.\n\nWhen fired, the event handler is called as handler!(engine::Engine, handler_args...).\n\n\n\n\n\n","category":"method"},{"location":"#Ignite.every_filter-Tuple{Union{Int64, AbstractVector{Int64}}}","page":"Home","title":"Ignite.every_filter","text":"every_filter(\n    every::Union{Int64, AbstractVector{Int64}}\n) -> Ignite.EveryFilter{_A} where _A\n\n\nCreates an event filter function for use in a FilteredEvent that returns true periodically depending on every:\n\nIf every = n::Int, the filter will trigger every nth firing of the event.\nIf every = Int[n₁, n₂, ...], the filter will trigger every n₁th firing, every n₂th firing, and so on.\n\n\n\n\n\n","category":"method"},{"location":"#Ignite.filter_event-Tuple{AbstractFiringEvent}","page":"Home","title":"Ignite.filter_event","text":"Filter the input event to fire conditionally:\n\nInputs:\n\nevent::AbstractFiringEvent: event to be filtered.\nevent_filter::Any: A event_filter function (::Engine, ::AbstractFiringEvent) -> Bool returning true if the filtered event should be fired.\nevery::Union{Int, <:AbstractVector{Int}}: the period(s) in which the filtered event should be fired; see every_filter.\nonce::Union{Int, <:AbstractVector{Int}}: the point(s) at which the filtered event should be fired; see once_filter.\n\n\n\n\n\n","category":"method"},{"location":"#Ignite.fire_event!-Tuple{Engine, AbstractFiringEvent}","page":"Home","title":"Ignite.fire_event!","text":"fire_event!(\n    engine::Engine,\n    e::AbstractFiringEvent\n) -> Engine\n\n\nExecute all event handlers triggered by the firing event e.\n\n\n\n\n\n","category":"method"},{"location":"#Ignite.fire_event_handler!-Tuple{Engine, EventHandler, AbstractFiringEvent}","page":"Home","title":"Ignite.fire_event_handler!","text":"fire_event_handler!(\n    engine::Engine,\n    event_handler!::EventHandler,\n    e::AbstractFiringEvent\n) -> Engine\n\n\nExecute event_handler! if it is triggered by the firing event e.\n\n\n\n\n\n","category":"method"},{"location":"#Ignite.once_filter-Tuple{Union{Int64, AbstractVector{Int64}}}","page":"Home","title":"Ignite.once_filter","text":"once_filter(\n    once::Union{Int64, AbstractVector{Int64}}\n) -> Ignite.OnceFilter{_A} where _A\n\n\nCreates an event filter function for use in a FilteredEvent that returns true at specific points depending on once:\n\nIf once = n::Int, the filter will trigger only on the nth firing of the event.\nIf once = Int[n₁, n₂, ...], the filter will trigger only on the n₁th firing, the n₂th firing, and so on.\n\n\n\n\n\n","category":"method"},{"location":"#Ignite.reset!-Tuple{Engine}","page":"Home","title":"Ignite.reset!","text":"reset!(engine::Engine) -> Engine\n\n\nReset the engine state.\n\n\n\n\n\n","category":"method"},{"location":"#Ignite.run!-Tuple{Engine, Any}","page":"Home","title":"Ignite.run!","text":"run!(\n    engine::Engine,\n    dataloader;\n    max_epochs,\n    epoch_length\n) -> Any\n\n\nRun the engine. Data batches are retrieved by iterating dataloader. The data loader may be infinite; by default, it is restarted if it empties.\n\nInputs:\n\nengine::Engine: An instance of the Engine struct containing the process_function to run each iteration.\ndataloader: A data loader to iterate over.\nmax_epochs::Int: the number of epochs to run. Defaults to 1.\nepoch_length::Int: the length of an epoch. If nothing, falls back to length(dataloader).\n\nConceptually, running the engine is roughly equivalent to the following:\n\nThe engine state is initialized.\nThe engine begins running for max_epochs epochs, or until engine.should_terminate == true.\nAt the start of each epoch, EPOCH_STARTED() event is fired.\nAn iteration loop is performed for epoch_length number of iterations, or until engine.should_terminate == true.\nAt the start of each iteration, ITERATION_STARTED() event is fired, and a batch of data is loaded.\nThe process_function is called on the loaded data batch.\nAt the end of each iteration, ITERATION_COMPLETED() event is fired.\nAt the end of each epoch, EPOCH_COMPLETED() event is fired.\nAt the end of all the epochs, COMPLETED() event is fired.\n\nIf engine.should_terminate is set to true while running the engine, the engine will be terminated gracefully after the next completed iteration. This will subsequently trigger a TERMINATE() event to be fired followed by a COMPLETED() event.\n\n\n\n\n\n","category":"method"},{"location":"#Ignite.terminate!-Tuple{Engine}","page":"Home","title":"Ignite.terminate!","text":"terminate!(engine::Engine) -> Engine\n\n\nTerminate the engine by setting engine.should_terminate = true.\n\n\n\n\n\n","category":"method"},{"location":"#Ignite.throttle_filter-Tuple{Real}","page":"Home","title":"Ignite.throttle_filter","text":"throttle_filter(throttle::Real) -> Ignite.ThrottleFilter\n\n\nCreates an event filter function for use in a FilteredEvent that returns true if at least throttle seconds has passed since it was last fired.\n\n\n\n\n\n","category":"method"},{"location":"#Ignite.timeout_filter-Tuple{Real}","page":"Home","title":"Ignite.timeout_filter","text":"timeout_filter(timeout::Real) -> Ignite.TimeoutFilter\n\n\nCreates an event filter function for use in a FilteredEvent that returns true if at least timeout seconds has passed since the filter function was created.\n\n\n\n\n\n","category":"method"}]
}
