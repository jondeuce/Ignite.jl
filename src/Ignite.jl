"""
$(README)
"""
module Ignite

using Logging: AbstractLogger, NullLogger, current_logger, global_logger, with_logger
using DataStructures: DefaultOrderedDict, OrderedDict
using DocStringExtensions: README, TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Parameters: @with_kw, @with_kw_noshow

export AbstractEvent, AbstractPrimitiveEvent, AbstractPrimitiveErrorEvent
export STARTED, EPOCH_STARTED, ITERATION_STARTED, GET_BATCH_STARTED, GET_BATCH_COMPLETED, ITERATION_COMPLETED, EPOCH_COMPLETED, COMPLETED
export INTERRUPT, EXCEPTION_RAISED, DATALOADER_STOP_ITERATION, TERMINATE
export State, Engine, EventHandler, FilteredEvent, OrEvent, AndEvent
export add_event_handler!, fire_event!

"""
Abstract event type for construction compound events.
"""
abstract type AbstractEvent end

"""
Basic events fired by the engine.
"""
abstract type AbstractPrimitiveEvent <: AbstractEvent end

"""
Basic events triggered by errors.
"""
abstract type AbstractPrimitiveErrorEvent <: AbstractPrimitiveEvent end

struct STARTED <: AbstractPrimitiveEvent end
struct EPOCH_STARTED <: AbstractPrimitiveEvent end
struct ITERATION_STARTED <: AbstractPrimitiveEvent end
struct GET_BATCH_STARTED <: AbstractPrimitiveEvent end
struct GET_BATCH_COMPLETED <: AbstractPrimitiveEvent end
struct ITERATION_COMPLETED <: AbstractPrimitiveEvent end
struct EPOCH_COMPLETED <: AbstractPrimitiveEvent end
struct COMPLETED <: AbstractPrimitiveEvent end

struct INTERRUPT <: AbstractPrimitiveErrorEvent end
struct EXCEPTION_RAISED <: AbstractPrimitiveErrorEvent end
struct DATALOADER_STOP_ITERATION <: AbstractPrimitiveErrorEvent end
struct TERMINATE <: AbstractPrimitiveErrorEvent end

struct TerminationException <: Exception end
struct DataLoaderEmptyException <: Exception end

"""
    $(TYPEDEF)

`EventHandler(event::E, handler!::H)` wraps an `event` and fires the corresponding `handler!` when `event` is triggered.

Fields: $(TYPEDFIELDS)
"""
@with_kw struct EventHandler{E <: AbstractEvent, H}
    "Event which triggers handler"
    event::E

    "Event handler which executes when triggered by `event`"
    handler!::H
end

"""
    $(TYPEDEF)

Current state of the engine.
Fields can be accessed and modified using `getproperty` and `setproperty!`.

`State` is a light wrapper around a `DefaultOrderedDict{Symbol, Any, Nothing}` with the following keys:
* `:iteration`: the current iteration, beginning with 1.
* `:epoch`: the current epoch, beginning with 1.
* `:dataloader`: The data loader passed to the engine.
* `:max_epochs`: The number of epochs to run.
* `:epoch_length`: The number of batches processed per epoch.
* `:batch`: The current batch passed to `process_function`.
* `:output`: The output of `process_function` after a single iteration.
* `:last_event`: The last event fired.
* `:counters`: A `DefaultOrderedDict{AbstractPrimitiveEvent, Int, Int}(0)` with primitive event firing counters.
* `:times`: An `OrderedDict{AbstractPrimitiveEvent, Float64}()` with total and per-epoch times fetched on primitive event keys.

For example, `engine.state.iteration` can be used to access the current iteration.
"""
@with_kw_noshow struct State
    state::DefaultOrderedDict{Symbol, Any, Nothing} = DefaultOrderedDict{Symbol, Any, Nothing}(
        nothing,
        OrderedDict{Symbol, Any}(
            :iteration    => nothing, # 1-based, the first iteration is 1
            :epoch        => nothing, # 1-based, the first epoch is 1
            :dataloader   => nothing, # data passed to engine
            :max_epochs   => nothing, # number of epochs to run
            :epoch_length => nothing, # number of batches processed per epoch
            :batch        => nothing, # batch passed to `process_function`
            :output       => nothing, # output of `process_function` after a single iteration
            :last_event   => nothing, # last event fired
            :counters     => DefaultOrderedDict{AbstractPrimitiveEvent, Int, Int}(0), # primitive event firing counters
            :times        => OrderedDict{AbstractPrimitiveEvent, Float64}(), # dictionary with total and per-epoch times fetched on primitive event keys
            # :seed       => nothing, # seed to set at each epoch
            # :metrics      => nothing, # dictionary with defined metrics
        ),
    )
end
state(s::State) = getfield(s, :state)

Base.getproperty(s::State, k::Symbol) = getindex(state(s), k)
Base.setproperty!(s::State, k::Symbol, v) = setindex!(state(s), v, k)

Base.show(io::IO, ::State) = print(io, "Engine state")

function Base.show(io::IO, ::MIME"text/plain", s::State)
    println(io, "Engine state:")
    for (k, v) in state(s)
        print(io, "  ", k, ": ")
        (v isa Number || Base.issingletontype(v)) ? show(io, v) : print(io, summary(v))
        println(io, "")
    end
end

"""
    $(TYPEDEF)

`Engine(process_function::P; kwargs...)` to be run; see [`Ignite.run!`](@ref).

Fields: $(TYPEDFIELDS)
"""
@with_kw mutable struct Engine{P}
    "A function that processes a single batch of data and returns an output."
    process_function::P

    "An iterable object that provides the data for the engine to process."
    dataloader::Any = nothing

    "An object that holds the current state of the engine."
    state::State = State()

    "A list of event handlers that are called at specific points when the engine is running."
    event_handlers::Vector{EventHandler} = EventHandler[]

    "An optional logger; if `nothing`, then `current_logger()` will be used."
    logger::Union{Nothing, AbstractLogger} = nothing

    "A flag that indicates whether the engine should stop running."
    should_terminate::Bool = false

    _dataloader_iter::Any = nothing
    # should_terminate_single_epoch::Bool = false
    # should_interrupt::Bool = false
end
Engine(process_function; kwargs...) = Engine(; process_function, kwargs...)

#### Engine methods

function initialize!(engine::Engine, dataloader; max_epochs::Int, epoch_length::Int)
    engine.should_terminate = false
    engine.state.iteration = 0
    engine.state.epoch = 0
    if dataloader !== nothing
        engine.dataloader = dataloader
        engine._dataloader_iter = (Iterators.cycle(dataloader),) # initial args for `iterate`; after first iteration, will be a tuple `(dataloader, state)`
    end
    engine.state.max_epochs = max_epochs
    engine.state.epoch_length = epoch_length
    engine.state.last_event = nothing
    empty!(engine.state.counters)
    empty!(engine.state.times)
    return engine
end

function reset!(engine::Engine)
    engine.state = State()
    return engine
end

function terminate!(engine::Engine)
    engine.should_terminate = true
    throw(TerminationException())
end

check_should_terminate!(engine::Engine) = engine.should_terminate && terminate!(engine)

function load_batch!(engine::Engine)
    check_should_terminate!(engine)

    engine.state.times[GET_BATCH_STARTED()] = time()
    fire_event!(engine, GET_BATCH_STARTED())

    # By default, `dataloader` should be wrapped in `Iterators.cycle` and therefore never finish
    batch_and_state = iterate(engine._dataloader_iter...)
    (batch_and_state === nothing) && throw(DataLoaderEmptyException())

    batch, state = batch_and_state
    engine.state.batch = batch
    engine._dataloader_iter = (engine._dataloader_iter[1], state)

    fire_event!(engine, GET_BATCH_COMPLETED())
    engine.state.times[GET_BATCH_COMPLETED()] = time() - engine.state.times[GET_BATCH_STARTED()]

    return engine
end

function process_function!(engine::Engine)
    check_should_terminate!(engine)

    engine.state.times[ITERATION_STARTED()] = time()
    fire_event!(engine, ITERATION_STARTED())

    engine.state.output = engine.process_function(engine, engine.state.batch)

    fire_event!(engine, ITERATION_COMPLETED())
    engine.state.times[ITERATION_COMPLETED()] = time() - engine.state.times[ITERATION_STARTED()]

    return engine
end

function fire_event!(engine::Engine, e::AbstractPrimitiveEvent)
    !(e isa AbstractPrimitiveErrorEvent) && check_should_terminate!(engine)

    engine.state.last_event = e
    engine.state.counters[e] += 1

    for handler in engine.event_handlers
        fire_event!(engine, handler, e)
    end

    return engine
end

#### EventHandler methods

"""
    $(TYPEDSIGNATURES)

Execute `handler` if it is triggered by the primitive event `e`.
"""
function fire_event!(engine::Engine, handler::EventHandler, e::AbstractPrimitiveEvent)
    if is_triggered_by(engine, handler.event, e)
        handler.handler!(engine)
    end
    return engine
end

"""
    $(TYPEDSIGNATURES)

Add event handler to engine which is fired when `event` is triggered.
"""
function add_event_handler!(handler, engine::Engine, event::AbstractEvent)
    push!(engine.event_handlers, EventHandler(event, handler))
    return engine
end

"""
`@on engine event handler` is syntax sugar for `add_event_handler!(handler, engine, event)`.
"""
macro on(engine, event, handler)
    quote
        add_event_handler!($(esc(handler)), $(esc(engine)), $(esc(event)))
    end
end

function try_length(dataloader)
    try
        return length(dataloader)
    catch e
        @error "`length` is not defined for this dataloader; must set `epoch_length` explicitly"
        throw(e)
    end
end

"""
    $(TYPEDSIGNATURES)

Run the `engine`.
Data batches are retrieved by iterating `dataloader`.
The data loader may be infinite; by default, it is wrapped in `Iterators.cycle` to restart if it empties.

Inputs:
* `engine::Engine`: An instance of the `Engine` struct containing the `process_function` to run each iteration.
* `dataloader`:  A data loader to iterate over. Defaults to `engine.dataloader`.
* `max_epochs::Int`: the number of epochs to run. Defaults to 1.
* `epoch_length::Int`: the length of an epoch. If not given, will try to default to `length(dataloader)`.

Conceptually, running the engine is roughly equivalent to the following:
1. The engine state is initialized.
2. The engine begins running for `max_epochs` epochs, or until the `should_terminate` flag is set to true.
3. At the start of each epoch, `EPOCH_STARTED()` event is fired and time is recorded.
4. An iteration loop is performed for `epoch_length` number of iterations, or until the `should_terminate` flag is set to true.
5. At the start of each iteration, `ITERATION_STARTED()` event is fired, and a batch of data is loaded.
6. The `process_function` is called on the loaded data batch.
7. At the end of each iteration, `ITERATION_COMPLETED()` event is fired.
8. At the end of each epoch, `EPOCH_COMPLETED()` event is fired.
9. At the end of all the epochs, `COMPLETED()` event is fired.
10. Finally, `TERMINATE()` event is fired if `should_terminate` flag is set to true.
"""
function run!(
        engine::Engine,
        dataloader = engine.dataloader;
        max_epochs::Int = 1,
        epoch_length::Int = 0,
    )

    logger = something(engine.logger, current_logger())
    with_logger(logger) do
        if epoch_length <= 0
            epoch_length = try_length(dataloader)
        end

        try
            initialize!(engine, dataloader; max_epochs, epoch_length)

            engine.state.times[STARTED()] = time()
            fire_event!(engine, STARTED())

            while engine.state.epoch < max_epochs && !engine.should_terminate
                engine.state.epoch += 1
                engine.state.times[EPOCH_STARTED()] = time()
                fire_event!(engine, EPOCH_STARTED())

                epoch_iteration = 0
                while epoch_iteration < epoch_length && !engine.should_terminate
                    load_batch!(engine)

                    epoch_iteration += 1
                    engine.state.iteration += 1
                    process_function!(engine)
                end

                fire_event!(engine, EPOCH_COMPLETED())
                engine.state.times[EPOCH_COMPLETED()] = time() - engine.state.times[EPOCH_STARTED()]

                hours, mins, secs = to_hours_mins_secs(engine.state.times[EPOCH_COMPLETED()])
                @info "Epoch[$(engine.state.epoch)] Complete. Time taken: $(hours):$(mins):$(secs)"
            end

            fire_event!(engine, COMPLETED())
            engine.state.times[COMPLETED()] = time() - engine.state.times[STARTED()]

            hours, mins, secs = to_hours_mins_secs(engine.state.times[COMPLETED()])
            @info "Engine run complete. Time taken: $(hours):$(mins):$(secs)"

        catch e
            engine.should_terminate = true
            @error sprint(showerror, e, catch_backtrace())

            if e isa InterruptException
                @info "User interrupt"
                fire_event!(engine, INTERRUPT())

            elseif e isa TerminationException
                @warn "Termination event triggered"

            elseif e isa DataLoaderEmptyException
                @error "Restarting data loader failed: `iterate(dataloader)` returned `nothing`"
                fire_event!(engine, DATALOADER_STOP_ITERATION())

            else
                @error "Exception raised during training"
                fire_event!(engine, EXCEPTION_RAISED())
            end

        finally
            if engine.should_terminate
                fire_event!(engine, TERMINATE())
            end
        end
    end

    return engine
end

#### Helpers

"""
    $(TYPEDSIGNATURES)

Creates a filter function for use in a `FilteredEvent` that returns `true` periodically depending on `every`:
* If `every = n::Int`, the filter will trigger every `n`th firing of the event. 
* If `every = Int[n₁, n₂, ...]`, the filter will trigger every `n₁`th firing, every `n₂`th firing, and so on.
"""
function every_filter(; every::Union{Int, <:AbstractVector{Int}})
    function every_filter_inner(engine::Engine, e::AbstractPrimitiveEvent)
        count = engine.state.counters[e]
        return count > 0 && any(mod1.(count, every) .== every)
    end
end

"""
    $(TYPEDSIGNATURES)

Creates a filter function for use in a `FilteredEvent` that returns `true` at specific points depending on `once`:
* If `once = n::Int`, the filter will trigger only on the `n`th firing of the event. 
* If `once = Int[n₁, n₂, ...]`, the filter will trigger only on the `n₁`th firing, the `n₂`th firing, and so on.
"""
function once_filter(; once::Union{Int, <:AbstractVector{Int}})
    function once_filter_inner(engine::Engine, e::AbstractPrimitiveEvent)
        count = engine.state.counters[e]
        return count > 0 && any(count .== once)
    end
end

function to_hours_mins_secs(time_taken)
    mins, secs = divrem(time_taken, 60.0)
    hours, mins = divrem(mins, 60.0)
    return map(x -> lpad(x, 2, '0'), (round(Int, hours), round(Int, mins), floor(Int, secs)))
end

#### Custom events

function (::Type{EVENT})(; event_filter = nothing, every = nothing, once = nothing) where {EVENT <: AbstractPrimitiveEvent}
    @assert sum(!isnothing, (event_filter, every, once)) == 1 "Exactly one of `event_filter`, `every`, `once` must be supplied"
    filter = every !== nothing ? every_filter(; every) : once !== nothing ? once_filter(; once) : event_filter
    return FilteredEvent(; event = EVENT(), filter = filter)
end

is_triggered_by(::Engine, e1::AbstractEvent, e2::AbstractPrimitiveEvent) = e1 == e2

"""
    $(TYPEDEF)

`FilteredEvent(event::E, filter::F = Returns(true))` wraps an `event` and a `filter` function.

When a primitive event `e` is fired, if `filter(engine, e)` returns `true` then the filtered event will be fired too.

Fields: $(TYPEDFIELDS)
"""
@with_kw struct FilteredEvent{E <: AbstractEvent, F} <: AbstractEvent
    "The wrapped event that will be fired if the filter function returns true when applied to a primitive event."
    event::E

    "The filter function `filter(engine::Engine, e::AbstractPrimitiveEvent)::Bool` returns true if the wrapped `event` should be fired."
    filter::F = Returns(true)
end

is_triggered_by(engine::Engine, e1::FilteredEvent, e2::AbstractPrimitiveEvent) = is_triggered_by(engine, e1.event, e2) && e1.filter(engine, e2)

"""
    $(TYPEDEF)

`OrEvent(event1::E1, event2::E2)` wraps two events and triggers if either of the wrapped events are triggered by a primitive event firing.

Fields: $(TYPEDFIELDS)
"""
@with_kw struct OrEvent{E1 <: AbstractEvent, E2 <: AbstractEvent} <: AbstractEvent
    "The first wrapped event that will be checked if it should be fired."
    event1::E1

    "The second wrapped event that will be checked if it should be fired."
    event2::E2
end
Base.:|(event1::AbstractEvent, event2::AbstractEvent) = OrEvent(event1, event2)

is_triggered_by(engine::Engine, e1::OrEvent, e2::AbstractPrimitiveEvent) = is_triggered_by(engine, e1.event1, e2) || is_triggered_by(engine, e1.event2, e2)

"""
    $(TYPEDEF)

`AndEvent(event1::E1, event2::E2)` wraps two events and triggers if and only if both wrapped events are triggered by the same primitive event firing.

Fields: $(TYPEDFIELDS)
"""
@with_kw struct AndEvent{E1 <: AbstractEvent, E2 <: AbstractEvent} <: AbstractEvent
    "The first wrapped event that will be considered for triggering."
    event1::E1

    "The second wrapped event that will be considered for triggering."
    event2::E2
end
Base.:&(event1::AbstractEvent, event2::AbstractEvent) = AndEvent(event1, event2)

is_triggered_by(engine::Engine, e1::AndEvent, e2::AbstractPrimitiveEvent) = is_triggered_by(engine, e1.event1, e2) && is_triggered_by(engine, e1.event2, e2)

end # module Ignite
