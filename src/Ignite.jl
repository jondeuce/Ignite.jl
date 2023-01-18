"""
$(README)
"""
module Ignite

using Logging: AbstractLogger, NullLogger, current_logger, global_logger, with_logger
using DataStructures: DefaultOrderedDict, OrderedDict
using DocStringExtensions: README, TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Parameters: @with_kw, @with_kw_noshow
using TimerOutputs: TimerOutput, @timeit, print_timer, reset_timer!

export AbstractEvent, AbstractFiringEvent, AbstractLoopEvent, AbstractErrorEvent
export STARTED, EPOCH_STARTED, ITERATION_STARTED, GET_BATCH_STARTED, GET_BATCH_COMPLETED, ITERATION_COMPLETED, EPOCH_COMPLETED, COMPLETED
export INTERRUPT, EXCEPTION_RAISED, DATALOADER_STOP_ITERATION, TERMINATE
export State, Engine, EventHandler, FilteredEvent, OrEvent, AndEvent
export filter_event, every_filter, once_filter, throttle_filter
export add_event_handler!, fire_event!

"""
$(TYPEDEF)

Abstract supertype for all events.
"""
abstract type AbstractEvent end

"""
$(TYPEDEF)

Abstract supertype for all events which can trigger event handlers via [`fire_event!`](@ref).
"""
abstract type AbstractFiringEvent <: AbstractEvent end

"""
$(TYPEDEF)

Abstract supertype for events fired during the normal execution of [`Ignite.run!`](@ref).

A default convenience constructor `(EVENT::Type{<:AbstractLoopEvent})(; kwargs...)` is provided to allow for easy filtering of `AbstractLoopEvent`s.
For example, `EPOCH_COMPLETED(every = 3)` will build a [`FilteredEvent`](@ref) which is triggered every third epoch.
See [`filter_event`](@ref) for allowed keywords.

By inheriting from `AbstractLoopEvent`, custom events will inherit these convenience constructors, too.
If this is undesired, one can instead inherit from the supertype `AbstractFiringEvent`.
"""
abstract type AbstractLoopEvent <: AbstractFiringEvent end

(EVENT::Type{<:AbstractLoopEvent})(; kwargs...) = filter_event(EVENT(); kwargs...)

"""
$(TYPEDEF)

Abstract supertype for events triggered by errors during the execution of [`Ignite.run!`](@ref).
"""
abstract type AbstractErrorEvent <: AbstractFiringEvent end

struct STARTED <: AbstractLoopEvent end
struct EPOCH_STARTED <: AbstractLoopEvent end
struct ITERATION_STARTED <: AbstractLoopEvent end
struct GET_BATCH_STARTED <: AbstractLoopEvent end
struct GET_BATCH_COMPLETED <: AbstractLoopEvent end
struct ITERATION_COMPLETED <: AbstractLoopEvent end
struct EPOCH_COMPLETED <: AbstractLoopEvent end
struct COMPLETED <: AbstractLoopEvent end

struct INTERRUPT <: AbstractErrorEvent end
struct EXCEPTION_RAISED <: AbstractErrorEvent end
struct DATALOADER_STOP_ITERATION <: AbstractErrorEvent end
struct TERMINATE <: AbstractErrorEvent end

struct TerminationException <: Exception end
struct DataLoaderEmptyException <: Exception end

"""
$(TYPEDEF)

`EventHandler`s wrap an `event` and corresponding `handler!` which is executed when `event` is triggered by a call to [`fire_event!`](@ref).

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

`State` is a light wrapper around a `DefaultOrderedDict{Symbol, Any, Nothing}` with the following keys:
* `:iteration`: the current iteration, beginning with 1.
* `:epoch`: the current epoch, beginning with 1.
* `:max_epochs`: The number of epochs to run.
* `:epoch_length`: The number of batches processed per epoch.
* `:batch`: The current batch passed to `process_function`.
* `:output`: The output of `process_function` after a single iteration.
* `:last_event`: The last event fired.
* `:counters`: A `DefaultOrderedDict{AbstractFiringEvent, Int, Int}(0)` with firing event firing counters.
* `:times`: An `OrderedDict{AbstractFiringEvent, Float64}()` with total and per-epoch times fetched on firing event keys.

Fields can be accessed and modified using `getproperty` and `setproperty!`.
For example, `engine.state.iteration` can be used to access the current iteration, and `engine.state.new_field = value` can be used to store `value` for later use e.g. by an event handler.
"""
@with_kw_noshow struct State <: AbstractDict{Symbol, Any}
    state::DefaultOrderedDict{Symbol, Any, Nothing} = DefaultOrderedDict{Symbol, Any, Nothing}(
        nothing,
        OrderedDict{Symbol, Any}(
            :iteration    => nothing, # 1-based, the first iteration is 1
            :epoch        => nothing, # 1-based, the first epoch is 1
            :max_epochs   => nothing, # number of epochs to run
            :epoch_length => nothing, # number of batches processed per epoch
            :batch        => nothing, # most recent batch passed to `process_function`
            :output       => nothing, # most recent output of `process_function`
            :last_event   => nothing, # most recent event fired
            :counters     => DefaultOrderedDict{AbstractFiringEvent, Int, Int}(0), # firing event counters
            :times        => OrderedDict{AbstractFiringEvent, Float64}(), # firing event times
            # :seed       => nothing, # seed to set at each epoch
            # :metrics    => nothing, # dictionary with defined metrics
        ),
    )
end
state(s::State) = getfield(s, :state)

Base.getindex(s::State, k::Symbol) = getindex(state(s), k)
Base.setindex!(s::State, v, k::Symbol) = setindex!(state(s), v, k)
Base.getproperty(s::State, k::Symbol) = s[k]
Base.setproperty!(s::State, k::Symbol, v) = (s[k] = v)

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

An `Engine` struct to be run using [`Ignite.run!`](@ref).
Can be constructed via `engine = Engine(process_function; kwargs...)`, where the process function takes two arguments: the parent `engine`, and a batch of data.

Fields: $(TYPEDFIELDS)
"""
@with_kw mutable struct Engine{P}
    "A function that processes a single batch of data and returns an output."
    process_function::P

    "An iterable that provides the data batches for the process function."
    dataloader::Any = nothing

    "An object that holds the current state of the engine."
    state::State = State()

    "A list of event handlers that are called at specific points when the engine is running."
    event_handlers::Vector{EventHandler} = EventHandler[]

    "An optional logger; if `nothing`, then `current_logger()` will be used."
    logger::Union{Nothing, AbstractLogger} = nothing

    "Internal timer. Can be used with `TimerOutputs` to record event timings"
    timer::TimerOutput = TimerOutput()

    "A flag that indicates whether the engine should stop running."
    should_terminate::Bool = false

    # should_terminate_single_epoch::Bool = false
    # should_interrupt::Bool = false
end
Engine(process_function; kwargs...) = Engine(; process_function, kwargs...)

#### Internal data loader cycler

struct DataCycler{D}
    iter::D
end

Base.iterate(dl::DataCycler) = iterate(dl.iter)

function Base.iterate(dl::DataCycler, state)
    batch_and_state = iterate(dl.iter, state)
    batch_and_state === nothing && return iterate(dl)
    return batch_and_state
end

function next_batch_and_state(dl::DataCycler, batch_and_state)
    if batch_and_state === nothing
        batch_and_state = iterate(dl) # first iteration
        batch_and_state === nothing && throw(DataLoaderEmptyException()) # empty iterator
        return batch_and_state
    else
        _, state = batch_and_state
        return iterate(dl, state)
    end
end

function default_epoch_length(dl::DataCycler)
    try
        return length(dl.iter)
    catch e
        @error "`length` is not defined for data loader; must set `epoch_length` explicitly"
        throw(e)
    end
end

#### Engine methods

function initialize!(engine::Engine, dl::DataCycler; max_epochs::Int, epoch_length::Int)
    engine.dataloader = dl.iter
    engine.should_terminate = false
    engine.state.iteration = 0
    engine.state.epoch = 0
    engine.state.max_epochs = max_epochs
    engine.state.epoch_length = epoch_length
    engine.state.last_event = nothing
    empty!(engine.state.counters)
    empty!(engine.state.times)
    return engine
end

"""
    $(TYPEDSIGNATURES)

Reset the engine state.
"""
function reset!(engine::Engine)
    engine.state = State()
    return engine
end

"""
    $(TYPEDSIGNATURES)

Terminate the engine by setting `engine.should_terminate = true` and throwing a `TerminationException`.
"""
function terminate!(engine::Engine)
    engine.should_terminate = true
    throw(TerminationException())
end

"""
    $(TYPEDSIGNATURES)

Terminate the engine if `engine.should_terminate == true`.
See [`terminate!`](@ref).
"""
function maybe_terminate!(engine::Engine)
    if engine.should_terminate
        terminate!(engine)
    end
end

"""
    $(TYPEDSIGNATURES)

Add an event handler to an engine which is fired when `event` is triggered.
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

#### Run engine

function load_batch!(engine::Engine, dl::DataCycler, batch_and_state)
    maybe_terminate!(engine)
    to = engine.timer

    engine.state.times[GET_BATCH_STARTED()] = curr_time()
    @timeit to "Event: GET_BATCH_STARTED" fire_event!(engine, GET_BATCH_STARTED())

    @timeit to "Iterate data loader" begin
        batch, state = next_batch_and_state(dl, batch_and_state)
        engine.state.batch = batch
    end

    @timeit to "Event: GET_BATCH_COMPLETED" fire_event!(engine, GET_BATCH_COMPLETED())
    engine.state.times[GET_BATCH_COMPLETED()] = curr_time() - engine.state.times[GET_BATCH_STARTED()]

    return batch, state
end

function process_function!(engine::Engine, batch)
    maybe_terminate!(engine)
    to = engine.timer

    engine.state.times[ITERATION_STARTED()] = curr_time()
    @timeit to "Event: ITERATION_STARTED" fire_event!(engine, ITERATION_STARTED())

    @timeit to "Process function" output = engine.state.output = engine.process_function(engine, batch)

    @timeit to "Event: ITERATION_COMPLETED" fire_event!(engine, ITERATION_COMPLETED())
    engine.state.times[ITERATION_COMPLETED()] = curr_time() - engine.state.times[ITERATION_STARTED()]

    return output
end

"""
    $(TYPEDSIGNATURES)

Run the `engine`.
Data batches are retrieved by iterating `dataloader`.
The data loader may be infinite; by default, it is restarted if it empties.

Inputs:
* `engine::Engine`: An instance of the `Engine` struct containing the `process_function` to run each iteration.
* `dataloader`: A data loader to iterate over.
* `max_epochs::Int`: the number of epochs to run. Defaults to 1.
* `epoch_length::Int`: the length of an epoch. If `nothing`, falls back to `length(dataloader)`.

Conceptually, running the engine is roughly equivalent to the following:
1. The engine state is initialized.
2. The engine begins running for `max_epochs` epochs, or until `engine.should_terminate == true`.
3. At the start of each epoch, `EPOCH_STARTED()` event is fired.
4. An iteration loop is performed for `epoch_length` number of iterations, or until `engine.should_terminate == true`.
5. At the start of each iteration, `ITERATION_STARTED()` event is fired, and a batch of data is loaded.
6. The `process_function` is called on the loaded data batch.
7. At the end of each iteration, `ITERATION_COMPLETED()` event is fired.
8. At the end of each epoch, `EPOCH_COMPLETED()` event is fired.
9. At the end of all the epochs, `COMPLETED()` event is fired.
10. Finally, `TERMINATE()` event is fired if `engine.should_terminate == true`.
"""
function run!(
        engine::Engine,
        dataloader;
        max_epochs::Int = 1,
        epoch_length::Union{Int, Nothing} = nothing,
    )

    logger = something(engine.logger, current_logger())
    to = engine.timer
    reset_timer!(to)

    @timeit to "Ignite.run!" with_logger(logger) do
        try
            dl = DataCycler(dataloader)
            batch_and_state = nothing
            (epoch_length === nothing) && (epoch_length = default_epoch_length(dl))

            initialize!(engine, dl; max_epochs, epoch_length)

            engine.state.times[STARTED()] = curr_time()
            @timeit to "Event: STARTED" fire_event!(engine, STARTED())

            @timeit to "Epoch loop" while engine.state.epoch < max_epochs && !engine.should_terminate
                engine.state.epoch += 1
                engine.state.times[EPOCH_STARTED()] = curr_time()
                @timeit to "Event: EPOCH_STARTED" fire_event!(engine, EPOCH_STARTED())

                epoch_iteration = 0
                @timeit to "Iteration loop" while epoch_iteration < epoch_length && !engine.should_terminate
                    (batch, _) = batch_and_state = load_batch!(engine, dl, batch_and_state)

                    epoch_iteration += 1
                    engine.state.iteration += 1
                    process_function!(engine, batch)
                end

                @timeit to "Event: EPOCH_COMPLETED" fire_event!(engine, EPOCH_COMPLETED())
                engine.state.times[EPOCH_COMPLETED()] = curr_time() - engine.state.times[EPOCH_STARTED()]

                hours, mins, secs = to_hours_mins_secs(engine.state.times[EPOCH_COMPLETED()])
                @info "Epoch[$(engine.state.epoch)] Complete. Time taken: $(hours):$(mins):$(secs)"
            end

            @timeit to "Event: COMPLETED" fire_event!(engine, COMPLETED())
            engine.state.times[COMPLETED()] = curr_time() - engine.state.times[STARTED()]

            hours, mins, secs = to_hours_mins_secs(engine.state.times[COMPLETED()])
            @info "Engine run complete. Time taken: $(hours):$(mins):$(secs)"

        catch e
            engine.should_terminate = true
            @error sprint(showerror, e, catch_backtrace())

            if e isa InterruptException
                @info "User interrupt"
                @timeit to "Event: INTERRUPT" fire_event!(engine, INTERRUPT())

            elseif e isa TerminationException
                @warn "Termination event triggered"

            elseif e isa DataLoaderEmptyException
                @error "Restarting data loader failed: `iterate(dataloader)` returned `nothing`"
                @timeit to "Event: DATALOADER_STOP_ITERATION" fire_event!(engine, DATALOADER_STOP_ITERATION())

            else
                @error "Exception raised during training"
                @timeit to "Event: EXCEPTION_RAISED" fire_event!(engine, EXCEPTION_RAISED())
            end

            throw(e)

        finally
            if engine.should_terminate
                @timeit to "Event: TERMINATE" fire_event!(engine, TERMINATE())
            end
        end
    end

    return engine
end

#### Event firing

"""
    $(TYPEDSIGNATURES)

Execute all event handlers triggered by the firing event `e`.
"""
function fire_event!(engine::Engine, e::AbstractFiringEvent)
    !(e isa AbstractErrorEvent) && maybe_terminate!(engine)

    engine.state.last_event = e
    engine.state.counters[e] += 1
    fire_event_handlers!(engine, e)

    return engine
end

"""
    $(TYPEDSIGNATURES)

Execute `handler` if it is triggered by the firing event `e`.
"""
function fire_event!(engine::Engine, handler::EventHandler, e::AbstractFiringEvent)
    if is_triggered_by(engine, handler.event, e)
        handler.handler!(engine)
    end
    return engine
end

function fire_event_handlers!(engine::Engine, e::AbstractFiringEvent)
    if length(engine.event_handlers) <= 256
        # Make the common case fast via loop unrolling
        fire_event_handlers_generated!(engine, (engine.event_handlers...,), e)
    else
        # Fallback to a dynamic loop for large numbers of handlers
        fire_event_handlers_loop!(engine, engine.event_handlers, e)
    end
end

@generated function fire_event_handlers_generated!(engine::Engine, handlers::H, e::AbstractFiringEvent) where {N, H <: Tuple{Vararg{EventHandler, N}}}
    quote
        Base.Cartesian.@nexprs $N i -> fire_event!(engine, handlers[i], e)
    end
end

function fire_event_handlers_loop!(engine::Engine, handlers::AbstractVector{EventHandler}, e::AbstractFiringEvent)
    for handler in handlers
        fire_event!(engine, handler, e)
    end
end

#### Helpers

@with_kw struct EveryFilter{T}
    every::T
end
function (f::EveryFilter)(engine::Engine, e::AbstractFiringEvent)
    count = engine.state.counters[e]
    return count > 0 && any(mod1.(count, f.every) .== f.every)
end

@with_kw struct OnceFilter{T}
    once::T
end
function (f::OnceFilter)(engine::Engine, e::AbstractFiringEvent)
    count = engine.state.counters[e]
    return count > 0 && any(count .== f.once)
end

@with_kw struct ThrottleFilter
    throttle::Float64
    last_fire::Base.RefValue{Float64}
end
function (f::ThrottleFilter)(engine::Engine, e::AbstractFiringEvent)
    t = curr_time()
    return t - f.last_fire[] >= f.throttle ? (f.last_fire[] = t; true) : false
end

"""
    $(TYPEDSIGNATURES)

Creates an event filter function for use in a `FilteredEvent` that returns `true` periodically depending on `every`:
* If `every = n::Int`, the filter will trigger every `n`th firing of the event.
* If `every = Int[n₁, n₂, ...]`, the filter will trigger every `n₁`th firing, every `n₂`th firing, and so on.
"""
every_filter(every::Union{Int, <:AbstractVector{Int}}) = EveryFilter(every)

"""
    $(TYPEDSIGNATURES)

Creates an event filter function for use in a `FilteredEvent` that returns `true` at specific points depending on `once`:
* If `once = n::Int`, the filter will trigger only on the `n`th firing of the event.
* If `once = Int[n₁, n₂, ...]`, the filter will trigger only on the `n₁`th firing, the `n₂`th firing, and so on.
"""
once_filter(once::Union{Int, <:AbstractVector{Int}}) = OnceFilter(once)

"""
    $(TYPEDSIGNATURES)

Creates an event filter function for use in a `FilteredEvent` that returns `true` if at least `throttle` seconds has passed since it was last fired.
"""
throttle_filter(throttle::Real) = ThrottleFilter(throttle, Ref(curr_time()))

function to_hours_mins_secs(time_taken)
    mins, secs = divrem(time_taken, 60.0)
    hours, mins = divrem(mins, 60.0)
    return map(x -> lpad(x, 2, '0'), (round(Int, hours), round(Int, mins), floor(Int, secs)))
end

curr_time() = time_ns() / 1e9 # nanosecond resolution

#### Custom events

is_triggered_by(::Engine, e1::AbstractEvent, e2::AbstractFiringEvent) = e1 == e2

"""
$(TYPEDEF)

`FilteredEvent(event::E, event_filter::F)` wraps an `event` and a `event_filter` function.

When a firing event `e` is fired, if `event_filter(engine, e)` returns `true` then the filtered event will be fired too.

Fields: $(TYPEDFIELDS)
"""
@with_kw struct FilteredEvent{E <: AbstractEvent, F} <: AbstractEvent
    "The wrapped event that will be fired if the filter function returns true when applied to a firing event."
    event::E

    "The filter function `(::Engine, ::AbstractFiringEvent) -> Bool` returns `true` if the filtered event should be fired."
    event_filter::F
end

"""
$(TYPEDEF)

Filter the input `event` to fire conditionally:

Inputs:
* `event::AbstractFiringEvent`: event to be filtered.
* `event_filter::Any`: A event_filter function `(::Engine, ::AbstractFiringEvent) -> Bool` returning `true` if the filtered event should be fired.
* `every::Union{Int, <:AbstractVector{Int}}`: the period(s) in which the filtered event should be fired; see [`every_filter`](@ref).
* `once::Union{Int, <:AbstractVector{Int}}`: the point(s) at which the filtered event should be fired; see [`once_filter`](@ref).
"""
function filter_event(event::AbstractFiringEvent; event_filter = nothing, every = nothing, once = nothing)
    @assert sum(!isnothing, (event_filter, every, once)) == 1 "Exactly one of `event_filter`, `every`, or `once` must be supplied."
    if event_filter === nothing
        event_filter = every !== nothing ? every_filter(every) : once_filter(once)
    end
    return FilteredEvent(event, event_filter)
end

is_triggered_by(engine::Engine, e1::FilteredEvent, e2::AbstractFiringEvent) = is_triggered_by(engine, e1.event, e2) && e1.event_filter(engine, e2)

"""
$(TYPEDEF)

`OrEvent(event1, event2)` wraps two events and triggers if either of the wrapped events are triggered by a firing event firing.

`OrEvent`s can be constructed via the `|` operator: `event1 | event2`.

Fields: $(TYPEDFIELDS)
"""
@with_kw struct OrEvent{E1 <: AbstractEvent, E2 <: AbstractEvent} <: AbstractEvent
    "The first wrapped event that will be checked if it should be fired."
    event1::E1

    "The second wrapped event that will be checked if it should be fired."
    event2::E2
end
Base.:|(event1::AbstractEvent, event2::AbstractEvent) = OrEvent(event1, event2)

is_triggered_by(engine::Engine, e1::OrEvent, e2::AbstractFiringEvent) = is_triggered_by(engine, e1.event1, e2) || is_triggered_by(engine, e1.event2, e2)

"""
$(TYPEDEF)

`AndEvent(event1, event2)` wraps two events and triggers if and only if both wrapped events are triggered by the same firing event firing.

`AndEvent`s can be constructed via the `&` operator: `event1 & event2`.

Fields: $(TYPEDFIELDS)
"""
@with_kw struct AndEvent{E1 <: AbstractEvent, E2 <: AbstractEvent} <: AbstractEvent
    "The first wrapped event that will be considered for triggering."
    event1::E1

    "The second wrapped event that will be considered for triggering."
    event2::E2
end
Base.:&(event1::AbstractEvent, event2::AbstractEvent) = AndEvent(event1, event2)

is_triggered_by(engine::Engine, e1::AndEvent, e2::AbstractFiringEvent) = is_triggered_by(engine, e1.event1, e2) && is_triggered_by(engine, e1.event2, e2)

end # module Ignite
