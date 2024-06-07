"""
$(README)
"""
module Ignite

using Logging: AbstractLogger, NullLogger, current_logger, global_logger, with_logger
using DataStructures: DefaultOrderedDict, OrderedDict
using DocStringExtensions: README, TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using TimerOutputs: TimerOutput, @timeit, print_timer, reset_timer!

export AbstractEvent, AbstractFiringEvent, AbstractLoopEvent
export STARTED, EPOCH_STARTED, ITERATION_STARTED, GET_BATCH_STARTED, GET_BATCH_COMPLETED, ITERATION_COMPLETED, EPOCH_COMPLETED, COMPLETED
export INTERRUPT, EXCEPTION_RAISED, DATALOADER_STOP_ITERATION, TERMINATE
export State, Engine, EventHandler, FilteredEvent, OrEvent, AndEvent
export @getsomething!, getsomething!
export filter_event, every_filter, once_filter, throttle_filter, timeout_filter
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
For example, `EPOCH_COMPLETED(every = 3)` will build a [`FilteredEvent`](@ref) which is triggered every third time an `EPOCH_COMPLETED()` event is fired.
See [`filter_event`](@ref) for allowed keywords.

By inheriting from `AbstractLoopEvent`, custom events will inherit these convenience constructors, too.
If this is undesired, one can instead inherit from the supertype `AbstractFiringEvent`.
"""
abstract type AbstractLoopEvent <: AbstractFiringEvent end

(EVENT::Type{<:AbstractLoopEvent})(; kwargs...) = filter_event(EVENT(); kwargs...)

struct EPOCH_STARTED <: AbstractLoopEvent end
struct ITERATION_STARTED <: AbstractLoopEvent end
struct GET_BATCH_STARTED <: AbstractLoopEvent end
struct GET_BATCH_COMPLETED <: AbstractLoopEvent end
struct ITERATION_COMPLETED <: AbstractLoopEvent end
struct EPOCH_COMPLETED <: AbstractLoopEvent end

struct STARTED <: AbstractFiringEvent end
struct TERMINATE <: AbstractFiringEvent end
struct COMPLETED <: AbstractFiringEvent end

struct INTERRUPT <: AbstractFiringEvent end
struct EXCEPTION_RAISED <: AbstractFiringEvent end
struct DATALOADER_STOP_ITERATION <: AbstractFiringEvent end

struct DataLoaderEmptyException <: Exception end
struct DataLoaderUnknownLengthException <: Exception end

"""
$(TYPEDEF)

`EventHandler`s wrap an `event` and a corresponding `handler!`.
The `handler!` is executed when `event` is triggered by a call to [`fire_event!`](@ref).
The output from `handler!` is ignored.
Additional `args` for `handler!` may be stored in `EventHandler` at construction; see [`add_event_handler!`](@ref).

When `h::EventHandler` is triggered, the event handler is called as `h.handler!(engine::Engine, h.args...)`.

Fields: $(TYPEDFIELDS)
"""
Base.@kwdef struct EventHandler{E <: AbstractEvent, H, A <: Tuple}
    "Event which triggers handler"
    event::E

    "Event handler which executes when triggered by `event`"
    handler!::H

    "Additional arguments passed to the event handler"
    args::A
end

"""
$(TYPEDEF)

Current state of the engine.

`State` is a light wrapper around a `DefaultOrderedDict{Symbol, Any, Nothing}` with the following keys:
* `:iteration`: the current iteration, beginning with 1.
* `:epoch`: the current epoch, beginning with 1.
* `:max_epochs`: The number of epochs to run.
* `:epoch_length`: The number of batches processed per epoch.
* `:output`: The output of `process_function` after a single iteration.
* `:last_event`: The last event fired.
* `:counters`: A `DefaultOrderedDict{AbstractFiringEvent, Int, Int}(0)` with firing event firing counters.
* `:times`: An `OrderedDict{AbstractFiringEvent, Float64}()` with total and per-epoch times fetched on firing event keys.

Fields can be accessed and modified using `getproperty` and `setproperty!`.
For example, `engine.state.iteration` can be used to access the current iteration, and `engine.state.new_field = value` can be used to store `value` for later use e.g. by an event handler.
"""
Base.@kwdef struct State <: AbstractDict{Symbol, Any}
    state::DefaultOrderedDict{Symbol, Any, Nothing} = DefaultOrderedDict{Symbol, Any, Nothing}(
        nothing,
        OrderedDict{Symbol, Any}(
            :iteration => nothing, # 1-based, the first iteration is 1
            :epoch => nothing, # 1-based, the first epoch is 1
            :max_epochs => nothing, # number of epochs to run
            :epoch_length => nothing, # number of batches processed per epoch
            :output => nothing, # most recent output of `process_function`
            :last_event => nothing, # most recent event fired
            :counters => DefaultOrderedDict{AbstractFiringEvent, Int, Int}(0), # firing event counters
            :times => OrderedDict{AbstractFiringEvent, Float64}(), # firing event times
            # :seed => nothing, # seed to set at each epoch
            # :metrics => nothing, # dictionary with defined metrics
        ),
    )
end
state(s::State) = getfield(s, :state)

Base.keys(s::State) = keys(state(s))
Base.values(s::State) = values(state(s))
Base.haskey(s::State, k::Symbol) = haskey(state(s), k)
Base.getindex(s::State, k::Symbol) = getindex(state(s), k)
Base.setindex!(s::State, v, k::Symbol) = setindex!(state(s), v, k)
Base.getproperty(s::State, k::Symbol) = s[k]
Base.setproperty!(s::State, k::Symbol, v) = (s[k] = v)
Base.propertynames(s::State) = collect(keys(state(s)))
Base.get(s::State, k::Symbol, v) = get(state(s), k, v)
Base.get!(s::State, k::Symbol, v) = get!(state(s), k, v)

Base.iterate(s::State, args...) = iterate(state(s), args...)
Base.length(s::State) = length(state(s))

"""
    y = @getsomething! engine.state.x = init

Get the field `x` from `engine.state` if it exists and is not `nothing`,
otherwise set it to `init` and return that value.
Equivalent to

```
y = engine.state.x
if y === nothing
    y = engine.state.x = init
end
```

Useful for initializing stateful fields.
For example, to record all losses during training one could do the following:

```
losses = @getsomething! engine.state.losses = Float64[]
push!(losses, loss)
```

See also the functional form [`getsomething!`](@ref).
"""
macro getsomething!(ex)
    ex.head === :(=) || error("Expected assignment expression")
    lhs, rhs = ex.args
    lhs.head === :(.) || error("Expected dot expression")
    f = Expr(:->, :(()), esc(rhs))
    return Expr(:call, getsomething!, f, esc(lhs.args[1]), esc(lhs.args[2]))
end

"""
    $(TYPEDSIGNATURES)

Get the field `key` from the state `s` if it exists and is not `nothing`,
otherwise set it to `f()` and return that value.

Functional form of [`@getsomething!`](@ref).
For example, the following two expressions are equivalent:

```
@getsomething! engine.state.x = init
getsomething!(()->init, engine.state, :x)
```
"""
getsomething!(f, s::State, key) = (val = s[key]) === nothing ? (s[key] = f()) : val

"""
$(TYPEDEF)

An `Engine` struct to be run using [`Ignite.run!`](@ref).
Can be constructed via `engine = Engine(process_function; kwargs...)`, where the process function takes two arguments: the parent `engine`, and a batch of data.

Fields: $(TYPEDFIELDS)
"""
Base.@kwdef mutable struct Engine{P}
    "A function that processes a single batch of data and returns an output."
    process_function::P

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

    "Exception thrown during training"
    exception::Union{Nothing, Exception} = nothing

    # should_terminate_single_epoch::Bool = false
    # should_interrupt::Bool = false
end

Engine(process_function; kwargs...) = Engine(; process_function, kwargs...)

# Copied from Parameters.jl: https://github.com/mauro3/Parameters.jl/blob/e55b025b96275142ba52b2a725aedf460f26ff6f/src/Parameters.jl#L581
Base.show(io::IO, engine::Engine) = dump(IOContext(io, :limit => true), engine; maxdepth = 1)

#### Internal data loader cycler

struct DataCycler{D}
    iter::D
end

function Base.iterate(dl::DataCycler, ::Nothing)
    batch_and_state = iterate(dl.iter)
    (batch_and_state === nothing) && throw(DataLoaderEmptyException()) # empty iterator
    return batch_and_state
end
Base.iterate(dl::DataCycler) = iterate(dl, nothing)

function Base.iterate(dl::DataCycler, iter_state)
    batch_and_state = iterate(dl.iter, iter_state)
    (batch_and_state === nothing) && return iterate(dl) # restart iterator
    return batch_and_state
end

Base.IteratorSize(::DataCycler) = Base.IsInfinite()
Base.IteratorEltype(dl::DataCycler) = Base.IteratorEltype(dl.iter)
Base.eltype(dl::DataCycler) = Base.eltype(dl.iter)

default_epoch_length(dl::DataCycler) = default_epoch_length(dl, Base.IteratorSize(dl.iter))
default_epoch_length(dl::DataCycler, ::Union{Base.HasLength, Base.HasShape}) = length(dl.iter)
default_epoch_length(::DataCycler, ::Base.IteratorSize) = throw(DataLoaderUnknownLengthException())

#### Engine methods

function initialize!(engine::Engine; max_epochs::Int, epoch_length::Int)
    engine.should_terminate = false
    engine.exception = nothing
    engine.state.iteration = 0
    engine.state.epoch = 0
    engine.state.max_epochs = max_epochs
    engine.state.epoch_length = epoch_length
    engine.state.last_event = nothing
    empty!(engine.state.counters)
    empty!(engine.state.times)
    return engine
end

function isdone(engine::Engine)
    iteration = engine.state.iteration
    epoch = engine.state.epoch
    max_epochs = engine.state.max_epochs
    epoch_length = engine.state.epoch_length

    return !any(isnothing, (iteration, epoch, max_epochs, epoch_length)) && ( # counters are initialized
        engine.should_terminate || ( # early termination
            epoch == max_epochs &&
            iteration == epoch_length * max_epochs # regular termination
        )
    )
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

Terminate the engine by setting `engine.should_terminate = true`.
"""
function terminate!(engine::Engine)
    engine.should_terminate = true
    return engine
end

"""
    $(TYPEDSIGNATURES)

Add an event handler to an engine which is fired when `event` is triggered.

When fired, the event handler is called as `handler!(engine::Engine, handler_args...)`.
"""
function add_event_handler!(handler!, engine::Engine, event::AbstractEvent, handler_args...)
    push!(engine.event_handlers, EventHandler(event, handler!, handler_args))
    return engine
end

#### Run engine

function load_batch!(engine::Engine, dl::DataCycler, iter_state)
    to = engine.timer

    engine.state.times[GET_BATCH_STARTED()] = time()
    @timeit to "Event: GET_BATCH_STARTED" fire_event!(engine, GET_BATCH_STARTED())

    @timeit to "Iterate data loader" batch, iter_state = iterate(dl, iter_state)

    @timeit to "Event: GET_BATCH_COMPLETED" fire_event!(engine, GET_BATCH_COMPLETED())
    engine.state.times[GET_BATCH_COMPLETED()] = time() - engine.state.times[GET_BATCH_STARTED()]

    return batch, iter_state
end

function process_function!(engine::Engine, batch)
    to = engine.timer

    engine.state.times[ITERATION_STARTED()] = time()
    @timeit to "Event: ITERATION_STARTED" fire_event!(engine, ITERATION_STARTED())

    @timeit to "Process function" output = engine.state.output = engine.process_function(engine, batch)

    @timeit to "Event: ITERATION_COMPLETED" fire_event!(engine, ITERATION_COMPLETED())
    engine.state.times[ITERATION_COMPLETED()] = time() - engine.state.times[ITERATION_STARTED()]

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

If `engine.should_terminate` is set to `true` while running the engine, the engine will be terminated gracefully after the next completed iteration.
This will subsequently trigger a `TERMINATE()` event to be fired followed by a `COMPLETED()` event.
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
            iter_state = nothing
            (epoch_length === nothing) && (epoch_length = default_epoch_length(dl))

            initialize!(engine; max_epochs, epoch_length)

            engine.state.times[STARTED()] = time()
            @timeit to "Event: STARTED" fire_event!(engine, STARTED())

            @timeit to "Epoch loop" while engine.state.epoch < max_epochs && !engine.should_terminate
                engine.state.epoch += 1
                engine.state.times[EPOCH_STARTED()] = time()
                @timeit to "Event: EPOCH_STARTED" fire_event!(engine, EPOCH_STARTED())

                epoch_iteration = 0
                @timeit to "Iteration loop" while epoch_iteration < epoch_length && !engine.should_terminate
                    batch, iter_state = load_batch!(engine, dl, iter_state)

                    epoch_iteration += 1
                    engine.state.iteration += 1
                    process_function!(engine, batch)
                end
                engine.should_terminate && break

                @timeit to "Event: EPOCH_COMPLETED" fire_event!(engine, EPOCH_COMPLETED())
                engine.state.times[EPOCH_COMPLETED()] = time() - engine.state.times[EPOCH_STARTED()]

                hours, mins, secs = to_hours_mins_secs(engine.state.times[EPOCH_COMPLETED()])
                @info "Epoch[$(engine.state.epoch)] Complete. Time taken: $(hours):$(mins):$(secs)"
            end

            if engine.should_terminate
                @info "Terminating run"
                @timeit to "Event: TERMINATE" fire_event!(engine, TERMINATE())
            end

            @timeit to "Event: COMPLETED" fire_event!(engine, COMPLETED())
            engine.state.times[COMPLETED()] = time() - engine.state.times[STARTED()]

            hours, mins, secs = to_hours_mins_secs(engine.state.times[COMPLETED()])
            @info "Engine run complete. Time taken: $(hours):$(mins):$(secs)"

        catch e
            engine.exception = e

            if e isa InterruptException
                @info "User interrupt"
                @timeit to "Event: INTERRUPT" fire_event!(engine, INTERRUPT())

            elseif e isa DataLoaderEmptyException
                @error "Restarting data loader failed: `iterate(dataloader)` returned `nothing`"
                @timeit to "Event: DATALOADER_STOP_ITERATION" fire_event!(engine, DATALOADER_STOP_ITERATION())

            elseif e isa DataLoaderUnknownLengthException
                @error "Length of data loader iterator is not known; must set `epoch_length` explicitly"

            else
                @error "Exception raised during training"
                @timeit to "Event: EXCEPTION_RAISED" fire_event!(engine, EXCEPTION_RAISED())
            end

            @error sprint(showerror, e, catch_backtrace())

        finally
            return engine
        end
    end
end

#### Event firing

"""
    $(TYPEDSIGNATURES)

Execute all event handlers triggered by the firing event `e`.
"""
function fire_event!(engine::Engine, e::AbstractFiringEvent)
    engine.state.last_event = e
    engine.state.counters[e] += 1

    fire_event_handlers!(engine, e)

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
        Base.Cartesian.@nexprs $N i -> fire_event_handler!(engine, handlers[i], e)
    end
end

function fire_event_handlers_loop!(engine::Engine, handlers::AbstractVector{EventHandler}, e::AbstractFiringEvent)
    for handler in handlers
        fire_event_handler!(engine, handler, e)
    end
end

"""
    $(TYPEDSIGNATURES)

Execute `event_handler!` if it is triggered by the firing event `e`.
"""
function fire_event_handler!(engine::Engine, event_handler!::EventHandler, e::AbstractFiringEvent)
    if is_triggered_by(engine, event_handler!, e)
        fire_event_handler!(engine, event_handler!)
    end
    return engine
end

is_triggered_by(engine::Engine, h::EventHandler, e) = is_triggered_by(engine, h.event, e)

fire_event_handler!(engine::Engine, h::EventHandler) = h.handler!(engine, h.args...)

#### Helpers

Base.@kwdef struct EveryFilter{T}
    every::T
end

function (f::EveryFilter)(engine::Engine, e::AbstractFiringEvent)
    count = engine.state.counters[e]
    return count > 0 && any(mod1.(count, f.every) .== f.every)
end

Base.@kwdef struct OnceFilter{T}
    once::T
end

function (f::OnceFilter)(engine::Engine, e::AbstractFiringEvent)
    count = engine.state.counters[e]
    return count > 0 && any(count .== f.once)
end

Base.@kwdef struct ThrottleFilter
    throttle::Float64
    last_fire::Base.RefValue{Float64}
end

function (f::ThrottleFilter)(engine::Engine, e::AbstractFiringEvent)
    t = time()
    return t - f.last_fire[] >= f.throttle ? (f.last_fire[] = t; true) : false
end

Base.@kwdef struct TimeoutFilter
    timeout::Float64
    start_time::Float64
end

function (f::TimeoutFilter)(engine::Engine, e::AbstractFiringEvent)
    return time() - f.start_time >= f.timeout
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
throttle_filter(throttle::Real, last_fire::Real = -Inf) = ThrottleFilter(throttle, Ref(last_fire))

"""
    $(TYPEDSIGNATURES)

Creates an event filter function for use in a `FilteredEvent` that returns `true` if at least `timeout` seconds has passed since the filter function was created.
"""
timeout_filter(timeout::Real, start_time::Real = time()) = TimeoutFilter(timeout, start_time)

function to_hours_mins_secs(time_taken)
    mins, secs = divrem(time_taken, 60.0)
    hours, mins = divrem(mins, 60.0)
    return map(x -> lpad(x, 2, '0'), (round(Int, hours), round(Int, mins), floor(Int, secs)))
end

#### Custom events

is_triggered_by(::Engine, e1::AbstractEvent, e2::AbstractFiringEvent) = e1 == e2

"""
$(TYPEDEF)

`FilteredEvent(event::E, event_filter::F)` wraps an `event` and a `event_filter` function.

When a firing event `e` is fired, if `event_filter(engine, e)` returns `true` then the filtered event will be fired too.

Fields: $(TYPEDFIELDS)
"""
Base.@kwdef struct FilteredEvent{E <: AbstractEvent, F} <: AbstractEvent
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
Base.@kwdef struct OrEvent{E1 <: AbstractEvent, E2 <: AbstractEvent} <: AbstractEvent
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
Base.@kwdef struct AndEvent{E1 <: AbstractEvent, E2 <: AbstractEvent} <: AbstractEvent
    "The first wrapped event that will be considered for triggering."
    event1::E1

    "The second wrapped event that will be considered for triggering."
    event2::E2
end

Base.:&(event1::AbstractEvent, event2::AbstractEvent) = AndEvent(event1, event2)

is_triggered_by(engine::Engine, e1::AndEvent, e2::AbstractFiringEvent) = is_triggered_by(engine, e1.event1, e2) && is_triggered_by(engine, e1.event2, e2)

end # module Ignite
