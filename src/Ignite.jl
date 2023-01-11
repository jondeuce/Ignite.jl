module Ignite

using Logging: AbstractLogger, NullLogger, current_logger, global_logger, with_logger
using DataStructures: DefaultOrderedDict, OrderedDict
using Parameters: @with_kw, @with_kw_noshow

export STARTED, EPOCH_STARTED, ITERATION_STARTED, GET_BATCH_STARTED, GET_BATCH_COMPLETED, ITERATION_COMPLETED, EPOCH_COMPLETED, COMPLETED, INTERRUPT, EXCEPTION_RAISED, DATALOADER_STOP_ITERATION, TERMINATE
export State, Engine, EventHandler, FilteredEvent, OrEvent, AndEvent
export add_event_handler!

"""Abstract event type for construction compound events"""
abstract type AbstractEvent end

"""Basic events fired by the engine"""
abstract type AbstractPrimitiveEvent <: AbstractEvent end

"""Basic events triggered by errors"""
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
struct DataLoaderException <: Exception end

"""Event handler wraps events and fires the corresponding handler at the appropriate time"""
@with_kw struct EventHandler{E <: AbstractEvent, H}
    event::E
    handler!::H = Returns(nothing)
end
EventHandler(event::AbstractEvent) = EventHandler(; event)

#TODO: this could be a `DefaultOrderedDict`, if we don't care about get/setproperty-style access
"""Engine state. Access and set properties with `getproperty` and `setproperty!`"""
@with_kw_noshow struct State
    state::DefaultOrderedDict{Symbol, Any, Nothing} = DefaultOrderedDict{Symbol, Any, Nothing}(
        nothing,
        OrderedDict{Symbol, Any}(
            :iteration    => nothing, # 1-based, the first iteration is 1
            :epoch        => nothing, # 1-based, the first epoch is 1
            :dataloader   => nothing, # data passed to engine
            :max_epochs   => nothing, # number of epochs to run
            :epoch_length => nothing, # optional length of an epoch
            :batch        => nothing, # batch passed to `process_function`
            :output       => nothing, # output of `process_function` after a single iteration
            :last_event   => nothing, # last event fired
            :counters     => DefaultOrderedDict{AbstractPrimitiveEvent, Int, Int}(0), # primitive event firing counters
            :times        => OrderedDict{AbstractPrimitiveEvent, Float64}(), # dictionary with total and per-epoch times fetched on keys: EPOCH_COMPLETED and COMPLETED
            :metrics      => nothing, # dictionary with defined metrics
            # :seed       => nothing, # seed to set at each epoch
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

"""Training engine"""
@with_kw mutable struct Engine{P}
    process_function::P
    dataloader::Any = nothing
    state::State = State()
    event_handlers::Vector{EventHandler} = EventHandler[]
    should_terminate::Bool = false
    logger::Union{Nothing, AbstractLogger} = nothing
    # should_terminate_single_epoch::Bool = false
    # should_interrupt::Bool = false
end
Engine(process_function; kwargs...) = Engine(; process_function, kwargs...)

#### Engine methods

function initialize!(engine::Engine, dataloader; max_epochs::Int, epoch_length::Int)
    (dataloader !== nothing) && (engine.dataloader = dataloader)
    engine.should_terminate = false
    engine.state.iteration = 0
    engine.state.epoch = 0
    engine.state.dataloader = (dataloader,) # initial args for `iterate`; after first iteration, will be a tuple `(dataloader, state)`
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

    batch_and_state = iterate(engine.state.dataloader...)
    (batch_and_state === nothing) && throw(DataLoaderException())
    batch, state = batch_and_state
    engine.state.batch = batch
    engine.state.dataloader = (engine.state.dataloader[1], state)

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

function fire_event!(engine::Engine, handler::EventHandler, e::AbstractPrimitiveEvent)
    if is_triggered_by(engine, handler.event, e)
        handler.handler!(engine)
    end
    return engine
end

function add_event_handler!(handler, engine::Engine, event::AbstractEvent)
    push!(engine.event_handlers, EventHandler(event, handler))
    return engine
end

function run!(
        engine::Engine,
        dataloader = engine.dataloader;
        max_epochs::Int,
        epoch_length::Int,
    )

    logger = something(engine.logger, current_logger())
    with_logger(logger) do
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

            elseif e isa DataLoaderException
                @error "Dataloader is empty: `iterate(dataloader)` returned `nothing`"
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

function every_filter(; every::Union{Int, <:AbstractVector{Int}})
    function every_filter_inner(engine::Engine, e::AbstractPrimitiveEvent)
        count = engine.state.counters[e]
        return count > 0 && any(mod1.(count, every) .== every)
    end
end

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

@with_kw struct FilteredEvent{E <: AbstractEvent, F} <: AbstractEvent
    event::E
    filter::F = Returns(true)
end

is_triggered_by(engine::Engine, e1::FilteredEvent, e2::AbstractPrimitiveEvent) = is_triggered_by(engine, e1.event, e2) && e1.filter(engine, e2)

@with_kw struct OrEvent{E1 <: AbstractEvent, E2 <: AbstractEvent} <: AbstractEvent
    event1::E1
    event2::E2
end
Base.:|(event1::AbstractEvent, event2::AbstractEvent) = OrEvent(event1, event2)

is_triggered_by(engine::Engine, e1::OrEvent, e2::AbstractPrimitiveEvent) = is_triggered_by(engine, e1.event1, e2) || is_triggered_by(engine, e1.event2, e2)

@with_kw struct AndEvent{E1 <: AbstractEvent, E2 <: AbstractEvent} <: AbstractEvent
    event1::E1
    event2::E2
end
Base.:&(event1::AbstractEvent, event2::AbstractEvent) = AndEvent(event1, event2)

is_triggered_by(engine::Engine, e1::AndEvent, e2::AbstractPrimitiveEvent) = is_triggered_by(engine, e1.event1, e2) && is_triggered_by(engine, e1.event2, e2)

end # module Ignite
