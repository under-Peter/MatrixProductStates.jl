struct tdvpIterable{TMPS <: AbstractMPS, TMPO <: AbstractMPO,T}
    initmps::TMPS
    mpo::TMPO
    Δt::T
end

struct tdvpState{TMPS <: AbstractMPS, TE <: AbstractTensor}
    mps::TMPS
    envs::Vector{TE}
end

function tdvpiterable(mps::TMPS, mpo::TMPO, Δt::T) where
        {TMPS <: AbstractMPS, TMPO <: AbstractMPO, T}
    cmps = normalize!(canonicalize(mps,1))
    return tdvpIterable{typeof(cmps), TMPO, T}(cmps , mpo, Δt)
end

function iterate(iter::tdvpIterable{TMPS}) where TMPS
    @unpack initmps, mpo = iter
    envs = initenvironments(initmps, mpo)
    TE = typeof(envs[1])
    state = tdvpState{TMPS,TE}(initmps, envs)
    return state, state
end

function iterate(iter::tdvpIterable, state::tdvpState)
    @unpack mps, envs = state
    @unpack mpo, Δt = iter
    tdvpsweep!(mps, mpo, envs, Δt)
    return state, state
end

leftenv(A::AbstractMPS{L,T,TE,TB}, O::MPO{L,T}, i) where
    {L,T,TE,TB}  = foldl(contractsites, zip(A[1:i], O[1:i], A[1:i]'), init=())

rightenv(A::AbstractMPS{L,T,TE,TB}, O::MPO{L,T}, i) where
    {L,T,TE,TB}  = foldr(contractsites, collect(zip(A[i:L], O[i:L], A[i:L]')), init=())

"""
    tdvp(mps::AbstractMPS, mpo::AbstractMPO, n, t, [;verbose=false, period=1, collectall=false])
returns the mps after applying the `mpo` for time `t` where `t` is split into `n` equal timesteps.
If `verbose=true`, information about the execution will be printed every `period` iteration.
If `collectall=true`, a tuple of times and states at those times will be returned.
"""
function tdvp(mps::AbstractMPS{L,T}, mpo::AbstractMPO{L,T}, n,t;
        verbose::Bool = false,
        period::Int = 1,
        collectall::Bool = false) where {L,T}
    Δt = t/n
    disp(state) = @printf("%5d \t| %.3e | %.3e\n",
                        state[2][1]-1,
                        state[1]/1e9,
                        (state[2][1]-1) * Δt)
    iter = tdvpiterable(mps, mpo, Δt)
    if collectall
        mpss = Vector{typeof(iter.initmps)}()
        iter = tee(iter, state -> push!(mpss, deepcopy(state.mps)))
    end
    iter = take(iter, n+1)
    iter = enumerate(iter)

    if verbose
        @printf("\tn \t| time (s)\t| t \n")
        iter = sample(iter, period)
        iter = stopwatch(iter)
        iter = tee(iter, disp)
        (_, (n, state)) = loop(iter)
    else
        (n, state) = loop(iter)
    end
    collectall && return ((1:n) .* Δt, normalize!.(mpss))
    return normalize!(state.mps)
end

function tdvpsweep!(mps::AbstractMPS{L}, mpo, envs, Δt) where L
    for i in 1:L-1
        onesitepropagate!(mps, mpo, i, envs, Δt/2)
        envsi = deepcopy(envs[i])
        updateenvironments!(envs, mps, mpo, i, true)
        zerositepropagate!(mps, envs[i], envsi, -Δt/2)
    end
    onesitepropagate!(mps, mpo, L, envs, Δt)
    envsi = deepcopy(envs[L-1])
    updateenvironments!(envs, mps, mpo, L)
    zerositepropagate!(mps, envsi, envs[L-1], -Δt/2)

    for i in L-1:-1:2
        onesitepropagate!(mps, mpo, i, envs, Δt/2)
        canonicalize!(mps,i-1)
        envsi = deepcopy(envs[i-1])
        updateenvironments!(envs, mps, mpo, i, false)
        zerositepropagate!(mps, envsi, envs[i-1], -Δt/2)
    end
    onesitepropagate!(mps,mpo,1, envs, Δt/2)
    canonicalize!(mps,1)
    return nothing
end


function onesitepropagate!(mps::CanonicalMPS{L}, mpo::AbstractMPO, i, envs, Δt) where L
    canonicalize!(mps, min(i, L-1))
    #absorb zero site
    centerlink(mps) ∈ (i,i-1) || error()
    if centerlink(mps) == i
        mps[i] = contractsite(mps[i], zerosite(mps))
    elseif centerlink(mps) == i-1
        mps[i] = contractsite(zerosite(mps), mps[i])
    end

    #propagate onesite
    heffmapfun = heffmapconst(mpo, envs, i)
    mps[i], info = exponentiate(
        heffmapfun,
        -1im * Δt,
        mps[i],
        ishermitian=true)

    #canonical form again
    if i == L
        U, S, mps[L] = onesitesvd2left(mps[L])
        mps.zerosite = contractsite(U, S)
    else
        mps[i], S, V = onesitesvd2right(mps[i])
        mps.zerosite = contractsite(S, V)
    end
    return nothing
end

function zerositepropagate!(mps::CanonicalMPS{L}, hl, hr, Δt) where {L}
    mps.zerosite, info = exponentiate(
        heffmap((hl,hr)),
        -1im * Δt,
        zerosite(mps),
        ishermitian=true)
    return nothing
end
