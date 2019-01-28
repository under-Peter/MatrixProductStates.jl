struct dmrgIterable{TMPS <: AbstractMPS, TMPO <: AbstractMPO}
    initmps::TMPS
    mpo::TMPO
end

struct dmrgState{TMPS <: AbstractMPS, TE <: AbstractTensor}
    mps::TMPS
    envs::Vector{TE}
    energies::Vector{Float64}
end

function dmrgiterable(mps::TMPS, mpo::TMPO) where {TMPS <: AbstractMPS, TMPO <: AbstractMPO}
    cmps = normalize!(canonicalize(mps,1))
    return dmrgIterable{typeof(cmps), TMPO}(cmps , mpo)
end

function iterate(iter::dmrgIterable{TMPS}) where TMPS
    @unpack initmps, mpo = iter
    envs = initenvironments(initmps, mpo)
    TE = typeof(envs[1])
    state = dmrgState{TMPS,TE}(initmps, envs, [real(inner(initmps,mpo,initmps))])
    return state, state
end

function iterate(iter::dmrgIterable, state::dmrgState)
    @unpack mps, envs, energies = state
    @unpack mpo = iter
    E = dmrgsweep!(mps, mpo, envs)
    push!(energies, real(E))
    return state, state
end


function initenvironments(mps::CanonicalMPS{L,T,TE,TB}, mpo::MPO{L,T}) where {L,T,TE,TB}
    centerlink(mps) == 1 || throw(ArgumentError("centerlink needs to be set to 1"))
    envs = TB[]
    foldr((x,y) -> push!(envs, contractsites(x,y))[end],
        collect(zip(mps[2:L], mpo[2:L], mps[2:L]')), init = ())
    return reverse(envs)
end

function updateenvironments!(envs::Vector{TB},
     mps::CanonicalMPS{L,T,TE,TB}, mpo, i, up = true) where {L,T,TE,TB}
    centerlink(mps) == i || centerlink(mps) + 1 == i || throw(
        ArgumentError("illegal environment update"))
    if i == 1 #down -> up
        envs[1]   = contractsites((), (mps[1],mpo[1],mps[1]'))
    elseif i == L #up -> down
        envs[L-1] = contractsites((mps[L],mpo[L],mps[L]'),())
    elseif up
        envs[i]   = contractsites(envs[i-1], (mps[i], mpo[i], mps[i]'))
    else
        envs[i-1] = contractsites((mps[i], mpo[i], mps[i]'), envs[i])
    end
    return envs
end


function dmrg(ansatz::AbstractMPS{L,T}, mpo::AbstractMPO{L,T};
                tol::Float64 = 1e-6,
                verbose::Bool = false,
                mincounter::Int = 2,
                period::Int = 1,
                maxit::Int = 1_000_000,) where {L,T}
    stop(state) = length(state.energies) > max(1, mincounter) &&
                    abs(state.energies[end] - state.energies[end-1]) < tol
    disp(state) = @printf("%5d \t| %.3e | %.3e\n", state[2][1]-1, state[1]/1e9, state[2][2].energies[end])

    iter = dmrgiterable(ansatz, mpo)
    tol > 0 && (iter = halt(iter, stop) )
    iter = take(iter, maxit)
    iter = enumerate(iter)

    if verbose
        @printf("\tn \t| time (s)\t| E \n")
        iter = sample(iter, period)
        iter = stopwatch(iter)
        iter = tee(iter, disp)
        (_, (n, state)) = loop(iter)
    else
        (n, state) = loop(iter)
    end
    # return (n, state)
    return (state.mps, state.energies[2:end])
end

function dmrgsweep!(mps::AbstractMPS{L}, mpo, envs) where L
    E = 0
    for i in 1:L
        E = minimisesite!(mps, mpo, i, envs)
        updateenvironments!(envs, mps, mpo, i, true)
    end
    for i in L-1:-1:2
        E = minimisesite!(mps, mpo, i, envs)
        canonicalize!(mps,i-1)
        updateenvironments!(envs, mps, mpo, i, false)
    end
    return E
end


function minimisesite!(mps::CanonicalMPS{L}, mpo::AbstractMPO, i, envs) where {L}
    canonicalize!(mps, min(i, L-1))
    #absorb zero site
    if centerlink(mps) >= i
        mps[i] = contractsite(mps[i], zerosite(mps))
    else
        mps[i] = contractsite(zerosite(mps), mps[i])
    end

    #smallest realpart eigenvector
    mps[i], E, info = minevec(mps, mpo, i, envs)

    #canonical form again
    if i == L
        U, S, mps[L] = onesitesvd2left(mps[L])
        mps.zerosite = contractsite(U, S)
    else
        mps[i], S, V = onesitesvd2right(mps[i])
        mps.zerosite = contractsite(S, V)
    end
    return E
end

function minevec(mps::CanonicalMPS{L}, mpo::AbstractMPO{L}, i, envs) where L
    heffmapfun = heffmapconst(mpo, envs, i)
    v0 = mps[i]
    res = eigsolve(heffmapfun, v0, 1, :SR, ishermitian=true)
    return (res[2][1], res[1][1], res[3]) #evec, eval, info
end
