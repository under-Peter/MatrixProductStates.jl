leftenv(A::AbstractMPS{L,T,TE,TB}, O::MPO{L,T}, i) where
    {L,T,TE,TB}  = foldl(contractsites, zip(A[1:i], O[1:i], A[1:i]'), init=())

rightenv(A::AbstractMPS{L,T,TE,TB}, O::MPO{L,T}, i) where
    {L,T,TE,TB}  = foldr(contractsites, collect(zip(A[i:L], O[i:L], A[i:L]')), init=())

function tdvp(ansatz::AbstractMPS{L,T}, mpo::AbstractMPO{L,T}, n,t;
        verbose::Bool = false,
        collectall::Bool = false) where {L,T}
    mps = normalize!(canonicalize(ansatz,1))
    collectall && (mpss = Vector{typeof(mps)}())
    envs = initenvironments(mps, mpo)
    Δt = t/n
    for i in 1:n
        verbose && println("step $i \t t = $(i*Δt) ")
        tdvpsweep!(mps, mpo, envs, Δt, verbose)
        collectall && push!(mpss, deepcopy(mps))
    end
    collectall && return ((1:n) .* Δt, mpss)
    return mps
end

function tdvpsweep!(mps::AbstractMPS{L}, mpo, envs, Δt, verbose) where L
    for i in 1:L-1
        verbose && println(">"^(i-1),"o","<"^(L-i))
        onesitepropagate!(mps, mpo, i, envs, Δt/2)
        envsi = deepcopy(envs[i])
        updateenvironments!(envs, mps, mpo, i, true)
        zerositepropagate!(mps, envs[i], envsi, -Δt/2)
    end

    verbose && println(">"^(L-1),"o")
    onesitepropagate!(mps, mpo, L, envs, Δt)
    envsi = deepcopy(envs[L-1])
    updateenvironments!(envs, mps, mpo, L)
    zerositepropagate!(mps, envsi, envs[L-1], -Δt/2)

    for i in L-1:-1:2
        verbose && println(">"^(i-1),"o","<"^(L-i))
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
