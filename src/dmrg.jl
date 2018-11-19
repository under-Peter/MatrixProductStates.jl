function initenvironments(mps::CanonicalMPS{L,T,TE,TB}, mpo::MPO{L,T}) where {L,T,TE,TB}
    centerlink(mps) == 1 || throw(ArgumentError("centerlink needs to be set to 1"))
    envs = TB[]
    foldr((x,y) -> push!(envs, contractsites(x,y))[end],
        collect(zip(mps[2:L], mpo[2:L], mps[2:L]')), init = ())
    return reverse(envs)
end

function updateenvironments!(envs::Vector{TB},
     mps::CanonicalMPS{L,T,TE,TB}, mpo, i, up) where {L,T,TE,TB}
    centerlink(mps) == i || centerlink(mps) + 1 == i || throw(
        ArgumentError("illegal environment update"))
    if i == 1 #down -> up
        envs[1]   = contractsites((), (mps[1],mpo[1],mps[1]'))
    elseif i == L #up -> down
        envs[L-1] = contractsites((mps[L],mpo[L],mps[L]'),())
    elseif up
        envs[i] = contractsites(envs[i-1], (mps[i], mpo[i], mps[i]'))
    else
        envs[i-1] = contractsites((mps[i], mpo[i], mps[i]'), envs[i])
    end
    return envs
end


function dmrg(ansatz::AbstractMPS{L,T}, mpo::AbstractMPO{L,T};
                tol::Float64 = 1e-6,
                verbose::Bool = false,
                mincounter::Int = 2,
                maxit::Int = 1_000_000) where {L,T}
    mps = normalize!(canonicalize(ansatz,1))
    envs = initenvironments(mps, mpo)
    energies = Float64[]
    envs = initenvironments(mps, mpo)
    counter = 1
    converged = false
    while (!converged || counter < mincounter) && counter <= maxit
        verbose && println("$counter ")
        counter += 1
        up = true
        E = dmrgsweep(mps, mpo, envs, verbose)
        push!(energies,E)

        if length(energies) > 1 && abs(energies[end] - energies[end-1]) < tol
            converged = true
        end
    end
    verbose && println(
    """converged after $(counter-1) passes
    with a ΔE of $(energies[end] - energies[end-1]) in the last step""")

    return (mps, energies)
end

function dmrgsweep(mps::AbstractMPS{L}, mpo, envs, verbose) where L
    E = 0
    for i in 1:L
        verbose && println(">"^(i-1),"o","<"^(L-i))
        E = minimisesite!(mps, mpo, i, envs)
        updateenvironments!(envs, mps, mpo, i, true)
    end
    for i in L-1:-1:2
        verbose && println(">"^(i-1),"o","<"^(L-i))
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
    heffmap = heffmapconst(mpo, envs, Val{i})
    v0 = mps[i]
    res = eigsolve(heffmap, v0, 1, :SR, ishermitian=true)
    return (res[2][1], res[1][1], res[3]) #evec, eval, info
end

function heffmapconst(mpo, envs, ::Type{Val{1}})
    hr = envs[1]
    c = mpo[1]
    return (x -> @tensor x[o2,o3] :=(c[c2,l2,o2] * (hr[c3,l2,o3] * x[c2,c3])))
end

function heffmapconst(mpo, envs, ::Type{Val{i}}) where i
    hl, hr = envs[i-1:i]
    c = mpo[i]
    return (x -> @tensor x[o1,o2,o3] := hl[c1,l1,o1] * (c[c2,l1,l2,o2] * (hr[c3,l2,o3] * x[c1,c2,c3])))
end

function heffmapconst(mpo::AbstractMPO{L}, envs, ::Type{Val{L}}) where L
    hl = envs[L-1]
    c = mpo[L]
    return (x -> @tensor x[o1,o2] := hl[o1,l1,c1] * (c[o2,l1,c2] * x[c1,c2]))
end
