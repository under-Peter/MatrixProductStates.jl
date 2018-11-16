function initenvironments(mps::CanonicalMPS{L,T,TE,TB}, mpo::MPO{L,T}) where {L,T,TE,TB}
    centerlink(mps) == 1 || throw(ArgumentError("centerlink needs to be set to 1"))
    envs = TB[]
    foldr((x,y) -> push!(envs, contractsites(x,y))[end],
                        [zip(mps[2:L], mpo[2:L], mps[2:L]')...],
                        init = ())
    return reverse(envs)
end

function updateenvironments!(environments::Vector{TB},
     mps::CanonicalMPS{L,T,TE,TB}, mpo, site, up) where {L,T,TE,TB}
    centerlink(mps) == site || centerlink(mps) + 1 == site || throw(
        ArgumentError("illegal environment update"))
    if site == 1 #down -> up
        environments[1]   = contractsites((), (mps[1],mpo[1],mps[1]'))
    elseif site == L #up -> down
        environments[L-1] = contractsites((mps[L],mpo[L],mps[L]'),())
    elseif up
        environments[site] =
            contractsites(environments[site-1], (mps[site], mpo[site], mps[site]'))
    else
        environments[site-1] =
            contractsites((mps[site], mpo[site], mps[site]'), environments[site])
    end
    return environments
end


function dmrg(ansatz::AbstractMPS{L,T}, mpo::AbstractMPO{L,T};
                tol::Float64 = 1e-6,
                verbose::Bool = false,
                mincounter::Int = 2,
                maxit::Int = 1_000_000) where {L,T}
    mps = normalize!(canonicalize(ansatz,1))
    environments = initenvironments(mps, mpo)
    energies = Float64[]
    envs = initenvironments(mps, mpo)
    counter = 1
    converged = false
    while (!converged || counter < mincounter) && counter <= maxit
        @show counter
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
    with an energy change of $(energies[end] - energies[end-1])
    between the last two steps""")

    return (mps, energies)
end

function dmrgsweep(mps::AbstractMPS{L}, mpo, envs, verbose) where L
    E = 0
    up = true
    for i in vcat(collect(1:L),collect(L-1:-1:2))
        verbose && println(repeat(">", i-1), "o", repeat("<", L-i))
        E, = minimisesite!(mps, mpo, i, envs)
        i == 1 && (up = true)
        i == L && (up = false)
        !up && canonicalize!(mps,i-1)
        updateenvironments!(envs, mps, mpo, i, up)
    end
    return E
end


function minimisesite!(mps::CanonicalMPS{L}, mpo::AbstractMPO,
        i, environments) where {L}
    canonicalize!(mps, min(i, L-1))
    if centerlink(mps) >= i
        mps[i] = contractsite(mps[i], zerosite(mps))
    else
        mps[i] = contractsite(zerosite(mps), mps[i])
    end

    #smallest realpart eigenvector
    E, mps[i] = minevec(mps, mpo, i, environments)

    #canonical form again
    if i == L
        U, S, mps[L] = onesitesvd2left(mps[L])
        mps.zerosite = contractsite(U, S)
    else
        mps[i], S, V = onesitesvd2right(mps[i])
        mps.zerosite = contractsite(S, V)
    end
    return E, mps
end

function minevec(mps::CanonicalMPS{L}, mpo::AbstractMPO{L}, i, envs) where L
    heffmap = heffmapconst(mpo, envs, Val{i})
    v0 = mps[i]
    res = eigsolve(heffmap, v0, 1, :SR, ishermitian=true)
    return (res[1][1], res[2][1]) #eval, evec
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
