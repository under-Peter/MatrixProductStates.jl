function heffmapLapply(A::AbstractTensor{<:Any, N}, L::AbstractTensor{<:Any,3}) where N
    IA = ntuple(i -> ifelse(i == 1, -1, i), N)
    IL = (-1,1,N+1)
    IB = ntuple(identity, N+1)
    B  = tensorcontract(A, IA, L, IL, IB)
    return B
end

function heffmapCapply(A::AbstractTensor{<:Any, N}, C::AbstractTensor{<:Any,4}) where N
    IA = ntuple(i -> ifelse(i == 1, -1, ifelse(i == 2, -2, i-1)), N)
    IC = (-2,-1,1,N)
    IB = ntuple(identity, N)
    B  = tensorcontract(A, IA, C, IC, IB)
    return B
end

function heffmapRapply(A::AbstractTensor{<:Any,N}, R::AbstractTensor{<:Any,3}) where N
    IA = ntuple(i -> ifelse(i == 1, -1, ifelse(i == 2, -2, i-2)), N)
    IR = (-2,-1,N-1)
    IB = ntuple(identity, N-1)
    B  = tensorcontract(A, IA, R, IR, IB)
    return B
end

function heffmapapply(A, heff)
    L, Cs, R = heff[1], heff[2:end-1], heff[end]
    A2 = heffmapLapply(A, L)
    for C in Cs
        A2 = heffmapCapply(A2, C)
    end
    A3 = heffmapRapply(A2, R)
    return A3
end

"""
    heffmap(heff)
returns a function that applies the effective hamiltonian to a tensor.
`heff` is an indexable with entries of type `AbstractTensor` with
`h[1]` being the left-environment or boundary mpo-tensor (rank-3),
`h[end]` being the right-environment or boundary mpo-tensor (rank-3) and
`h[2:end-1]` being the central mpo-tensors (rank-4).
Note that heff needs to have at least 2 elements.
"""
heffmap(heff) = x -> heffmapapply(x, heff)


function heffmapconst(mpo::AbstractMPO{L}, envs, i) where L
    if i == 1
        heff = (mpo[1],envs[1])
    elseif i == L
        heff = (envs[L-1], mpo[L])
    else
        heff = (envs[i-1], mpo[i], envs[i])
    end
    return heffmap(heff)
end
