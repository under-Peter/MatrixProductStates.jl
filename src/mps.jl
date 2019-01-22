mutable struct MPS{L, T, TE <: AbstractTensor{T,2}, TB <: AbstractTensor{T,3}} <: AbstractMPS{L,T,TE,TB}
    #= MPS
      (1)                 (2)                (2)                 (2)
       ↑         …         ↑                  ↑          …        ↑
    [ledge] →(2)   (1)→[bulk(i)]→(3) (1)→[bulk(i+1)]→(3)   (1)→[redge]
     =#
    sites::Vector{Union{TE,TB}}
    MPS(a::TE, b::Vector{TB}, c::TE) where {T, TE <: AbstractTensor{T,2},
        TB <: AbstractTensor{T,3}} = new{length(b)+2,T,TE,TB}([a; b; c])
end

Base.getindex(mps::AbstractMPS{L}, i) where L = mps.sites[i]
Base.setindex!(mps::AbstractMPS{L}, tensor, i) where {L} = setindex!(mps.sites, tensor, i)
Base.length(::AbstractMPS{L}) where L = L
Base.adjoint(mps::MPS{L,T,TE,TB}) where {L,T,TE,TB} =
    MPS(mps[1]', adjoint.(convert(Array{TB},mps[2:L-1])), mps[L]')


mutable struct CanonicalMPS{L, T, TE <: AbstractTensor{T,2}, TB <: AbstractTensor{T,3}} <: AbstractMPS{L,T,TE,TB}
    #= CMPS with centerlink = i
      (1)                 (2)                               (2)                 (2)
       ↑         …         ↑                                 ↑          …        ↑
    [ledge] →(2)   (1)→[bulk(i)]→(3)  (1)→[zs]→(2)  (1)→[bulk(i+1)]→(3)   (1)→[redge]
    =#
    sites::Vector{Union{TB,TE}}
    centerlink::Int
    zerosite::TE
    CanonicalMPS(a::TE, b::Vector{TB}, c::TE, cl::Int, zs::TE) where {T, TE <: AbstractTensor{T,2},
        TB <: AbstractTensor{T,3}} = new{length(b)+2,T,TE,TB}([a; b; c], cl, zs)
end

@inline centerlink(mps::CanonicalMPS) = mps.centerlink
@inline zerosite(mps::CanonicalMPS) = mps.zerosite
Base.adjoint(cmps::CanonicalMPS{L,T,TE,TB}) where {L,T,TE,TB} =
    CanonicalMPS(cmps[1]', adjoint.(convert(Array{TB},cmps[2:L-1])), cmps[L]', centerlink(cmps), zerosite(cmps)')

mutable struct MPO{L, T, TE <: AbstractTensor{T,3}, TB <: AbstractTensor{T,4}} <: AbstractMPO{L,T,TE,TB}
    #=MPO
    (3)          (4)         (3)
     ↑            ↑           ↑
    [lb]→(2) (2)→[b]→(3) (2)→[rb]
     ↑            ↑           ↑
    (1)          (1)         (1)
    =#
    sites::Vector{Union{TE,TB}}
    MPO{L}(a::TE, b::TB, c::TE) where {T, TE <: AbstractTensor{T,3},
        TB <: AbstractTensor{T,4},L} = new{L,T,TE,TB}([a; fill(b ,L-2); c])
end

@inline Base.getindex(mpo::AbstractMPO{L},i) where L = mpo.sites[i]
@inline Base.setindex!(mpo::AbstractMPO{L}, tensor, i) where L =
    setindex!(mpo.sites, tensor, i)


function TNTensors.todense(mps::MPS{L,T,TE,TB}) where {L,T,TE,TB}
    le = mps[1]
    re = mps[L]
    blk = mps[2:L-1]
    return MPS( todense(le), map(todense,blk), todense(re))
end

function TNTensors.todense(mps::MPO{L,T,TE,TB}) where {L,T,TE,TB}
    le = mps[1]
    re = mps[L]
    blk = mps[2]
    return MPO{L}( todense(le), todense(blk), todense(re))
end

TNTensors.charge(mps::MPS{L}) where L = sum(charge.(mps[1:L]))
TNTensors.charge(mps::CanonicalMPS{L}) where L = sum(charge.(mps[1:L])) + charge(zerosite(mps))

randmps(N, T::Type, (d, χ)::Tuple{Int,Int}) =  randmps(N, DTensor{T}((χ,d,χ)))
function randmps(N, T::Type, (sym, (pchs, vchs), (pds, vds)))
    a = DASTensor{T,3}(sym, (vchs, pchs, vchs), (vds, pds, vds), InOut(-1,-1,1))
    randmps(N, a)
end

function randmps(N, A::AbstractTensor{T,3}) where T
    if A isa DASTensor
        in_out(A) == InOut(-1,-1,1) || throw(
            ArgumentError("DASTensor for MPS needs InOut(-1,-1,1)"))
    end
    bulk = [similar(A) for i in 2:N-1]
    lb = checked_similar_from_indices(nothing, T, (2,3) , (), A)
    rb = checked_similar_from_indices(nothing, T, (1,2) , (), A)
    initwithrand!.(bulk)
    initwithrand!(lb)
    initwithrand!(rb)
    return MPS(lb, bulk, rb)
end


#= mpscontract to scalar =#
function contractsites(A::T, B::T) where T <: AbstractTensor{<:Any,3}
    @tensor res = A[1,2,3] * B[1,2,3]
end

function contractsites(A::T, B::T) where T <: AbstractTensor{<:Any,2}
    @tensor res = A[1,2] * B[1,2]
end

function contractsites((B,C)::Tuple{AbstractTensor{T,2},AbstractTensor{T,2}},
    A::AbstractTensor{T,2}) where T
    @tensor res = A[1,2] * B[-1,1] * C[-1, 2]
end

function contractsites(A::AbstractTensor{T,2},
    (B,C)::Tuple{AbstractTensor{T,2},AbstractTensor{T,2}}) where T
    @tensor res = A[1,2] * B[1,-1] * C[2, -1]
end

function contractsites((B,C,D)::Tuple{AbstractTensor{T,2},AbstractTensor{T,3},AbstractTensor{T,2}},
    A::AbstractTensor{T,3}) where T
    @tensor res = A[1,2,3] * B[-1,1] * C[-1,2,-2] * D[-2,3]
end

function contractsites(A::AbstractTensor{T,3},
    (B,C,D)::Tuple{AbstractTensor{T,2},AbstractTensor{T,3},AbstractTensor{T,2}}) where T
    @tensor res = A[1,2,3] * B[1,-1] * C[-1,2,-2] * D[3,-2]
end

#= mpscontract to grow edges (rank == 2 || rank == 3) =#
# leftedge rank 3
function contractsites(A::AbstractTensor{T,3},
     (C,D,E)::Tuple{AbstractTensor{T,3}, AbstractTensor{T,4}, AbstractTensor{T,3}}) where T
    @tensor res[1,2,3] := A[-1,-2,-3] * C[-1,-4,1] * D[-4,-2,2,-5] * E[-3,-5,3]
    return res
end

# leftedge rank 2
function contractsites(A::AbstractTensor{T,2},
     (C,D)::Tuple{AbstractTensor{T,3}, AbstractTensor{T,3}}) where T
    @tensor res[1,2] := A[-1,-2] * C[-1,-3,1] * D[-2,-3,2]
    return res
end

#rightedge rank 3
function contractsites((C,D,E)::Tuple{AbstractTensor{T,3}, AbstractTensor{T,4}, AbstractTensor{T,3}},
     A::AbstractTensor{T,3}) where T
    @tensor res[1,2,3] := A[-1,-2,-3] * C[1,-4, -1] * D[-4,2,-2,-5] * E[3,-5,-3]
    return res
end

#rightedge rank 2
function contractsites((C,D)::Tuple{AbstractTensor{T,3}, AbstractTensor{T,3}},
     A::AbstractTensor{T,2}) where T
    @tensor res[1,2] := A[-1,-2] * C[1,-3,-1] * D[2,-3,-2]
    return res
end

#= mpscontract to initialize edges (rank == 2 || rank == 3) =#

#left edge, rank 2
function contractsites(::Tuple{},(A,B)::Tuple{AbstractTensor{T,2},AbstractTensor{T,2}}) where T
    @tensor res[1,2] := A[-1,1] * B[-1,2]
    return res
end
#left edge, rank 3
function contractsites(::Tuple{},(A,B,C)::Tuple{AbstractTensor{T,2},AbstractTensor{T,3},AbstractTensor{T,2}}) where T
    @tensor res[1,2,3] := A[-1,1] * B[-1,2,-2] * C[-2,3]
end
#right edge, rank 2
function contractsites((A,B)::Tuple{AbstractTensor{T,2},AbstractTensor{T,2}},::Tuple{}) where T
    @tensor res[1,2] := A[1,-1] * B[2,-1]
end
#right edge, rank 3
function contractsites((A,B,C)::Tuple{AbstractTensor{T,2},AbstractTensor{T,3},AbstractTensor{T,2}},::Tuple{}) where T
    @tensor res[1,2,3] := A[1,-1] * B[-1,2,-2] * C[3,-2]
end

#= mpscontract to contract neighboring sites =#
#rank 3 with rank 2 on the right
contractsite(A::AbstractTensor{T,3},Bs::AbstractTensor{T,2}...) where T =
    contractsite(A, contractsite(Bs...))

function contractsite(A::AbstractTensor{T,3},B::AbstractTensor{T,2}) where T
    @tensor res[1,2,3] := A[1,2,-1] * B[-1,3]
end

#rank 3 with rank 2 on the left
contractsite(A::AbstractTensor{T,2}, B::AbstractTensor{T,2}, C::AbstractTensor{T,3}) where
    T = contractsite(contractsite(A,B),C)

function contractsite(A::AbstractTensor{T,2}, B::AbstractTensor{T,3}) where T
    @tensor res[1,2,3] := A[1,-1] * B[-1,2,3]
end

#rank 2 with rank 2 on the left
contractsite(A::AbstractTensor{T,2}, Bs::AbstractTensor{T,2}...) where T =
    foldl(contractsite, Bs, init=A)

function contractsite(A::AbstractTensor{T,2},B::AbstractTensor{T,2}) where T
    @tensor res[1,2] := A[1,-1] * B[-1,2]
end


inner(A) = inner(A,A)

inner(A::MPS{L,T,TE,TB}, B::MPS{L,T,TE,TB}) where {L,T,TE,TB} =
    foldl(contractsites, zip(A[:], B[:]'), init=())

inner(A::MPS{L,T,TE,TB}, O::MPO{L,T}, B::MPS{L,T,TE,TB}) where
    {L,T,TE,TB}  = foldl(contractsites, zip(A[:], O[1:L], B[:]'), init=())
inner(A::CanonicalMPS{L,T,TE,TB}, O::MPO{L,T}, B::MPS{L,T,TE,TB}) where
    {L,T,TE,TB}  = foldl(contractsites, zip(convert(MPS,A)[:], O[1:L], B[:]'), init=())
inner(A::CanonicalMPS{L,T,TE,TB}, O::MPO{L,T}, B::CanonicalMPS{L,T,TE,TB}) where
    {L,T,TE,TB}  = foldl(contractsites, zip(convert(MPS,A)[:], O[1:L], convert(MPS,B)[:]'), init=())
inner(A::MPS{L,T,TE,TB}, O::MPO{L,T}, B::CanonicalMPS{L,T,TE,TB}) where
    {L,T,TE,TB}  = foldl(contractsites, zip(A[:], O[1:L], convert(MPS,B)[:]'), init=())


function Base.convert(::Type{MPS}, A::CanonicalMPS{L,T,TE,TB}, left = true) where {L,T,TE,TB}
    l::TE = A[1]
    b::Vector{TB} = A[2:L-1]
    r::TE = A[L]
    mps = MPS(l, b, r)
    cl = centerlink(A)
    z  = zerosite(A)
    if left
        mps[cl]   = contractsite(mps[cl], z)
    else
        mps[cl+1] = contractsite(z, mps[cl+1])
    end
    return mps
end

inner(A::CanonicalMPS{L,T}, B::MPS{L,T}) where {L,T} = inner(convert(MPS, A), B)
inner(A::MPS{L,T}, B::CanonicalMPS{L,T}) where {L,T} = inner(B, A)

inner(A::CanonicalMPS{L,T}, B::MPO{L,T}, C::MPS{L,T}) where {L,T} = inner(convert(MPS, A), B, C)
inner(A::MPS{L,T}, B::MPO{L,T}, C::CanonicalMPS{L,T}) where {L,T} = inner(A, B, convert(MPS, C))

function inner(A::CanonicalMPS{L,T}, B::CanonicalMPS{L,T}) where {L,T}
    if A == B
        zs = zerosite(A)
        @tensor res[] := zs[1, 2] * zs'[1, 2]
        return scalar(res)
    else
        return inner(convert(MPS, A), convert(MPS, B))
    end
end


function onesitesvd2right(A::AbstractTensor{T,3}) where T
    #    ↑          ↑
    #  →[A]→  ⇒ ( →[U]→ , →[S]→, →[V]→ )
    return tensorsvd(A,((1,2),(3,)))
end

onesitesvd2right(A::AbstractTensor{<:Any,2}) = tensorsvd(A)

onesitesvd2left(A::AbstractTensor{<:Any,3}) = tensorsvd(A, ((1,),(2,3)))

onesitesvd2left(A::AbstractTensor{T,2}) where T = tensorsvd(A)

function leftcanonicalize!(mps::MPS{L,T}, uptolink::Int) where {L,T}
    #=
    IN:
    ↑ ↑ ↑ ↑ ↑ ↑ ↑
    o-o-o-o-o-o-o, e.g uptolink = 3

    OUT:
    ↑ ↑ ↑ ↑ ↑ ↑ ↑
    >->->-ø-o-o-o where > are leftcanonical
                        ø is changed
    =#
    uptolink < 1 && return mps
    uptolink > L-1 && throw( ArgumentError(
    "can't bring in to leftcanonical-form up to link $uptolink with only $(L-1) links"))

    for i in 1:uptolink
        mps[i], Sm, Vm = onesitesvd2right(mps[i])
        mps[i+1] = contractsite(Sm,Vm,mps[i+1])
    end

    return mps
end

function rightcanonicalize!(mps::AbstractMPS{L,T}, downtolink::Int) where {L,T}
    #=
    IN:
    ↑ ↑ ↑ ↑ ↑ ↑ ↑
    o-o-o-o-o-o-o, e.g downtolink = 3

    OUT:
    ↑ ↑ ↑ ↑ ↑ ↑ ↑
    o-o-ø-<-<-<-<, where < is rightcanonicalize
                         ø is changed
    =#
    downtolink >= L && return mps
    downtolink < 1 && throw( ArgumentError(
    "can't bring in to rightcanonical-form down to link $downtolink < 1 "))

    for i in reverse(downtolink+1:L)
        Um, Sm, mps[i] = onesitesvd2left(mps[i])
        mps[i-1] = contractsite(mps[i-1], Um, Sm)
    end

    return mps
end

canonicalize(mps::MPS) = canonicalize(mps,1)
function canonicalize(mps::MPS{L,T,TE,TB}, link) where {L,T,TE,TB}
    #=
    IN:     ↑ ↑ ↑ ↑ ↑ ↑ ↑
    mps =   o-o-o-o-o-o-o
            link = 3
    =#
    mps = deepcopy(mps)
    1 <= link <= (L-1) || throw(
        ArgumentError("link=$link must be between 1 and $(L-1)"))

    leftcanonicalize!(mps, link)
    #=    ↑(link)
    ↑ ↑ ↑ ↑ ↑ ↑ ↑
    >->->-ø-o-o-o
    =#
    rightcanonicalize!(mps, link)
    #=    ↑(link)
    ↑ ↑ ↑ ↑ ↑ ↑ ↑
    >->->-ø-<-<-<
    =#
    mps[link], Sm, Vm = onesitesvd2right(mps[link])
    Sm = contractsite(Sm, Vm)
    local l::TE = mps[1]
    local r::TE = mps[L]
    local b::Vector{TB} = mps[2:L-1]

    return CanonicalMPS(l, b, r, link, Sm)::CanonicalMPS{L}
end

function canonicalize!(mps::CanonicalMPS{L}, link::Int) where {L}
    1 <= link <= (L-1) || throw(
        ArgumentError("link must be between 1 and $(L-1), is $link"))
    clink = centerlink(mps)
    clink == link && return mps
    while centerlink(mps) != link
        movecenterlink!(mps, link > centerlink(mps))
    end
    return mps::CanonicalMPS{L}
end

canonicalize(mps::CanonicalMPS{L,T}, link) where {L,T} = canonicalize!(deepcopy(mps), link)

function movecenterlink!(mps::CanonicalMPS{L,T}, up::Bool) where {L,T}
    clink = centerlink(mps)
    up && clink == L-1 && throw(
        ArgumentError("cannot move centerlink off the right boundary") )
    !up && clink == 1 && throw(
        ArgumentError("cannot move centerlink off the left boundary") )
    S = zerosite(mps)
    if up
        mps[clink+1], S, V = onesitesvd2right(contractsite(S, mps[clink+1]))
        mps.centerlink += 1
        mps.zerosite = contractsite(S, V)
    else
        U, S, mps[clink] = onesitesvd2left(contractsite(mps[clink], S))
        mps.centerlink -= 1
        mps.zerosite = contractsite(U, S)
    end
    return mps
end

function LinearAlgebra.normalize!(mps::CanonicalMPS)
    fac = sqrt(inner(mps))
    apply!(zerosite(mps), x -> x .= x ./ fac)
    return mps
end

LinearAlgebra.normalize(mps::MPS) = normalize!(canonicalize(mps))
LinearAlgebra.normalize(mps::CanonicalMPS) = normalize!(deepcopy(mps))

function onesiteexp(mps::AbstractMPS{L,T,TE}, op::TE, site) where {T,L,TE}
    mps2 = applyonesite(mps, op, site)
    return inner(mps,mps2)
end

function onesitevar(mps::AbstractMPS{L,T,TE}, op::TE, site) where {T,L,TE}
     mps1 = applyonesite(mps, op, site)
     mps2 = applyonesite(mps1, op, site)
    return inner(mps2,mps) - inner(mps1,mps)^2
end

function twositecorr(mps::AbstractMPS{L,T,TE},
        (op1,op2)::NTuple{2,TE},
        (site1,site2)::NTuple{2,Int}) where {L,T,TE}
     mps1 = applyonesite(mps, op1, site1)
     applyonesite!(mps1, op2, site2)
     return inner(mps1, mps)
end

function applyonesite!(mps::AbstractMPS{L,T,TE}, op::TE, site::Int) where {L,T,TE}
    a = mps[site]
    if site == 1
        @tensor a[1,2] := a[-1,2] * op[-1,1]
    elseif site == L
        @tensor a[1,2] := a[1,-1] * op[-1,2]
    else
        @tensor a[1,2,3] := a[1,-1,3] * op[-1,2]
    end
    mps[site] = a
    return mps
end

applyonesite(mps, op, site) = applyonesite!(deepcopy(mps), op, site)

function entropy(mps::AbstractMPS, link::Int)
    zs = zerosite(canonicalize(mps, link))
    Σ = tensorsvd(zs)[2]
    schmidtvals = abs.(diag(Σ))
    sv2 = schmidtvals.^2
    return -(sv2' * log.(sv2))
end
