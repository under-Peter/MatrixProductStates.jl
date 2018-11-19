using MatrixProductStates, TNTensors, LinearAlgebra, TensorOperations, KrylovKit
@testset "DMRG" begin
    l = 3; pcharges = -3:3; vcharges = -8:8
    pdims = [1 for i in pcharges]; vdims = [5 for i in vcharges]
    mpo   = hubbardMPO(1, 1, pcharges, l, T = Float64)
    mps  = randu1mps(l,Float64, (pcharges,vcharges),(pdims,vdims))
    gs, es   = dmrg(mps, mpo)
    dims =  size.(todense(mps).sites)
    ansatz = MPS(DTensor(rand(dims[1]...)),[DTensor(rand(d...)) for d in dims[2:end-1]],DTensor(rand(dims[end]...)))
    gsa, eas = dmrg(ansatz, todense(mpo))

    particleconfigs(pcharges, l) = collect(Iterators.filter(iszero ∘ sum, Iterators.product([pcharges for i in 1:l]...)))
    left, blk, right = todense(mpo)[[1,2,l]]
    @tensor h[1,2,3,4,5,6] := left[1,-1,4] * blk[2,-1,-2,5] * right[3,-2,6];
    s = size(h.array)
    hmat = reshape(h.array, prod(s[1:l]), prod(s[l+1:2*l]))
    ref = min(eigvals(hmat)...)
    res = eas[end][end]
    @test ref ≈ res

    left, blk, right = mpo[[1,2,l]]
    @tensor h[1,2,3,4,5,6] := left[1,-1,4] * blk[2,-1,-2,5] * right[3,-2,6];
    baseconfigs = particleconfigs(pcharges, l)
    hmat = [get(tensor(h), (b1..., b2...), Float64[0])[1] for b1 in baseconfigs, b2 in baseconfigs]
    ref = minimum(eigvals(hmat))
    res = es[end][end]
    @test  ref ≈ res

    l = 4
    pcharges = -2:2
    vcharges = -8:8
    pdims = [1 for i in pcharges]
    vdims = [10 for i in vcharges]
    mpo   = hubbardMPO(1.5, 1, pcharges,l, T = Float64)
    mps  = randu1mps(l,Float64, (pcharges,vcharges),(pdims,vdims))
    gs, es   = dmrg(mps, mpo)
    dims =  size.(todense(mps).sites)
    ansatz = MPS(DTensor(rand(dims[1]...)),[DTensor(rand(d...)) for d in dims[2:end-1]],DTensor(rand(dims[end]...)))
    gsa, eas = dmrg(ansatz, todense(mpo))

    particleconfigs(pcharges, l) = collect(Iterators.filter(iszero ∘ sum, Iterators.product([pcharges for i in 1:l]...)))

    left, blk, right = todense(mpo)[[1,2,l]]
    @tensor h[1,2,3,4,5,6,7,8] := left[1,-1,5] * blk[2,-1,-2,6] * blk[3,-2,-3,7] * right[4,-3,8];
    hmat = fuselegs(h,((1,2,3,4),(5,6,7,8)))[1];
    ref = eigsolve(hmat.array,1,:SR)[1]
    res = eas[end][end]
    @test ref[1] ≈ res

    left, blk, right = mpo[[1,2,l]]
    @tensor h[1,2,3,4,5,6,7,8] := left[1,-1,5] * blk[2,-1,-2,6] * blk[3,-2,-3,7]*right[4,-3,8];
    baseconfigs = particleconfigs(pcharges, l)
    hmat = [get(tensor(h), (b1..., b2...), Float64[0])[1] for b1 in baseconfigs, b2 in baseconfigs]
    ref = minimum(eigvals(hmat))
    res = es[end][end]
    @test res ≈ ref
end
