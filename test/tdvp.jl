using MatrixProductStates, TNTensors, LinearAlgebra, TensorOperations, KrylovKit
@testset "TDVP" begin
    l = 6
    pcharges = -2:2
    vcharges = -4:4
    pdims = [1 for i in pcharges]
    vdims = [5 for i in vcharges]
    mpo = hubbardMPO(1, 1, pcharges,l, T = ComplexF64)
    mps = randu1mps(l,ComplexF64, (pcharges,vcharges),(pdims,vdims))
    gs, es  = dmrg(mps, mpo)

    ansatz = MPS(DTensor{ComplexF64,2}(rand(5,20)),
                [DTensor{ComplexF64,3}(rand(20,5,20)) for i in 1:l-2],
                 DTensor{ComplexF64,2}(rand(20,5)))
    gsa, eas = dmrg(ansatz, todense(mpo))


    cmps = normalize(mps)
    mpss = tdvp(cmps,mpo,10,0.01, verbose = false, collectall = true)
    vals = [sqrt(abs2(inner(cmps, mps))) for mps in mpss[2]]
    @test all(diff(vals) .< 0)

    cmpsa = normalize(ansatz)
    mpsas = tdvp(cmpsa,todense(mpo),10,0.01, verbose = false, collectall = true)
    vals = [sqrt(abs2(inner(cmpsa, mps))) for mps in mpsas[2]]
    @test all(diff(vals) .< 0)

    cmps = normalize(gs)
    mpss = tdvp(cmps,mpo,10,0.01, verbose = false, collectall = true)
    vals = [sqrt(abs2(inner(cmps, mps))) for mps in mpss[2]]
    @test all(vals .≈ 1)

    cmpsa = normalize(gsa)
    mpsas = tdvp(cmpsa,todense(mpo),10,0.01, verbose = false, collectall = true)
    vals = [sqrt(abs2(inner(cmpsa, mps))) for mps in mpsas[2]]
    @test all(vals .≈ 1)

    l, pcharges, vcharges = 6, -2:2, -4:4
    pdims = [1 for i in pcharges]
    vdims = [5 for i in vcharges]
    mpo   = hubbardMPO(1, 1, pcharges,l, T = ComplexF64)
    ansatz = MPS(DTensor{ComplexF64,2}(rand(5,20)),
        [DTensor{ComplexF64,3}(rand(20,5,20)) for i in 1:l-2],
        DTensor{ComplexF64,2}(rand(20,5)))
    gsa, eas = dmrg(ansatz, todense(mpo))

    mps = tdvp(ansatz,todense(mpo),30, -5im)
    e = inner(mps, todense(mpo), mps)/inner(mps)
    @test abs(e - eas[end])/eas[end] < 0.01

    @testset "ED" begin
        l, pcharges, vcharges = 3, -1:1, -5:5
        pdims = [1 for i in pcharges]
        vdims = [5 for i in vcharges]
        #= DENSE =#
        #= hamiltonian=#
        mpo = todense(hubbardMPO(1, 1, pcharges,l, T = ComplexF64))
        left, blk, right = mpo[[1,2,l]]
        @tensor h[1,2,3,4,5,6] := left[1,-1,4] * blk[2,-1,-2,5] * right[3,-2,6];
        harr = toarray(fuselegs(h,((1,2,3),(4,5,6)))[1])

        #= state =#
        mps  = todense(convert(MPS,normalize(randu1mps(l,ComplexF64, (pcharges,vcharges),(pdims,vdims)))))
        v1,v2,v3 = mps[[1,2,3]]
        @tensor v[1,2,3] := v1[1,-1] * v2[-1,2,-2] * v3[-2,3]

        t = 1e-0
        #= ED =#
        varr = toarray(fuselegs(v,((1,2,3),))[1])
        varr2 = exp(-1im * t * harr) * varr
        norm(varr) ≈ norm(varr2) ≈ 1
        sqrt(abs2(dot(varr,varr2)))

        #= tdvp =#
        mps2 = convert(MPS,tdvp(mps, mpo, 10, t))
        inner(mps2) ≈ inner(mps) ≈ 1
        res = sqrt(abs2(inner(mps2,mps)))

        v1p,v2p,v3p = mps2[[1,2,3]]
        @tensor v2[1,2,3] := v1p[1,-1] * v2p[-1,2,-2] * v3p[-2,3]
        varr2p = toarray(fuselegs(v2,((1,2,3),))[1])

        @test varr2p ≈ varr2

        #= U1Symm =#
        #= hamiltonian=#
        mpo = hubbardMPO(1, 1, pcharges,l, T = ComplexF64)
        left, blk, right = mpo[[1,2,l]]
        @tensor h[1,2,3,4,5,6] := left[1,-1,4] * blk[2,-1,-2,5] * right[3,-2,6];
        harr = toarray(fuselegs(h,((1,2,3),(4,5,6)),(1,-1))[1])
        #= state =#
        mps  = convert(MPS,normalize(randu1mps(l,ComplexF64, (pcharges,vcharges),(pdims,vdims))))
        v1,v2,v3 = mps[[1,2,3]]
        @tensor v[1,2,3] := v1[1,-1] * v2[-1,2,-2] * v3[-2,3]

        t = 1e-0
        #= ED =#
        varr = toarray(fuselegs(v,((1,2,3),))[1])
        varr2 = exp(-1im * t * harr) * varr
        norm(varr) ≈ norm(varr2) ≈ 1
        sqrt(abs2(dot(varr,varr2)))

        #= tdvp =#
        mps2 = convert(MPS,tdvp(mps, mpo, 10, t))
        inner(mps2) ≈ inner(mps) ≈ 1
        res = sqrt(abs2(inner(mps2,mps)))

        v1p,v2p,v3p = mps2[[1,2,3]]
        @tensor v2[1,2,3] := v1p[1,-1] * v2p[-1,2,-2] * v3p[-2,3]
        varr2p = toarray(fuselegs(v2,((1,2,3),),(1,))[1])

        @test varr2p ≈ varr2
    end
end
