@testset "TDVP" begin
    @testset "DAS" begin
        l = 6
        pchs = U1Charges(-2:2)
        vchs = U1Charges(-4:4)
        pds = [1 for i in pchs]
        vds = [5 for i in vchs]
        mpo = hubbardMPO(1, 1, pchs,l, T = ComplexF64)
        mps = randmps(l,ComplexF64, (U1(), (pchs, vchs), (pds, vds)))
        gs, es  = dmrg(mps, mpo)

        cmps = normalize(mps)
        mpss = tdvp(cmps,mpo,10,0.01, verbose = false, collectall = true)
        vals = [sqrt(abs2(inner(cmps, mps))) for mps in mpss[2]]
        @test all(diff(vals) .< 0)

        cmps = normalize(gs)
        mpss = tdvp(cmps,mpo,10,0.01, verbose = false, collectall = true)
        vals = [sqrt(abs2(inner(cmps, mps))) for mps in mpss[2]]
        @test all(vals .≈ 1)
    end

    @testset "Dense" begin
        l = 6
        mpo = hubbardMPO(1, 1, U1Charges(-2:2),l, T = ComplexF64)
        ansatz = randmps(l, ComplexF64, (5,20))
        gsa, eas = dmrg(ansatz, todense(mpo))

        cmpsa = normalize(ansatz)
        mpsas = tdvp(cmpsa,todense(mpo),10,0.01, verbose = false, collectall = true)
        vals = [sqrt(abs2(inner(cmpsa, mps))) for mps in mpsas[2]]
        @test all(diff(vals) .< 0)

        cmpsa = normalize(gsa)
        mpsas = tdvp(cmpsa,todense(mpo),10,0.01, verbose = false, collectall = true)
        vals = [sqrt(abs2(inner(cmpsa, mps))) for mps in mpsas[2]]
        @test all(vals .≈ 1)
    end


    @testset "ED" begin
        l = 3
        pchs = U1Charges(-1:1)
        vchs = U1Charges(-5:5)
        pds = [1 for i in pchs]
        vds = [5 for i in vchs]
        @testset "Dense" begin
            #= hamiltonian=#
            mpo = todense(hubbardMPO(1, 1, pchs,l, T = ComplexF64))
            left, blk, right = mpo[[1,2,l]]
            @tensor h[1,2,3,4,5,6] := left[1,-1,4] * blk[2,-1,-2,5] * right[3,-2,6];
            harr = toarray(fuselegs(h,((1,2,3),(4,5,6)))[1])

            #= state =#
            mps = todense(convert(MPS,normalize(randmps(l,ComplexF64,(U1(), (pchs,vchs),(pds,vds))))))
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
            @test inner(mps2) ≈ inner(mps) ≈ 1
            res = sqrt(abs2(inner(mps2,mps)))

            v1p,v2p,v3p = mps2[[1,2,3]]
            @tensor v2[1,2,3] := v1p[1,-1] * v2p[-1,2,-2] * v3p[-2,3]
            varr2p = toarray(fuselegs(v2,((1,2,3),))[1])

            @test varr2p ≈ varr2
        end

        @testset "DAS" begin
            l = 3
            mpo = hubbardMPO(1, 1, pchs,l, T = ComplexF64)
            left, blk, right = mpo[[1,2,l]]
            @tensor h[1,2,3,4,5,6] := left[1,-1,4] * blk[2,-1,-2,5] * right[3,-2,6];
            harr = toarray(fuselegs(h,((1,2,3),(4,5,6)),InOut(1,-1))[1])

            mps  = convert(MPS,normalize(randmps(l,ComplexF64, (U1(), (pchs,vchs),(pds,vds)))))
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
            @test inner(mps2) ≈ inner(mps) ≈ 1
            res = sqrt(abs2(inner(mps2,mps)))

            v1p,v2p,v3p = mps2[[1,2,3]]
            @tensor v2[1,2,3] := v1p[1,-1] * v2p[-1,2,-2] * v3p[-2,3]
            varr2p = toarray(fuselegs(v2,((1,2,3),),InOut(1))[1])
            @test varr2p ≈ varr2
        end
    end
end
