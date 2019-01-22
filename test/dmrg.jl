@testset "DMRG" begin
    @testset "DAS" begin
        #Hubbard
        l = 3
        pchs = U1Charges(-3:3)
        vchs = U1Charges(-8:8)
        pds = [1 for i in pchs]
        vds = [5 for i in vchs]

        mpo = hubbardMPO(1, 1, pchs, l, T = Float64)
        mps = randmps(l, Float64, (U1(), (pchs, vchs), (pds, vds)))
        gs, es = dmrg(mps, mpo)

        left, blk, right = mpo[[1,2,l]]
        @tensor h[1,2,3,4,5,6] := left[1,-1,4] * blk[2,-1,-2,5] * right[3,-2,6];
        hmattens = fuselegs(h, ((1,2,3),(4,5,6)), InOut(1,-1))[1]
        hmat = hmattens[DASSector(U1Charge(0),U1Charge(0))]
        ref = minimum(eigvals(hmat))
        res = es[end][end]
        @test  ref ≈ res

        #tfIsing
        for l in 2:10 #edge cases
            mps = randmps(l, ComplexF64, (Z2(), (Z2Charges(), Z2Charges()), ([1,1], [5,5])));
            @test dmrg(mps, tfisingMPO(0,l))[2][end] ≈ -(l-1)
            @test dmrg(mps, tfisingMPO(1e6,l))[2][end]/1e6 ≈ -l
        end
        l = 22
        mps = randmps(l, ComplexF64, (Z2(), (Z2Charges(), Z2Charges()), ([1,1], [5,5])));
        gs = dmrg(mps, tfisingMPO(0,l))[1]
        @test isapprox(onesiteexp(gs, sigmaZop(), 11), 0; atol = 1e-9)
        @test isapprox(onesitevar(gs, sigmaZop(), 11), 1; atol = 1e-9)

        gs = dmrg(mps, tfisingMPO(1e6,l))[1]
        @test isapprox(onesiteexp(gs, sigmaZop(), 11), 1; atol = 1e-9)
        @test isapprox(onesitevar(gs, sigmaZop(), 11), 0; atol = 1e-9)

        gs = dmrg(mps, tfisingMPO(-1e6,l))[1]
        @test isapprox(onesiteexp(gs, sigmaZop(), div(l,2)), -1; atol = 1e-9)
        @test isapprox(onesitevar(gs, sigmaZop(), div(l,2)), 0; atol = 1e-9)
    end

    @testset "Dense" begin
        #Hubbard
        l = 3
        ansatz = randmps(l, Float64, (7,85))
        mpo = hubbardMPO(1, 1, U1Charges(-3:3), l, T = Float64)
        gsa, eas = dmrg(ansatz, todense(mpo))

        left, blk, right = todense(mpo)[[1,2,l]]
        @tensor h[1,2,3,4,5,6] := left[1,-1,4] * blk[2,-1,-2,5] * right[3,-2,6];
        hmat = fuselegs(h, ((1,2,3),(4,5,6)))[1]
        ref = min(eigvals(toarray(hmat))...)
        res = eas[end][end]
        @test ref ≈ res

        #tfIsing
        l = 22
        mps = randmps(l, ComplexF64, (Z2(), (Z2Charges(), Z2Charges()), ([1,1], [5,5])));
        dmps = todense(mps)
        gs = dmrg(dmps, todense(tfisingMPO(0,l)))[1]
        @test isapprox(onesiteexp(gs, todense(sigmaZop()), div(l,2)), 0; atol = 1e-9)
        @test isapprox(onesitevar(gs, todense(sigmaZop()), div(l,2)), 1; atol = 1e-9)

        gs = dmrg(dmps, todense(tfisingMPO(1e6,l)))[1]
        @test isapprox(onesiteexp(gs, todense(sigmaZop()), div(l,2)), 1; atol = 1e-9)
        @test isapprox(onesitevar(gs, todense(sigmaZop()), div(l,2)), 0; atol = 1e-9)

        gs = dmrg(dmps, todense(tfisingMPO(-1e6,l)))[1]
        @test isapprox(onesiteexp(gs, todense(sigmaZop()), div(l,2)), -1; atol = 1e-9)
        @test isapprox(onesitevar(gs, todense(sigmaZop()), div(l,2)), 0; atol = 1e-9)
    end
end
