using MatrixProductStates, TNTensors, LinearAlgebra, TensorOperations, KrylovKit
@testset "MPS" begin
    @testset "Inner" begin
        l = 10;
        pchs = U1Charges(-2:2)
        vchs = U1Charges(-3:3)
        pds = [1 for i in pchs]
        vds = [2 for i in vchs]
        mpo = hubbardMPO(1, 1, pchs,l, T = ComplexF64)
        mps = randmps(l, ComplexF64, (U1(), (pchs, vchs), (pds, vds)))
        mpsa = todense(mps)
        mpoa = todense(mpo)

        @test inner(mps) ≈ inner(mpsa)
        @test inner(mps,mpo,mps) ≈ inner(mpsa,mpoa,mpsa)
    end

    @testset "Canonicalization" begin
        l = 10;
        pchs = U1Charges(-2:2)
        vchs = U1Charges(-3:3)
        pds = [1 for i in pchs]
        vds = [2 for i in vchs]
        mps = randmps(l, ComplexF64, (U1(), (pchs, vchs), (pds, vds)))
        for i in 1:l-1
            lmps = MatrixProductStates.leftcanonicalize!(deepcopy(mps),i)
            lenv = foldl(MatrixProductStates.contractsites,zip(lmps[1:i],lmps[1:i]'), init=())
            @test all([v ≈ I for (k,v) in tensor(lenv)])
        end

        for i in l-1:-1:1
            rmps = MatrixProductStates.rightcanonicalize!(deepcopy(mps),i)
            renv = foldr(MatrixProductStates.contractsites,[zip(rmps[i+1:l],rmps[i+1:l]')...], init=())
            @test all([v ≈ I for (k,v) in tensor(renv)])
        end

        for i in 1:l-1
            cmps = canonicalize(mps,i)
            renv = foldr(MatrixProductStates.contractsites,[zip(cmps[i+1:l],cmps[i+1:l]')...], init=())
            x2 = all([v ≈ I for (k,v) in tensor(renv)])
            lenv = foldl(MatrixProductStates.contractsites,[zip(cmps[1:i],cmps[1:i]')...], init=())
            @test all([v ≈ I for (k,v) in tensor(lenv)])
        end
    end
end
