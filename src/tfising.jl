"""
    tfisingMPO(h, N[; T = ComplexF64])
returns an MPO for the transverse field Ising model:
```
    H = -∑_i σ^i_x σ^(i+1)_x - ∑_i h σ^i_z
```
on N sites.
"""
function tfisingMPO(h, N; T = ComplexF64)
    bulk = DASTensor{T,4}(
        Z2(),
        ntuple(i -> Z2Charges(),4),
        ([2,2], [1,1], [1,1], [2,2]),
        InOut(1,1,-1,-1))
    #this construction is dangerous insofar as we deliberately don't
    #populate certain sectors to avoid having to introduce more logic
    #for a more robust setting:
    #vd = 2
    #track whether operation happened or not

    preapp  = T[0 0; 0 1] # 1 -> 1
    postapp = T[1 0; 0 0] # 2 -> 2
    app     = T[0 0; 1 0] # 1 -> 2
    peye= Matrix{T}(I, 1, 1)

    for ch in Z2Charges()
        #transverse field term
        hz = ifelse(iszero(ch), h, -h)
        @tensor deg[1,2,3,4] := (preapp[1,4] + postapp[1,4] - hz*app[1,4]) * peye[2,3]
        bulk[DASSector(Z2Charge(0), ch, ch, Z2Charge(0))] = deg

        #XX term
        @tensor deg[1, 2, 3, 4] := -preapp[1, 4] * peye[2, 3]
        bulk[DASSector(Z2Charge(0), ch ⊕ Z2Charge(1), ch, Z2Charge(1))] = deg

        @tensor deg[1, 2, 3, 4] :=  app[1, 4] * peye[2, 3]
        bulk[DASSector(Z2Charge(1), ch, ch ⊕ Z2Charge(1), Z2Charge(0))] = deg
    end

    # (3)        (4)        (3)
    # [lb](2) (2)[b](3) (2)[rb]
    # (1)        (1)        (1)
    lb = DASTensor{T,1}(Z2(), (Z2Charges(),), ([2,2],), InOut(-1))
    rb = DASTensor{T,1}(Z2(), (Z2Charges(),), ([2,2],), InOut( 1))
    lb[DASSector(Z2Charge(0))] = [0, 1]
    rb[DASSector(Z2Charge(0))] = [1, 0]

    @tensor bulk[1,2,3,4]       := bulk[2,1,4,3]
    @tensor leftbound[1, 2, 3]  := lb[-1] *  bulk[1, -1, 2, 3]
    @tensor rightbound[1, 2, 3] := bulk[1, 2, -1, 3] * rb[-1]

    return MPO{N}(leftbound, bulk,  rightbound)
end

function sigmaZop(T = ComplexF64)
    a = DASTensor{T,2}(Z2(),
        (Z2Charges(), Z2Charges()),
        ([1,1],[1,1]),
        InOut(1,-1))
    for ch in Z2Charges()
        a[DASSector(ch, ch)] = reshape([ifelse(iszero(ch), 1, -1)],1,1)
    end
    return a
end
