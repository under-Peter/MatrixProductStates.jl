function hubbardMPO(t, U, pcharges, N; T = Complex128)
    shift = -minimum(pcharges)
    shift < 0 && (shift = 0)
    pdims = [1 for i in pcharges]
    preapp  = T[0 0
                0 1] # |0><0|_p ⊗ |1><1|_d
    postapp = T[1 0
                0 0] # |0><0|_p ⊗ |0><0|_d
    app     = T[0 0
                1 0] # |0><0|_p ⊗ |1><0|_d
    mat00to1 = reshape(T[0, 1], 2, 1)
    mat1to01 = reshape(T[1, 0], 1, 2)

    tensordict = Dict{NTuple{4,Int64}, Array{T, 4}}()
    for (i, ch) in enumerate(pcharges)
        peye= Matrix{T}(I,pdims[i],pdims[i])
        #local interaction term
        @tensor deg[1, 2, 3, 4] :=
            preapp[1, 4]  * peye[2, 3] +  # |1><1| * Ι
            postapp[1, 4] * peye[2, 3] +  # |0><0| * Ι
            U/2 * (ch+shift) * (ch+shift-1)  * app[1, 4] * peye[2, 3]
            # U * ch * (ch-1) * |1><0|
        tensordict[(0, ch, ch, 0)] = deg

        #hopping term
        if (ch - 1) in pcharges
            #a a^†
            # (0,0) → a_i → (1,0)
            @tensor deg[1, 2, 3, 4] := mat00to1[1, 4] * peye[2, 3]
            tensordict[(0, ch, ch-1, 1)] = -t * sqrt(ch+shift) * deg
            # (1,0) → a^†_{i+1} → (0,1)
            @tensor deg[1, 2, 3, 4] := mat1to01[1, 4] * peye[2, 3]
            tensordict[(1, ch-1, ch, 0)] = sqrt(ch+shift) * deg

            #a^† a
            # (0,0) → a^†_i → (-1,0)
            @tensor deg[1, 2, 3, 4] := mat00to1[1, 4] * peye[2, 3]
            tensordict[(0, ch-1, ch, -1)] = -t * sqrt(ch+shift) * deg
            # (-1,0) → a_{i+1} → (0,1)
            @tensor deg[1, 2, 3, 4] := mat1to01[1, 4] * peye[2, 3]
            tensordict[(-1, ch, ch-1, 0)] = sqrt(ch+shift) * deg
        end
    end
    bulk = U1Tensor((-1:1, pcharges, pcharges, -1:1),
                    ([1, 2, 1], pdims, pdims, [1, 2, 1]),
                    (1, 1, -1, -1),
                    tensordict)
    leftbound  = U1Tensor(  (-1:1,), ([1, 2, 1],), (-1,),
                            Dict{NTuple{1,Int}, Array{T, 1}}((0,) => [0, 1]))
    rightbound = U1Tensor(  (-1:1,), ([1, 2, 1],), (1,),
                            Dict{NTuple{1,Int}, Array{T, 1}}((0,) => [1, 0]))
    #=
    (3)          (4)         (3)
     ↑            ↑           ↑
    [lb]→(2) (2)→[b]→(3) (2)→[rb]
     ↑            ↑           ↑
    (1)          (1)         (1)
    =#
    @tensor bulk[1,2,3,4]       := bulk[2,1,4,3]
    @tensor leftbound[1, 2, 3]  := leftbound[-1] *  bulk[1, -1, 2, 3]
    @tensor rightbound[1, 2, 3] := bulk[1, 2, -1, 3] * rightbound[-1]

    return MPO{N}(leftbound, bulk,  rightbound)
end

function creationOp(pchs::UnitRange, T = ComplexF64)
    ds = fill(1, length(pchs))
    ts =     Dict{NTuple{2,Int}, Array{T, 2}}(
        (ch-1,ch) => reshape([sqrt(ch)],1,1)
        for ch in pchs if (ch-1) in pchs
    )
    return U1Tensor((pchs, pchs), (ds, ds), (1, -1), ts)
end

function annihilationOp(pchs::UnitRange, T = ComplexF64)
    ds = fill(1, length(pchs))
    ts =     Dict{NTuple{2,Int}, Array{T, 2}}(
        (ch,ch-1) => reshape([sqrt(ch)],1,1)
        for ch in pchs if (ch-1) in pchs
    )
    return U1Tensor((pchs, pchs), (ds, ds), (1, -1), ts)
end
