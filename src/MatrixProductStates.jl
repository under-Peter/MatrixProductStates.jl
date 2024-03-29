module MatrixProductStates
using TensorNetworkTensors
using LinearAlgebra
using TensorOperations: @tensor, scalar, checked_similar_from_indices, tensorcontract
using Parameters
using Printf
using KrylovKit
using Base.Iterators: take, rest, Rest
using Base.Printf
import Base.iterate

include("auxiliary-iterators.jl")

abstract type AbstractMPS{L,T,TE,TB} end
abstract type AbstractMPO{L,T,TE,TB} end

export MPS, CanonicalMPS, MPO
export centerlink, zerosite, inner, canonicalize, canonicalize!, normalize!, normalize
export randmps
export onesiteexp, onesitevar, twositecorr, applyonesite!, entropy
include("mps.jl")

export heffmap
include("heffapplication.jl")

export dmrg
include("dmrg.jl")

export tdvp
include("tdvp.jl")

export hubbardMPO, annihilationOp, creationOp
include("hubbard.jl")

export tfisingMPO, sigmaZop
include("tfising.jl")
end # module
