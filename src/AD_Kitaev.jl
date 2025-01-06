module AD_Kitaev

using FileIO
using KrylovKit
using LinearAlgebra
using LineSearches
using Random
using Optim
using OMEinsum
using Printf
using Parameters
using Zygote
using TeneT
using HDF5

using TeneT: _arraytype, update!, checkpoint, leg3, leg4
using TeneT: ALCtoAC

export Heisenberg
export hamiltonian
export diaglocal, TFIsing, Heisenberg, Kitaev, Kitaev_Heisenberg, K_J_Γ_Γ′, K_Γ
export observable
export iPEPSOptimize, init_ipeps, optimise_ipeps, energy

include("defaults.jl")
include("hamiltonian_models.jl")
include("optimise_ipeps.jl")
include("init_ipeps.jl")
include("wp_operator.jl")
include("build_A_M.jl")
include("observable.jl")

end
