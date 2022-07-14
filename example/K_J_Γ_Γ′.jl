using ADBCVUMPS
using CUDA
using Random
using Test
using OMEinsum
using Optim
using Zygote
CUDA.allowscalar(false)

Random.seed!(99)
# device!(7)
folder = "./example/K_J_Γ_Γ′_1x2/"
bulk, key = init_ipeps(K_J_Γ_Γ′(-1.0, -0.1, 0.3, -0.02), [1.0,1.0,1.0], 0.19;folder=folder, type = "_zigzag", atype = Array, D=2, χ=20, tol=1e-10, maxiter=10, miniter=1)
optimiseipeps(bulk, key; f_tol = 1e-10, opiter = 100, verbose = true)
