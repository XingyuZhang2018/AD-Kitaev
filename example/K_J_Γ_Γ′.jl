using ADBCVUMPS
using CUDA
using Random
using Test
using OMEinsum
using Optim
using Zygote
CUDA.allowscalar(false)

Random.seed!(100)
folder = "./example/K_J_Γ_Γ′_1x2/"
bulk, key = init_ipeps(K_J_Γ_Γ′(-1.0, -0.1, 0.3, -0.02), [1.0,1.0,1.0], 0.11;folder=folder, type = "_ferro", atype = CuArray, D=5, χ=50, tol=1e-10, maxiter=10, miniter=1)
optimiseipeps(bulk, key; f_tol = 1e-10, opiter = 100, verbose = true)