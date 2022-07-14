using ADBCVUMPS
using BCVUMPS
using CUDA
using Random
using Test
using OMEinsum
using Optim
using Zygote
CUDA.allowscalar(false)

Random.seed!(100)
degree = 270.0
folder = "./data/all/"
bulk, key = init_ipeps(Kitaev_Heisenberg(degree);folder=folder, atype = CuArray, D=4, χ=20, tol=1e-10, maxiter=10, miniter=2)
optimiseipeps(bulk, key; f_tol = 1e-6, opiter = 100, verbose = true)