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
device!(3)
folder = "./../../../../data1/xyzhang/ADBCVUMPS/Kitaev_complex_1x2/"
bulk, key = init_ipeps(Kitaev(-1.0, -1.0, -1.0), [0.0, 0.0, 0.0];folder=folder, atype = CuArray, D=4, χ=30, tol=1e-10, maxiter=10, miniter=2)
optimiseipeps(bulk, key; f_tol = 1e-6, opiter = 1000, verbose = true)