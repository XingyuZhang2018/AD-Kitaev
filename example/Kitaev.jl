using ADBCVUMPS
using ADBCVUMPS:optcont,buildbcipeps
using CUDA
using Random
using Test
using OMEinsum
using Optim
using Zygote
CUDA.allowscalar(false)

Random.seed!(100)
folder = "/data/xyzhang/ADBCVUMPS/"
bulk, key = init_ipeps(K_J_Γ_Γ′(-1.0, 0.0, 0.0, 0.0), [1.0,1.0,1.0], 0.0; folder=folder, type = "_random", atype = CuArray, Ni = 1, Nj = 1, D=5, χ=130, tol=1e-10, maxiter=50, miniter=1)
folder, model, field, atype, Ni, Nj, D, χ, tol, maxiter, miniter = key
key = (folder, model, field, atype, Ni, Nj, D, χ, tol, maxiter, miniter)
h = hamiltonian(model)
oc = optcont(D, χ)
real(energy(h, buildbcipeps(atype(bulk),Ni,Nj), oc, key; verbose=true))
# optimiseipeps(bulk, key; f_tol = 1e-10, opiter = 100, verbose = true)