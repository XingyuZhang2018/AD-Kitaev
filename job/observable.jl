using AD_Kitaev
using AD_Kitaev: optcont, buildbcipeps
using CUDA
using Random
using Optim
CUDA.allowscalar(false)

Random.seed!(42)
# folder = "./example/data/"
folder = "./../../../../data/xyzhang/AD_Kitaev/"
model = K_J_Γ_Γ′(-1.0, 0.0, 0.0, 0.0)
atype = CuArray
D,χ = 4,80
Ni,Nj = 1,1
bulk, key = init_ipeps(model, 
                       [1.0,1.0,1.0], 0.0; 
                       Ni = Ni, Nj = Nj, 
                       D = D, χ = χ, 
                       tol = 1e-10, maxiter = 50, miniter = 1,
                       folder = folder, 
                       type = "_random",
                       atype = CuArray,
                       ifcheckpoint = false,
                       )

folder, model, field, atype, Ni, Nj, D, χ, tol, maxiter, miniter, ifcheckpoint, verbose = key
key = folder, model, field, atype, Ni, Nj, D, 140, tol, 300, miniter, ifcheckpoint, verbose 
h = hamiltonian(model)                      
oc = optcont(D, χ)
energy(h, buildbcipeps(atype(bulk),Ni,Nj), oc, key; show_every = 1)