using AD_Kitaev
using CUDA
using Random
using Optim
CUDA.allowscalar(false)

Random.seed!(100)
degree = 270.0
folder = "./example/data/"
bulk, key = init_ipeps(Kitaev_Heisenberg(degree);
                       Ni=1, Nj=2,
                       D=2, χ=10, 
                       tol=1e-10, maxiter=10, miniter=2,
                       folder=folder, 
                       atype = Array
                       )

optimiseipeps(bulk, key; 
              f_tol = 1e-6, opiter = 100, 
              verbose = true
              )