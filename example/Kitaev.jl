using AD_Kitaev
using CUDA
using Random
using Optim
CUDA.allowscalar(false)

Random.seed!(101)
# folder = "./example/data/"
folder = "./../../../../data/xyzhang/AD_Kitaev/"
bulk, key = init_ipeps(Kitaev(1.0,0.4,0.4), 
                       [1.0,1.0,-2.0], 0.0; 
                       Ni = 1, Nj = 1, 
                       D = 4, χ = 80, 
                       tol = 1e-10, maxiter = 50, miniter = 1,
                       folder=folder, 
                       type = "_random",
                       atype = CuArray,
                       ifcheckpoint = false,
                       )

optimiseipeps(bulk, key; 
              f_tol = 1e-10, 
              opiter = 1000, 
              maxiter_ad = 10,
              miniter_ad = 3,
              verbose = true
              )