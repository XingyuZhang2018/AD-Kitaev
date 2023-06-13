using AD_Kitaev
using CUDA
using Random
using Optim
CUDA.allowscalar(false)

Random.seed!(100)
folder = "./example/data/"
bulk, key = init_ipeps(K_J_Γ_Γ′(-1.0, 0.0, 0.0, 0.0), 
                       [1.0,1.0,1.0], 0.0; 
                       Ni = 1, Nj = 1, 
                       D = 5, χ = 80, 
                       tol = 1e-10, maxiter = 50, miniter = 1,
                       folder=folder, 
                       type = "_random",
                       atype = CuArray,
                       ifcheckpoint = true,
                       )

optimiseipeps(bulk, key; 
              f_tol = 1e-10, 
              opiter = 10, 
              maxiter_ad = 10,
              miniter_ad = 3,
              verbose = true
              )