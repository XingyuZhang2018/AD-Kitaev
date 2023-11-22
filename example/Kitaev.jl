using AD_Kitaev
using CUDA
using Random
using Optim
using LineSearches
CUDA.allowscalar(false)

Random.seed!(101)
# folder = "./example/data/"
folder = "./example/data/fold/"
bulk, key = init_ipeps(Kitaev(1.0,1.0,1.0), 
                       [1.0,1.0,-2.0], 0.0; 
                       Ni = 1, Nj = 1, 
                       D = 3, χ = 20, 
                       tol = 1e-10, maxiter = 50, miniter = 1,
                       folder=folder, 
                       type = "_random",
                       atype = Array,
                       ifcheckpoint = false,
                       )

optimiseipeps(bulk, key; 
              f_tol = 1e-10, 
              opiter = 100, 
              maxiter_ad = 10,
              miniter_ad = 3,
              verbose = true,
              optimmethod = LBFGS(m = 20,
                alphaguess=LineSearches.InitialStatic(alpha=1e-5,scaled=true))
              )