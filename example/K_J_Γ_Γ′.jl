using AD_Kitaev
using CUDA
using Random
using Optim
CUDA.allowscalar(false)

Random.seed!(100)
# folder = "./example/data/"
folder = "./../../../../data/xyzhang/AD_Kitaev/"
bulk, key = init_ipeps(K_J_Γ_Γ′(0.9888369128416001, 0.00680642148156, 0.00462836660746, -0.0043561097482), 
                       [1.0,1.0,1.0], 0.80; 
                       Ni=1, Nj=1, 
                       D=4, χ=80, 
                       tol=1e-10, maxiter=100, miniter=1,
                       folder=folder, 
                       type = "_QSL",
                       atype = CuArray
                       )

optimiseipeps(bulk, key; 
              f_tol = 1e-10, 
              opiter = 0, 
              verbose = true
              )