using AD_Kitaev
using CUDA
using Random

Random.seed!(100)
model   = Kitaev(1.0,0.4,0.4)
folder  = "./../../../../data/xyzhang/AD_Kitaev/"
# folder = "./example/data/"
atype   = CuArray
D, χ    = 4, 80
tol     = 1e-10
maxiter = 50
miniter = 1
Ni, Nj  = 1, 1
field   = 0.0
fdirect = [1.0, 1.0, 1.0]
type    = "_random"

for targχ in 80:10:80
    observable(model, fdirect, field, type, folder, atype, D, χ, targχ, tol, maxiter, miniter, Ni, Nj; ifload = false)
end