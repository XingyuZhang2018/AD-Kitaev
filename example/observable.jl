using AD_Kitaev
using CUDA
using Random
using Printf

Random.seed!(100)
model   = K_J_Γ_Γ′(0.9888369128416001, 0.00680642148156, 0.00462836660746, -0.0043561097482)
folder  = "./../../../../data/xyzhang/AD_Kitaev/"
# folder = "./example/data/"
# folder  = "../data/AD_Kitaev/"
atype   = CuArray
D, χ    = 4, 80
tol     = 1e-10
maxiter = 50
miniter = 1
Ni, Nj  = 1, 1

fdirect = [1.0, 1.0, 1.0]
type    = "_QSL"

for targχ in 80:10:80, field in 0.8:0.01:0.8
    file = joinpath(folder, "$(Ni)x$(Nj)", "$(model)_field$(fdirect)_$(@sprintf("%0.2f", field))$type", "D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter)_miniter$(miniter).jld2")
    if ispath(file)
        @show field
        observable(model, fdirect, field, type, folder, atype, D, χ, targχ, tol, maxiter, miniter, Ni, Nj; ifload = false)
    end
end