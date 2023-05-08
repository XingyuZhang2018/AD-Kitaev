using AD_Kitaev
using CUDA
using FileIO
using Printf: @sprintf
using Random

Random.seed!(100)
folder  = "/data/xyzhang/AD_Kitaev/"
atype   = CuArray
D, χ    = 4, 80
tol     = 1e-10
maxiter = 10
miniter = 1
Ni, Nj  = 4, 4
f       = 0.0:0.1:0.0
fdirect = [1.0, 1.0, 1.0]
model   = K_J_Γ_Γ′(-1.0, -0.1, 0.3, -0.02)
# 0.985263
# 0.963424
# 0.825221
type = "_random"
E, Ex, Ey, Ez, M = [], [], [], [], [], [], [], [], [], []
for x in f
    @show x
    if x == 0.0
        tfolder = folder*"$(Ni)x$(Nj)/$(model)/"
    else
        # if x > 0.13
        #     type = "_random"
        # else
        #     type = "_zigzag"
        # end
        # type = ""
        tfolder = folder*"$(Ni)x$(Nj)/$(model)_field$(fdirect)_$(@sprintf("%0.2f", x))$(type)/"
    end
    @show isdir(tfolder)
    if isdir(tfolder)
        y1, y2, y3, y4, y5 = observable(model, fdirect, x, "$(type)", folder, atype, D, χ, tol, maxiter, miniter, Ni, Nj)

        global E  = [E;  y1]
        global Ey = [Ey; y2]
        global Ey = [Ey; y3]
        global Ez = [Ez; y4]
        global M  = [M;  y5]
    end
end

# print({})
# for
# @show collect(f) E