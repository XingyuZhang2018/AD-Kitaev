using ADBCVUMPS
using CUDA
using FileIO
using Printf: @sprintf
using Random
using LinearAlgebra: norm

Random.seed!(100)
folder  = "/data/xyzhang/ADBCVUMPS/"
atype   = CuArray
D, χ    = 5, 100
tol     = 1e-10
maxiter = 10
miniter = 1
Ni, Nj  = 1, 2
f       = 0.01:0.01:1.0
fdirect = [1.0, 1.0, 1.0]
model   = K_J_Γ_Γ′(-1.0, -0.1, 0.3, -0.02)
# 0.985263
# 0.963424
# 0.825221
type = "_NP"
h, E, Ex, Ey, Ez, M = [], [], [], [], [], [], [], [], [], [], []
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
        push!(h,  x)
        push!(E,  y1)
        push!(Ex, y2)
        push!(Ey, y3)
        push!(Ez, y4)
        push!(M,  y5)
    end
end

print("{")
for k in 1:length(h)
    print("{$(h[k]),$(E[k])},")
end
print("};\n")

print("{")
for k in 1:length(h)
    mag = [0,0,0]
    for j in 1:Nj, i in 1:Ni
        mag += sum(M[k][i,j,:])
    end
    mag = real(sum(mag/4/sqrt(3)))
    print("{$(h[k]),$(mag)},")
end
print("};\n")

print("{")
for k in 1:length(h)
    mag = [0,0,0]
    for j in 1:Nj, i in 1:Ni
        mag += sum(M[k][i,j,:])
    end
    mag = norm(mag/4)
    print("{$(h[k]),$(mag)},")
end
print("}\n")