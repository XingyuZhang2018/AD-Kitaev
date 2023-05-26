using AD_Kitaev
using CUDA
using FileIO

model = K_J_Γ_Γ′(-1.0, -0.1, 0.3, -0.02)
fdirection = [1.0, 1.0, 1.0]
type = "_random"
folder = "./../../../../data/xyzhang/AD_Kitaev/K_J_Γ_Γ′_1x2/"
D = 2
χ = 20
tol = 1e-10
maxiter = 10
miniter = 1
f = 0.61:0.02:0.73
field, F = [], []
for i = 1:(length(f)-1)
    key1 = model, fdirection, f[i], type, folder, D, χ, tol, maxiter, miniter
    key2 = model, fdirection, f[i+1], type, folder, D, χ, tol, maxiter, miniter
    field = [field; f[i]]
    F = [F; fidelity(key1,key2)]
end
# fidelityplot = plot()
plot!(fidelityplot, field, F, shape = :auto, label = "5° fidelity D = $(D)", lw = 2,legend = :bottomright)