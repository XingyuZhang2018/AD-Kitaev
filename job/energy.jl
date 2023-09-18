using AD_Kitaev
using AD_Kitaev: optcont, buildbcipeps
using CUDA
using Random
using Optim
CUDA.allowscalar(false)

Random.seed!(100)
# folder = "./example/data/"
folder = "./../../../../data/xyzhang/AD_Kitaev/"
model = K_J_Γ_Γ′(-1.0, 0.0, 0.0, 0.0)
atype = CuArray
D,χ = 5,200
Ni,Nj = 1,1
bulk, key = init_ipeps(model, 
                       [1.0,1.0,1.0], 0.0; 
                       Ni = Ni, Nj = Nj, 
                       D = D, χ = χ, 
                       tol = 1e-10, maxiter = 50, miniter = 1,
                       folder = folder, 
                       type = "_random",
                       atype = CuArray,
                       ifcheckpoint = false,
                       )

folder, model, field, atype, Ni, Nj, D, χ, tol, maxiter, miniter, ifcheckpoint, verbose = key
h = hamiltonian(model)
e = []
for χ in 10:10:40
    key = folder, model, field, atype, Ni, Nj, D, χ, tol, 300, miniter, ifcheckpoint, verbose 
    oc = optcont(D, χ)
    push!(e, [χ, energy(h, buildbcipeps(atype(bulk),Ni,Nj), oc, key; show_every = 1)])
end

print("{")
for i in e
    print("{$(real(i[1])),$(real(i[2]))},")
end
print("};\n")