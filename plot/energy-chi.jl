using AD_Kitaev
using AD_Kitaev: buildbcipeps, energy, optcont
using CUDA
using Plots
CUDA.allowscalar(false)

device!(7)
function new_energy(bulk, new_key)
    folder, model, field, atype, D, χ, tol, maxiter, miniter = new_key
    h = hamiltonian(model)
    oc = optcont(D, χ)
    Ni,Nj = 1,2
    energy(h, buildbcipeps(atype(bulk),Ni,Nj), oc, new_key; verbose = true, savefile = true)
end

model = K_J_Γ_Γ′(-1.0, -0.1, 0.3, -0.02)
type = "_random"
for field in 0.61:0.01:0.61
    folder = "./../../../../data/xyzhang/AD_Kitaev/K_J_Γ_Γ′_1x2/"
    fdirection, atype, D, χ, tol, maxiter, miniter = [1.0,1.0,0.963424], CuArray, 4, 80, 1e-10, 10, 2

    bulk, key = init_ipeps(model, fdirection, field; folder = folder, type = type, atype = atype, D=D, χ=χ, tol=tol, maxiter=maxiter, miniter=miniter)
    folder, model, field, atype, D, χ, tol, maxiter, miniter = key

    x = 80

    yenergy = []
    # ymag = []
    for χ in x
        new_key = (folder, model, field, atype, D, χ, tol, 10, 2)
        ener = new_energy(bulk, new_key)
        yenergy = [yenergy; ener]
        # ymag = [ymag; mag]
    end
end
# yenergy /= 2
# energyplot = plot()
# magplot = plot()
# plot!(energyplot, x, real(yenergy), title = "energy", label = "energy",legend = :bottomright, xlabel = "χ", ylabel = "E", lw = 3)
# plot!(magplot, x, ymag, title = "magnetization", label = "magnetization",legend = :bottomright, xlabel = "χ", ylabel = "M", lw = 3)
# obs = plot(energyplot, magplot, layout = (2,1), xlabel="χ", size = [800, 1000])
# savefig(energyplot,"./plot/obs-$(model)_D$(D)_χ$(χ).svg")