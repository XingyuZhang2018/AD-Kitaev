using AD_Kitaev
using AD_Kitaev:σx,σy,σz,buildbcipeps, optcont
using TeneT
using CUDA
using FileIO
using LinearAlgebra: norm, I, cross
using OMEinsum
using Plots
using Random
using Statistics: std

function observable(model, folder, D, χ, tol, maxiter, miniter)
    bulk, key = init_ipeps(model; folder = folder, atype = CuArray, D=D, χ=χ, tol=tol, maxiter=maxiter, miniter=miniter, verbose = true)
    folder, model, field, atype, D, χ, tol, maxiter, miniter = key
    h = hamiltonian(model)
    oc = optcont(D, χ)
    Ni = 1
    Nj = Int(size(bulk,6) / Ni)
    bulk = buildbcipeps(bulk,Ni,Nj)
    ap = [ein"abcdx,ijkly -> aibjckdlxy"(bulk[i], conj(bulk[i])) for i = 1:Ni*Nj]
    ap = [reshape(ap[i], D^2, D^2, D^2, D^2, 4, 4) for i = 1:Ni*Nj]
    ap = reshape(ap, Ni, Nj)
    a = [ein"ijklaa -> ijkl"(ap[i]) for i = 1:Ni*Nj]
    a = reshape(a, Ni, Nj)
    
    chkp_file_obs = folder*"obs_D$(D^2)_chi$(χ).jld2"
    FL, FR = load(chkp_file_obs)["env"]
    chkp_file_up = folder*"up_D$(D^2)_χ$(χ).jld2"                     
    rtup = SquareBCVUMPSRuntime(a, chkp_file_up, χ; verbose = false)   
    FLu, FRu, ALu, ARu, Cu = rtup.FL, rtup.FR, rtup.AL, rtup.AR, rtup.C
    chkp_file_down = folder*"down_D$(D^2)_χ$(χ).jld2"                              
    rtdown = SquareBCVUMPSRuntime(a, chkp_file_down, χ; verbose = false)   
    ALd,ARd,Cd = rtdown.AL,rtdown.AR,rtdown.C

    M = Array{Array{ComplexF64,1},2}(undef, Nj, 2)
    Sx1 = reshape(ein"ab,cd -> acbd"(σx/2, I(2)), (4,4))
    Sx2 = reshape(ein"ab,cd -> acbd"(I(2), σx/2), (4,4))
    Sy1 = reshape(ein"ab,cd -> acbd"(σy/2, I(2)), (4,4))
    Sy2 = reshape(ein"ab,cd -> acbd"(I(2), σy/2), (4,4))
    Sz1 = reshape(ein"ab,cd -> acbd"(σz/2, I(2)), (4,4))
    Sz2 = reshape(ein"ab,cd -> acbd"(I(2), σz/2), (4,4))
    for j = 1:Nj, i = 1:Ni
        jr = j + 1 - (j==Nj)*Nj
        ir = Ni + 1 - i
        lr3 = ein"(((((aeg,abc),cd),ehfbpq),ghi),ij),dfj -> pq"(FL[i,j],ALu[i,j],Cu[i,j],ap[i,j],ALd[ir,j],Cd[ir,j],FR[i,j])
        Mx1 = ein"pq, pq -> "(Array(lr3),Sx1)
        Mx2 = ein"pq, pq -> "(Array(lr3),Sx2)
        My1 = ein"pq, pq -> "(Array(lr3),Sy1)
        My2 = ein"pq, pq -> "(Array(lr3),Sy2)
        Mz1 = ein"pq, pq -> "(Array(lr3),Sz1)
        Mz2 = ein"pq, pq -> "(Array(lr3),Sz2)
        n3 = ein"pp -> "(lr3)
        M[j,1] = [Array(Mx1)[]/Array(n3)[], Array(My1)[]/Array(n3)[], Array(Mz1)[]/Array(n3)[]]
        M[j,2] = [Array(Mx2)[]/Array(n3)[], Array(My2)[]/Array(n3)[], Array(Mz2)[]/Array(n3)[]]
        print("M[[$(j),$(1)]] = {")
        for k = 1:3 
            print(real(M[j,1][k])) 
            k == 3 ? println("};") : print(",")
        end
        print("M[[$(j),$(2)]] = {")
        for k = 1:3 
            print(real(M[j,2][k])) 
            k == 3 ? println("};") : print(",")
        end
        # println("M = $(sqrt(real(M[i,j]' * M[i,j])))")
    end
    Cross = norm(cross(M[1,1],M[1,2]))
    @show Cross
    mag = (norm(M[1,1]) + norm(M[2,1]) + norm(M[2,2]) + norm(M[1,2]))/4
    ferro = norm((M[1,1] + M[1,2] + M[2,2] + M[2,1])/4)
    zigzag = norm((M[1,1] + M[1,2] - M[2,2] - M[2,1])/4)
    stripy = norm((M[1,1] - M[1,2] + M[2,2] - M[2,1])/4)
    Neel = norm((M[1,1] - M[1,2] - M[2,2] + M[2,1])/4)

    oc1, oc2 = oc
    hx, hy, hz = h
    ap /= norm(ap)
    etol = 0
    hx = reshape(permutedims(hx, (1,3,2,4)), (4,4))
    hy = reshape(ein"ae,bfcg,dh -> abefcdgh"(I(2), hy, I(2)), (4,4,4,4))
    hz = reshape(ein"ae,bfcg,dh -> abefcdgh"(I(2), hz, I(2)), (4,4,4,4))

    Ex, Ey, Ez = 0, 0, 0
    for j = 1:Nj, i = 1:Ni
        ir = Ni + 1 - i
        jr = j + 1 - (j==Nj) * Nj
        lr = oc1(FL[i,j],ALu[i,j],ap[i,j],ALd[ir,j],Cu[i,j],Cd[ir,j],FR[i,jr],ARu[i,jr],ap[i,jr],ARd[ir,jr])
        ey = ein"pqrs, pqrs -> "(lr,hy)
        n = ein"pprr -> "(lr)
        Ey += Array(ey)[]/Array(n)[]
        println("hy = $(Array(ey)[]/Array(n)[])")
        etol += Array(ey)[]/Array(n)[]

        lr2 = ein"(((((aeg,abc),cd),ehfbpq),ghi),ij),dfj -> pq"(FL[i,j],ALu[i,j],Cu[i,j],ap[i,j],ALd[ir,j],Cd[ir,j],FR[i,j])
        ex = ein"pq, pq -> "(lr2,hx)
        n = Array(ein"pp -> "(lr2))[]
        Ex += Array(ex)[]/n
        println("hx = $(Array(ex)[]/n)")
        etol += Array(ex)[]/n
    end
    
    for j = 1:Nj, i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        lr3 = oc2(ALu[i,j],Cu[i,j],FLu[i,j],ap[i,j],FRu[i,j],FL[ir,j],ap[ir,j],FR[ir,j],ALd[i,j],Cd[i,j])
        ez = ein"pqrs, pqrs -> "(lr3,hz)
        n = ein"pprr -> "(lr3)
        Ez += Array(ez)[]/Array(n)[]
        println("hz = $(Array(ez)[]/Array(n)[])") 
        etol += Array(ez)[]/Array(n)[]
    end
    # ΔE = real(Ex - (Ey + Ez)/2)
    ΔE = std(real.([Ex, Ey, Ez]))
    @show ΔE
    if field != [0.0,0.0,0.0]
        for j = 1:Nj, i = 1:Ni
            ir = Ni + 1 - i
            lr3 = ein"(((((aeg,abc),cd),ehfbpq),ghi),ij),dfj -> pq"(FL[i,j],ALu[i,j],Cu[i,j],ap[i,j],ALd[ir,j],Cd[ir,j],FR[i,j])
            Mx = ein"pq, pq -> "(lr3,atype(σx/2))
            My = ein"pq, pq -> "(lr3,atype(σy/2))
            Mz = ein"pq, pq -> "(lr3,atype(σz/2))
            n3 = Array(ein"pp -> "(lr3))[]
            M = [Array(Mx)[]/n3, Array(My)[]/n3, Array(Mz)[]/n3]
            @show M
            etol += M' * field
        end
    end

    println("e = $(etol/Ni/Nj)")
    return mag, ferro, stripy, zigzag, Neel, real(etol/Ni/Nj), ΔE, Cross
end


Random.seed!(100)
folder, tol, maxiter, miniter = "./../../../../data/xyzhang/AD_Kitaev/K_J_Γ_Γ′_1x2/", 1e-10, 10, 2
Γ = 0.03
Dχ = [[2,20]]
mag, ferro, stripy, zigzag, Neel, E, ΔE, Cross = [], [], [], [], [], [], [], []
for (D,χ) in Dχ
    model = K_J_Γ_Γ′(-1.0, 0.0, Γ, 0.0)
    y1, y2, y3, y4, y5, y6, y7, y8 = observable(model, folder, D, χ, tol, maxiter, miniter)
    mag = [mag; y1]
    ferro = [ferro; y2]
    stripy = [stripy; y3]
    zigzag = [zigzag; y4]
    Neel = [Neel; y5]
    E = [E; y6]
    ΔE = [ΔE; y7]
    Cross = [Cross; y8]
end

# D = 1 ./ (2:1:5)
# magplot = plot()
# plot!(magplot, D, mag, shape = :circle, title = "mag-1/D Γ/|K| = $(Γ)", label = "mag ",legend = :inside, lw = 2)
# plot!(magplot, D, ferro, shape = :dtriangle, label = "ferro",legend = :inside, lw = 2)
# plot!(magplot, D, stripy, shape = :heptagon, label = "stripy",legend = :inside,  lw = 2)
# plot!(magplot, D, zigzag, shape = :cross, label = "zigzag", lw = 2)
# plot!(magplot, D, Neel, shape = :diamond, label = "Neel",legend = :topleft, xlabel = "1/D", ylabel = "Order Parameters", lw = 2)

# Eplot = plot()
# plot!(Eplot, D, E, shape = :auto, label = "E",legend = :bottomright, xlabel = "1/D", ylabel = "E", lw = 2)

# ΔEplot = plot()
# plot!(ΔEplot, D, ΔE, shape = :auto, label = "ΔE",legend = :bottomright, xlabel = "1/D", ylabel = "ΔE", lw = 2)

# # Crossplot = plot()
# # plot!(Crossplot, D, Cross, shape = :auto, label = "Cross",legend = :bottomright, xlabel = "1/D", ylabel = "Cross norm", lw = 2)
# # savefig(magplot,"./plot/K_Γ_1x2_E-Γ-D$(D)_χ$(χ).svg")
@show zigzag stripy ferro mag Neel E ΔE Cross
