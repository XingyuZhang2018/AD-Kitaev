using ADBCVUMPS
using ADBCVUMPS:σx,σy,σz,buildbcipeps, optcont
using BCVUMPS
using BCVUMPS: ALCtoAC
using CUDA
using FileIO
using LinearAlgebra: norm, I, cross
using OMEinsum
using Plots
using Random
using Statistics: std
    
function read_xy(file)
    f = open(file, "r" )
    n = countlines(f)
    seekstart(f)
    X = zeros(n)
    Y = zeros(n)
    for i = 1:n
        x,y = split(readline(f), ",")
        X[i],Y[i] = parse.(Float64,[x,y])
    end
    X,Y
end

function read_Exy(file)
    f = open(file, "r" )
    n = countlines(f)
    seekstart(f)
    X = zeros(n)
    Y1 = zeros(n)
    Y2 = zeros(n)
    Y3 = zeros(n)
    Y4 = zeros(n)
    for i = 1:n
        x,e1,e2,e3,e4 = split(readline(f), "  ")
        # X[i],Y[i] = parse(Float64,x),min(parse(Float64,e1),parse(Float64,e2),parse(Float64,e3),parse(Float64,e4))
        X[i],Y1[i],Y2[i],Y3[i],Y4[i] = parse.(Float64,[x,e1,e2,e3,e4])
    end
    X,Y1,Y2,Y3,Y4
end

function observable(model, fdirection, field, type, folder, D, χ, tol, maxiter, miniter)
    if field == 0.0
        observable_log = folder*"$(model)/D$(D)_χ$(χ)_observable.log"
    else
        observable_log = folder*"$(model)_field$(fdirection)_$(field)$(type)/D$(D)_χ$(χ)_observable.log"
    end
    if isfile(observable_log)
        println("load observable from $(observable_log)")
        f = open(observable_log, "r" )
        mag, ferro, stripy, zigzag, Neel, etol, ΔE, Cross = parse.(Float64,split(readline(f), "   "))
    else
        bulk, key = init_ipeps(model, fdirection, field; folder = folder, type = type, atype = CuArray, D=D, χ=χ, tol=tol, maxiter=maxiter, miniter=miniter, verbose = true)
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
        ACu = ALCtoAC(ALu, Cu)
        ACd = ALCtoAC(ALd, Cd)

        M = Array{Array{ComplexF64,1},2}(undef, Nj, 2)
        Sx1 = reshape(ein"ab,cd -> acbd"(σx/2, I(2)), (4,4))
        Sx2 = reshape(ein"ab,cd -> acbd"(I(2), σx/2), (4,4))
        Sy1 = reshape(ein"ab,cd -> acbd"(σy/2, I(2)), (4,4))
        Sy2 = reshape(ein"ab,cd -> acbd"(I(2), σy/2), (4,4))
        Sz1 = reshape(ein"ab,cd -> acbd"(σz/2, I(2)), (4,4))
        Sz2 = reshape(ein"ab,cd -> acbd"(I(2), σz/2), (4,4))
        etol = 0.0
        for j = 1:Nj, i = 1:Ni
            jr = j + 1 - (j==Nj)*Nj
            ir = Ni + 1 - i
            lr3 = ein"(((aeg,abc),ehfbpq),ghi),cfi -> pq"(FL[i,j],ACu[i,j],ap[i,j],ACd[ir,j],FR[i,j])
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
            if field != 0.0
                etol -= (real(M[j,1] + M[j,2]))' * field / 2
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
        hx = reshape(permutedims(hx, (1,3,2,4)), (4,4))
        hy = reshape(ein"(ae,bfcg),dh -> abefcdgh"(I(2), hy, I(2)), (4,4,4,4))
        hz = reshape(ein"(ae,bfcg),dh -> abefcdgh"(I(2), hz, I(2)), (4,4,4,4))

        Ex, Ey, Ez = 0, 0, 0
        for j = 1:Nj, i = 1:Ni
            ir = Ni + 1 - i
            jr = j + 1 - (j==Nj) * Nj
            lr = oc1(FL[i,j],ACu[i,j],ap[i,j],ACd[ir,j],FR[i,jr],ARu[i,jr],ap[i,jr],ARd[ir,jr])
            ey = ein"pqrs, pqrs -> "(lr,hy)
            n = ein"pprr -> "(lr)
            Ey += Array(ey)[]/Array(n)[]
            println("hy = $(Array(ey)[]/Array(n)[])")
            etol += Array(ey)[]/Array(n)[]

            lr2 = ein"(((aeg,abc),ehfbpq),ghi),cfi -> pq"(FL[i,j],ACu[i,j],ap[i,j],ACd[ir,j],FR[i,j])
            ex = ein"pq, pq -> "(lr2,hx)
            n = Array(ein"pp -> "(lr2))[]
            Ex += Array(ex)[]/n
            println("hx = $(Array(ex)[]/n)")
            etol += Array(ex)[]/n
        end
        
        for j = 1:Nj, i = 1:Ni
            ir = i + 1 - Ni * (i==Ni)
            lr3 = oc2(ACu[i,j],FLu[i,j],ap[i,j],FRu[i,j],FL[ir,j],ap[ir,j],FR[ir,j],ACd[i,j])
            ez = ein"pqrs, pqrs -> "(lr3,hz)
            n = ein"pprr -> "(lr3)
            Ez += Array(ez)[]/Array(n)[]
            println("hz = $(Array(ez)[]/Array(n)[])") 
            etol += Array(ez)[]/Array(n)[]
        end
        println("e = $(etol/Ni/Nj)")
        etol = real(etol/(Ni * Nj))
        # ΔE = real(Ex - (Ey + Ez)/2)
        ΔE = std(real.([Ex, Ey, Ez]))
        @show ΔE

        message = "$(mag)   $(ferro)   $(stripy)   $(zigzag)   $(Neel)   $(etol)   $(ΔE)   $(Cross)\n"
        logfile = open(observable_log, "a")
        write(logfile, message)
        close(logfile)
    end
    return mag, ferro, stripy, zigzag, Neel, etol, ΔE, Cross
end

function fidelity(key1,key2)
    model1, fdirection1, field1, type1, folder1, D1, χ1, tol1, maxiter1, miniter1 = key1
    model2, fdirection2, field2, type2, folder2, D2, χ2, tol2, maxiter2, miniter2 = key2
    if field == 0.0
        fidelity_log = folder*"$(model1)/D$(D1)_χ$(χ1)_fidelity.log"
    else
        fidelity_log = folder*"$(model1)_field$(fdirection1)_$(field1)$(type1)/D$(D1)_χ$(χ1)_fidelity.log"
    end
    # if isfile(fidelity_log)
    #     println("load fidelity from $(fidelity_log)")
    #     f = open(fidelity_log, "r" )
    #     Ftol = parse.(Float64,readline(f))
    # else
        bulk1, key1 = init_ipeps(model1, fdirection1, field1; folder = folder1, type = type1, atype = CuArray, D=D1, χ=χ1, tol=tol1, maxiter=maxiter1, miniter=miniter1, verbose = true)
        folder1, model1, field1, atype1, D1, χ1, tol1, maxiter1, miniter1 = key1
        Ni = 1
        Nj = Int(size(bulk1,6) / Ni)
        bulk1 = buildbcipeps(bulk1,Ni,Nj)
        ap1 = [ein"abcdx,ijkly -> aibjckdlxy"(bulk1[i], conj(bulk1[i])) for i = 1:Ni*Nj]
        ap1 = [reshape(ap1[i], D^2, D^2, D^2, D^2, 4, 4) for i = 1:Ni*Nj]
        ap1 = reshape(ap1, Ni, Nj)
        a1 = [ein"ijklaa -> ijkl"(ap1[i]) for i = 1:Ni*Nj]
        a1 = reshape(a1, Ni, Nj)
        
        chkp_file_obs1 = folder1*"obs_D$(D1^2)_chi$(χ1).jld2"
        FL1, FR1 = load(chkp_file_obs1)["env"]
        chkp_file_up1 = folder1*"up_D$(D1^2)_χ$(χ1).jld2"                     
        rtup1 = SquareBCVUMPSRuntime(a1, chkp_file_up1, χ1; verbose = false)   
        ALu1, Cu1 = rtup1.AL, rtup1.C
        chkp_file_down1 = folder1*"down_D$(D1^2)_χ$(χ1).jld2"                              
        rtdown1 = SquareBCVUMPSRuntime(a1, chkp_file_down1, χ1; verbose = false)   
        ALd1,Cd1 = rtdown1.AL,rtdown1.C
        ACu1 = ALCtoAC(ALu1, Cu1)
        ACd1 = ALCtoAC(ALd1, Cd1) 

        bulk2, key2 = init_ipeps(model2, fdirection2, field2; folder = folder2, type = type2, atype = CuArray, D=D2, χ=χ2, tol=tol2, maxiter=maxiter2, miniter=miniter2, verbose = true)
        folder2, model2, field2, atype2, D2, χ2, tol2, maxiter2, miniter2 = key2
        bulk2 = buildbcipeps(bulk2,Ni,Nj)
        ap2 = [ein"abcdx,ijkly -> aibjckdlxy"(bulk2[i], conj(bulk2[i])) for i = 1:Ni*Nj]
        ap2 = [reshape(ap2[i], D^2, D^2, D^2, D^2, 4, 4) for i = 1:Ni*Nj]
        ap2 = reshape(ap2, Ni, Nj)
        a2 = [ein"ijklaa -> ijkl"(ap2[i]) for i = 1:Ni*Nj]
        a2 = reshape(a2, Ni, Nj)

        chkp_file_obs2 = folder2*"obs_D$(D2^2)_chi$(χ2).jld2"
        FL2, FR2 = load(chkp_file_obs2)["env"]
        chkp_file_up2 = folder2*"up_D$(D2^2)_χ$(χ2).jld2"
        rtup2 = SquareBCVUMPSRuntime(a2, chkp_file_up2, χ2; verbose = false)
        ALu2, Cu2 = rtup2.AL, rtup2.C
        chkp_file_down2 = folder2*"down_D$(D2^2)_χ$(χ2).jld2"
        rtdown2 = SquareBCVUMPSRuntime(a2, chkp_file_down2, χ2; verbose = false)
        ALd2, Cd2 = rtdown2.AL, rtdown2.C
        ACu2 = ALCtoAC(ALu2, Cu2)
        ACd2 = ALCtoAC(ALd2, Cd2)

        Ftol = 1
        for j = 1:Nj, i = 1:Ni
            ir = Ni + 1 - i
            lr1 = ein"(((aeg,abc),ehfbpq),ghi),cfi -> pq"(FL2[i,j],ACu2[i,j],ap2[i,j],ACd2[ir,j],FR2[i,j])
            n1 = ein"pp -> "(lr1)
            ρ = Array(lr1) / Array(n1)[]
            lr2 = ein"(((aeg,abc),ehfbpq),ghi),cfi -> pq"(FR1[i,j],ACu1[i,j],ap1[i,j],ACd1[ir,j],FL1[i,j])
            F = ein"pq,qp ->"(lr2,ρ)
            n2 = ein"pp -> "(lr2)
            Ftol *= Array(F)[]/Array(n2)[]
            @show Ftol
        end
        Ftol = sqrt(norm(Ftol))
        message = "$(Ftol)\n"
        logfile = open(fidelity_log, "w")
        write(logfile, message)
        close(logfile)
    # end
    return Ftol
end

function deriv_y(x,y)
    n = length(x)
    dy = zeros(n)
    for i = 1:n
        if i == 1
            dy[i] = (y[i+1] - y[i])/(x[i+1] - x[i])
        elseif i == n
            dy[i] = (y[i] - y[i-1])/(x[i] - x[i-1])
        else
            dy[i] = (y[i+1] - y[i-1])/(x[i+1] - x[i-1])
        end
    end
    return dy
end

model = K_J_Γ_Γ′(-1.0, -0.1, 0.3, -0.02)
fdirection = [1.0, 1.0, 0.825221]
type = "_random"
folder = "./../../../../data/xyzhang/ADBCVUMPS/K_J_Γ_Γ′_1x2/"
D = 4
χ = 80
tol = 1e-10
maxiter = 10
miniter = 2
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

# Random.seed!(100)
# folder, D, χ, tol, maxiter, miniter = "./../../../../data/xyzhang/ADBCVUMPS/K_J_Γ_Γ′_1x2/", 4, 80, 1e-10, 10, 5
# f = 0.61:0.01:0.8
# # f = 0.0:0.01:0.03
# fdirection = [1.0, 1.0, 0.985263]
# # 0.985263
# # 0.963424
# # 0.825221
# type = "_random"
# # Γ = 0.3
# field, mag, ferro, stripy, zigzag, Neel, E, ΔE, Cross = [], [], [], [], [], [], [], [], []
# for x in f
#     @show x
#     model = K_J_Γ_Γ′(-1.0, -0.1, 0.3, -0.02)
#     if x == 0.0
#         tfolder = folder*"$(model)/"
#     else
#         # if x > 0.13
#         #     type = "_random"
#         # else
#         #     type = "_zigzag"
#         # end
#         # type = ""
#         tfolder = folder*"$(model)_field$(fdirection)_$(x)$(type)/"
#         if isdir(tfolder)
#             y1, y2, y3, y4, y5, y6, y7, y8 = observable(model, fdirection, x, "$(type)", folder, D, χ, tol, maxiter, miniter)
#             field = [field; x]
#             mag = [mag; y1]
#             ferro = [ferro; y2]
#             stripy = [stripy; y3]
#             zigzag = [zigzag; y4]
#             Neel = [Neel; y5]
#             E = [E; y6]
#             ΔE = [ΔE; y7]
#             Cross = [Cross; y8]
#         end
#     end
# end

# magplot = plot()
# # plot!(magplot, field, mag, shape = :auto, title = "mag-h", label = "mag D = $(D)", lw = 2)
# plot!(magplot, field, ferro, shape = :auto, label = "0.4° mag D = $(D)", lw = 2)
# # plot!(magplot, field, stripy, shape = :auto, label = "stripy D = $(D)", lw = 2)
# # plot!(magplot, field, Neel, shape = :auto, label = "Neel D = $(D)", lw = 2)
# # plot!(magplot, field, zigzag, shape = :auto, label = "zigzag D = $(D)",legend = :outertop, xlabel = "h", ylabel = "Order Parameters", lw = 2)
# dferro = deriv_y(field, ferro)*1.5
# plot!(magplot, field, dferro, shape = :auto, label = "0.4° ∂mag D = $(D)", lw = 2)
# X,Y = read_xy(folder*"2021WeiLi-mag.log")
# plot!(magplot, X/187.782, Y, shape = :auto, label = "2021WeiLi-mag", lw = 2, rightmargin = 2.5Plots.cm)
# dmag = deriv_y(X/187.782, Y)/2
# plot!(magplot, X/187.782, dmag, shape = :auto, label = "2021WeiLi-dmag", lw = 2, rightmargin = 2.5Plots.cm)
# # X,Y = read_xy(folder*"2019Gordon-dmag.log")
# # # plot!(magplot, X, Y, shape = :auto, label = "2019Gordon 5° dmag",legend = :topright, xlabel = "h", ylabel = "Order Parameters",rightmargin = 1.5Plots.cm, lw = 2)

# # # ΔEplot = plot()
# ΔEplot = twinx()
# plot!(ΔEplot, field, abs.(ΔE), shape = :x, label = "ΔE D = $(D) ferro",legend = :topright, xlabel = "h", ylabel = "ΔE", lw = 2)

# Eplot = plot()
# plot!(Eplot, field, E, shape = :auto, label = "E 1x2 cell D = $(D) $(type)",legend = :bottomleft, xlabel = "h", ylabel = "E", lw = 2)
# dEplot = plot()
# dEplot = twinx()
# dE = deriv_y(field, E)
# plot!(dEplot, field, dE, shape = :auto, color = :red, label = "∂E 1x2 cell D = $(D)",legend = :bottomright, xlabel = "Γ/|K|", ylabel = "∂E", lw = 2, rightmargin = 2.5Plots.cm)
# ddEplot = twinx()
# ddE = deriv_y(field, dE)
# plot!(ddEplot, field, ddE, shape = :auto, label = "∂²E 1x2 cell D = $(D)", legend = :topright, xlabel = "Γ/|K|", ylabel = "∂²E", lw = 2)
# X,Y1,Y2,Y3,Y4 = read_Exy(folder*"2021WeiLi-E.log")
# plot!(Eplot, X*sqrt(3), Y1, shape = :auto, label = "2021WeiLi-fDMRG-YC4x12x2",legend = :topright, xlabel = "h", ylabel = "E", lw = 2)
# plot!(Eplot, X*sqrt(3), Y2, shape = :auto, label = "2021WeiLi-fDMRG-YC6x12x2",legend = :topright, xlabel = "h", ylabel = "E", lw = 2)
# plot!(Eplot, X*sqrt(3), Y3, shape = :auto, label = "2021WeiLi-iDMRG-YC4",legend = :topright, xlabel = "h", ylabel = "E", lw = 2)
# plot!(Eplot, X*sqrt(3), Y4, shape = :auto, label = "2021WeiLi-iDMRG-YC6",legend = :bottomleft, xlabel = "h", ylabel = "E", lw = 2)
# dE2 = deriv_y(X*sqrt(3), Y4)
# plot!(dEplot, X*sqrt(3), dE2, shape = :auto,  color = :red, label = "2021WeiLi-iDMRG-YC6 dE",legend =:topright, xlabel = "h", ylabel = "dE", lw = 2)


# # Crossplot = plot()
# # plot!(Crossplot, Γ, Cross, shape = :auto, label = "Cross D = $(D)",legend = :bottomright, xlabel = "Γ/|K|", ylabel = "Cross norm", lw = 2)
# # savefig(Eplot,"./plot/K_Γ_1x2&1x3_E-Γ-D$(D)_χ$(χ).svg")
# # @show zigzag stripy ferro mag Neel ΔE Cross
# for i in 1:length(dferro)
#     if dferro[i] > 0.5
#         dferro[i] = 0.5
#     end
# end