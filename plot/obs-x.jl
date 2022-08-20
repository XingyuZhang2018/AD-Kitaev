using ADBCVUMPS
using CUDA
using FileIO
using Printf: @sprintf
using Random

Random.seed!(100)
folder, atype, D, χ, tol, maxiter, miniter, Ni, Nj = "/data/xyzhang/ADBCVUMPS/", CuArray, 4, 80, 1e-10, 10, 1, 3, 3
f = [0.0]
fdirection = [1.0, 1.0, 1.0]
# 0.985263
# 0.963424
# 0.825221
type = "_zigzag"
field, mag, ferro, stripy, zigzag, Neel, E, ΔE, Cross = [], [], [], [], [], [], [], [], []
for x in f
    @show x
    model = K_J_Γ_Γ′(-1.0, -0.0, 1.0, -0.0)
    if x == 0.0
        tfolder = folder*"$(Ni)x$(Nj)/$(model)/"
    else
        # if x > 0.13
        #     type = "_random"
        # else
        #     type = "_zigzag"
        # end
        # type = ""
        tfolder = folder*"$(Ni)x$(Nj)/$(model)_field$(fdirection)_$(@sprintf("%0.2f", x))$(type)/"
    end
    if isdir(tfolder)
        y1, y2, y3, y4, y5, y6, y7, y8 = observable(model, fdirection, x, "$(type)", folder, atype, D, χ, tol, maxiter, miniter, Ni, Nj)
        # field = [field; x]
        # mag = [mag; y1]
        # ferro = [ferro; y2]
        # stripy = [stripy; y3]
        # zigzag = [zigzag; y4]
        # Neel = [Neel; y5]
        # E = [E; y6]
        global ΔE = [ΔE; y7]
        # Cross = [Cross; y8]
    end
end

@show ΔE

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