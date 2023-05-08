using AD_Kitaev
using ArgParse
using TeneT
using CUDA
using Random
using Test
using OMEinsum
using Optim
using Zygote
CUDA.allowscalar(false)

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--tol"
            help = "tol error for vumps"
            arg_type = Float64
            default = 1e-10
        "--maxiter"
            help = "max iterition for vumps"
            arg_type = Int
            default = 10
        "--miniter"
            help = "min iterition for vumps"
            arg_type = Int
            default = 1
        "--opiter"
            help = "iterition for optimise"
            arg_type = Int
            default = 200
        "--f_tol"
            help = "tol error for optimise"
            arg_type = Float64
            default = 1e-10
        "--K"
            help = "K"
            arg_type = Float64
            required = true
        "--J"
            help = "J"
            arg_type = Float64
            required = true
        "--Γ"
            help = "Γ"
            arg_type = Float64
            required = true
        "--Γ′"
            help = "Γ′"
            arg_type = Float64
            required = true
        "--field"
            help = "external field"
            arg_type = Float64
            required = true
        "--Ni"
            help = "Cell size Ni"
            arg_type = Int
            required = true
        "--Nj"
            help = "Cell size Nj"
            arg_type = Int
            required = true
        "--D"
            help = "ipeps virtual bond dimension"
            arg_type = Int
            required = true
        "--chi"
            help = "vumps virtual bond dimension"
            arg_type = Int
            required = true
        "--folder"
            help = "folder for output"
            arg_type = String
            default = "./data/"
        "--type"
            help = "initial type "
            arg_type = String
            default = ""
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    Random.seed!(100)
    K = parsed_args["K"]
    J = parsed_args["J"]
    Γ = parsed_args["Γ"]
    Γ′ = parsed_args["Γ′"]
    field = parsed_args["field"]
    Ni = parsed_args["Ni"]
    Nj = parsed_args["Nj"]
    D = parsed_args["D"]
    χ = parsed_args["chi"]
    tol = parsed_args["tol"]
    maxiter = parsed_args["maxiter"]
    miniter = parsed_args["miniter"]
    opiter = parsed_args["opiter"]
    f_tol = parsed_args["f_tol"]
    folder = parsed_args["folder"]
    type = parsed_args["type"]
    bulk, key = init_ipeps(K_J_Γ_Γ′(K,J,Γ,Γ′), [1.0,1.0,1.0], field; folder = folder, type = type, atype = CuArray, Ni=Ni, Nj=Nj, D=D, χ=χ, tol=tol, maxiter=maxiter, miniter=miniter)
    optimiseipeps(bulk, key; f_tol = f_tol, opiter = opiter, verbose = true)
end

main()