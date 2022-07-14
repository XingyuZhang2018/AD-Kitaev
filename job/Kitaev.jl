using ADBCVUMPS
using ArgParse
using BCVUMPS
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
            default = 2
        "--opiter"
            help = "iterition for optimise"
            arg_type = Int
            default = 200
        "--f_tol"
            help = "tol error for optimise"
            arg_type = Float64
            default = 1e-10
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
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    Random.seed!(100)
    D = parsed_args["D"]
    χ = parsed_args["chi"]
    tol = parsed_args["tol"]
    maxiter = parsed_args["maxiter"]
    miniter = parsed_args["miniter"]
    opiter = parsed_args["opiter"]
    f_tol = parsed_args["f_tol"]
    folder = parsed_args["folder"]
    bulk, key = init_ipeps(Kitaev(-1.0,-1.0,-1.0); folder = folder, atype = CuArray, D=D, χ=χ, tol=tol, maxiter=maxiter, miniter=miniter)
    optimiseipeps(bulk, key; f_tol = f_tol, opiter = opiter, verbose = true)
end

main()