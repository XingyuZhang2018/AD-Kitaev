@kwdef mutable struct iPEPSOptimize{F}
    boundary_alg::VUMPS
    reuse_env::Bool = Defaults.reuse_env
    verbosity::Int = Defaults.verbosity
    maxiter::Int = Defaults.fpgrad_maxiter
    tol::Real = Defaults.fpgrad_tol
    optimizer = Defaults.optimizer
    folder::String = Defaults.folder
    show_every::Int = Defaults.show_every
    save_every::Int = Defaults.save_every
    ifflatten::Bool = true
    ifcheckpoint::Bool = false
end

"""
    energy(h, bcipeps; χ, tol, maxiter)
return the energy of the `bcipeps` 2-site hamiltonian `h` and calculated via a
BCVUMPS with parameters `χ`, `tol` and `maxiter`.
"""
function energy(A, model, Dz, rt, params::iPEPSOptimize)
    A = bulid_A(A, params)
    M = bulid_M(A, params)
    rt′ = leading_boundary(rt, M, params.boundary_alg)
    Zygote.@ignore params.reuse_env && update!(rt, rt′)
    env = VUMPSEnv(rt′, M, params.boundary_alg)
    return energy_value(model, Dz, A, env, params)
end

"""
    optimise_ipeps(A::AbstractArray, key; f_tol = 1e-6, opiter = 100, optimmethod = LBFGS(m = 20))

return the tensor `A'` that describes an ipeps that minimises the energy of the
two-site hamiltonian `h`. The minimization is done using `Optim` with default-method
`LBFGS`. Alternative methods can be specified by loading `LineSearches` and
providing `optimmethod`. Other options to optim can be passed with `optimargs`.
The energy is calculated using vumps with key include parameters `χ`, `tol` and `maxiter`.
"""
function optimise_ipeps(A::AbstractArray, model, χ::Int, params::iPEPSOptimize; ifWp=false, Dz::Real=0.0)
    D = size(A, 1)
    A = restriction_ipeps(A)
    if ifWp
        Wp = _arraytype(A)(bulid_Wp(model.S, params))
        A′ = bulid_A(A, Wp, params)
        A′ = bulid_A(A′, params)
        M = bulid_M(A′, params)
    else
        A′ = bulid_A(A, params)
        M = bulid_M(A′, params)
    end

    rt = VUMPSRuntime(M, χ, params.boundary_alg)

    function f(A) 
        A = restriction_ipeps(A)
        ifWp && (A = bulid_A(A, Wp, params))
        return params.ifcheckpoint ? real(checkpoint(energy, A, model, Dz, rt, params)) : real(energy(A, model, Dz, rt, params))
    end
    function g(A)
        # f(x)
        grad = Zygote.gradient(f,A)[1]
        return grad
    end
    res = optimize(f, g, 
        A, params.optimizer, inplace = false,
        Optim.Options(f_tol=params.tol, 
                      iterations=params.maxiter,
                      extended_trace=true,
                      callback=os->writelog(os, params, D, χ)
        )
    )
    return res
end

"""
    writelog(os::OptimizationState, key=nothing)

return the optimise infomation of each step, including `time` `iteration` `energy` and `g_norm`, saved in `/data/model_D_chi_tol_maxiter.log`. Save the final `ipeps` in file `/data/model_D_chi_tol_maxiter.jid2`
"""
function writelog(os::OptimizationState, params::iPEPSOptimize, D::Int, χ::Int)
    @unpack folder = params

    message = @sprintf("i = %5d\tt = %0.2f sec\tenergy = %.15f \tgnorm = %.3e\n", os.iteration, os.metadata["time"], os.value, os.g_norm)

    folder = joinpath(folder, "D$(D)_χ$(χ)")
    !(ispath(folder)) && mkpath(folder)
    if params.verbosity >= 3 && os.iteration % params.show_every == 0
        printstyled(message; bold=true, color=:red)
        flush(stdout)

        logfile = open(joinpath(folder, "history.log"), "a")
        write(logfile, message)
        close(logfile)
    end
    if params.save_every != 0 && os.iteration % params.save_every == 0
        save(joinpath(folder, "ipeps", "ipeps_No.$(os.iteration).jld2"), "bcipeps", Array(os.metadata["x"]))
    end

    return false
end