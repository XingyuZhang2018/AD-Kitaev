using FileIO
using LinearAlgebra: I, norm
using LineSearches
using OMEinsum: get_size_dict, optimize_greedy,  MinSpaceDiff
using Optim
using Printf: @sprintf
using TimerOutputs
using TeneT
using TeneT: ALCtoAC
using CUDA

export init_ipeps, energy, optimiseipeps
"""
   │     │                 a     b 
───┼─────┼───           c ─┼─ d ─┼─ e
   │   ╱ │                 │ ╲   │
   │  ╱  │                 f  m  g 
   │ ╱   │                 │   ╲ │
───┼─────┼───           h ─┼─ i ─┼─ j
   │     │                 k     l
"""
function buildM(ipeps, atype)
    D = size(ipeps,1)
    d = size(ipeps,5)
    ID = Matrix{Float64}(I, D, D)
    Id = Matrix{Float64}(I, d, d)
    M11 = ein"ae, bf, cd -> abcdef"(ID, ID, Id)
    M11 = reshape(M11, D, D*d, D*d, D)
    M12 = permutedims(conj(ipeps), (5,1,2,3,4))
    M12 = reshape(M12, D*d, D, D, D)
    M21 = reshape(ipeps, D, D, D, D*d)
    M22 = ein"ac, bd -> abcd"(ID, ID)
    reshape([atype(M11), atype(M21), atype(M12), atype(M22)], 2,2)
end


"""
    energy(h, ipeps, oc, key; savefile = true, show_every = Inf)

return the energy of the `bcipeps` 2-site hamiltonian `h` and calculated via a
TeneT with parameters `χ`, `tol` and `maxiter`.
"""
function energy(h, ipeps, oc, key; savefile = true, show_every = Inf)
    folder, _, _, atype, Ni, Nj, D, χ, tol, maxiter, miniter, ifcheckpoint, verbose = key
    # bcipeps = indexperm_symmetrize(bcipeps)  # NOTE: this is not good
    ipeps /= norm(ipeps)
    ipeps = reshape([ipeps[:,:,:,:,:,i] for i = 1:Ni*Nj], (Ni, Nj))
    ap = [ein"abcdx,ijkly -> aibjckdlxy"(ipeps[i], conj(ipeps[i])) for i = 1:Ni*Nj]
    ap = [reshape(ap[i], D^2, D^2, D^2, D^2, 4, 4) for i = 1:Ni*Nj]
    ap = reshape(ap, Ni, Nj)
    
    # a = Zygote.Buffer(ap, Ni,Nj)
    # for j in 1:Nj, i in 1:Ni
    #     a[i,j] = ein"ijklaa -> ijkl"(ap[i,j])
    # end
    # a = copy(a)

    # a = reshape([ein"ijklaa -> ijkl"(ap[i]) for i in 1:Ni*Nj], Ni, Nj)
    a = buildM(ipeps[1], atype)

    env = obs_env(a; χ = χ, tol = tol, maxiter = maxiter, miniter = miniter, verbose = verbose, savefile = savefile, infolder = folder, outfolder = folder, savetol = 1, show_every = show_every)
    e = ifcheckpoint ? checkpoint(expectationvalue, h, ap, env, oc, key) : expectationvalue(h, ap, env, oc, key)
    return e
end

"""
    oc1, oc2 = optcont(D::Int, χ::Int)

optimise the follow two einsum contractions for the given `D` and `χ` which are used to calculate the energy of the 2-site hamiltonian:

```
                                            a ────┬──── c          
a ────┬──c ──┬──── f                        │     b     │  
│     b      e     │                        ├─ e ─┼─ f ─┤  
├─ g ─┼─  h ─┼─ i ─┤                        g     h     i 
│     k      n     │                        ├─ j ─┼─ k ─┤ 
j ────┴──l ──┴──── o                        │     m     │ 
                                            l ────┴──── n 
```
where the central two block are six order tensor have extra bond `pq` and `rs`
"""
function optcont(D::Int, χ::Int)
    sd = Dict('a' => χ, 'b' => D^2,'c' => χ, 'e' => D^2, 'f' => χ, 'g' => D^2, 'h' => D^2, 'i' => D^2, 'j' => χ, 'k' => D^2, 'l' => χ, 'n' => D^2, 'o' => χ, 'p' => 4, 'q' => 4, 'r' => 4, 's' => 4)
    oc1 = optimize_greedy(ein"agj,abc,gkhbpq,jkl,fio,cef,hniers,lno -> pqrs", sd; method=MinSpaceDiff())
    sd = Dict('a' => χ, 'b' => D^2, 'c' => χ, 'e' => D^2, 'f' => D^2, 'g' => χ, 'h' => D^2, 'i' => χ, 'j' => D^2, 'k' => D^2, 'l' => χ, 'm' => D^2, 'n' => χ, 'r' => 4, 's' => 4, 'p' => 4, 'q' => 4)
    oc2 = optimize_greedy(ein"abc,aeg,ehfbpq,cfi,gjl,jmkhrs,ikn,lmn -> pqrs", sd; method=MinSpaceDiff())
    oc1, oc2
end


"""
    expectationvalue(h, ap, env)

return the expectationvalue of a two-site operator `h` with the sites
described by rank-6 tensor `ap` each and an environment described by
a `SquareBCVUMPSRuntime` `env`.
"""
function expectationvalue(h, ap, env, oc, key)
    _, ALu, Cu, ARu, ALd, Cd, ARd, FLo, FRo, FLu, FRu = env
    _, _, field, atype, Ni, Nj, _, _, _, _, _, _, verbose = key
    oc1, oc2 = oc
    ACu = ALCtoAC(ALu, Cu)
    ACd = ALCtoAC(ALd, Cd)
    hx, hy, hz = h
    ap /= norm(ap)
    etol = 0
    Zygote.@ignore begin
        hx = atype(reshape(permutedims(hx, (1,3,2,4)), (4,4)))
        hy = atype(reshape(ein"ae,bfcg,dh -> abefcdgh"(I(2), hy, I(2)), (4,4,4,4)))
        hz = atype(reshape(ein"ae,bfcg,dh -> abefcdgh"(I(2), hz, I(2)), (4,4,4,4)))
    end

    χ, D, _ = size(ALu[1,1])
    LARu = reshape(ein"adb, bec -> adec"(ARu[1,1],ARu[1,2]), (χ, D^2, χ))
    LARd = reshape(ein"adb, bec -> adec"(ARd[1,1],ARd[1,2]), (χ, D^2, χ))
    LACu = reshape(ein"adb,bf,fec -> adec"(ALu[1,1],Cu[1,1],ARu[1,2]), (χ, D^2, χ))
    LACd = reshape(ein"adb,bf,fec -> adec"(ALd[1,1],Cd[1,1],ARd[1,2]), (χ, D^2, χ))

    LFLu = reshape(ein"adb, bec -> aedc"(FLu[1,1],FLu[2,1]), (χ, D^2, χ))
    LFRu = reshape(ein"adb, bec -> aedc"(FRu[1,2],FRu[2,2]), (χ, D^2, χ))
    LFLo = reshape(ein"adb, bec -> aedc"(FLu[1,1],FLo[2,1]), (χ, D^2, χ))
    LFRo = reshape(ein"adb, bec -> aedc"(FRu[1,2],FRo[2,2]), (χ, D^2, χ))

    for j = 1:Nj, i = 1:Ni
        verbose && println("===========$i,$j===========")
        ir = Ni + 1 - i
        jr = j + 1 - (j==Nj) * Nj
        lr = oc1(LFLo,LACu,ap[1],conj(LACd),LFRo,LARu,ap[1],conj(LARd))

        e = Array(ein"pqrs, pqrs -> "(lr,hz))[]
        n = Array(ein"pprr -> "(lr))[]
        verbose && println("hz = $(e/n)")
        etol += e/n

        lr = ein"(((aeg,abc),ehfbpq),ghi),cfi -> pq"(LFLo,LACu,ap[1],conj(LACd),LFRo)
        e = Array(ein"pq, pq -> "(lr,hx))[]
        n = Array(ein"pp -> "(lr))[]
        verbose && println("hx = $(e/n)")
        etol += e/n

        ir  =  i + 1 - (i==Ni) * Ni
        irr = Ni - i + (i==Ni) * Ni
        lr = oc2(LACu,LFLu,ap[1],LFRu,LFLo,ap[1],LFRo,conj(LACd))
        e = Array(ein"pqrs, pqrs -> "(lr,hy))[]
        n =  Array(ein"pprr -> "(lr))[]
        verbose && println("hy = $(e/n)")
        etol += e/n
    end
    
    if field != 0.0
        Sx1, Sx2, Sy1, Sy2, Sz1, Sz2 = [],[],[],[],[],[]
        Zygote.@ignore begin
            Sx1 = reshape(ein"ab,cd -> acbd"(σx/2, I(2)), (4,4))
            Sx2 = reshape(ein"ab,cd -> acbd"(I(2), σx/2), (4,4))
            Sy1 = reshape(ein"ab,cd -> acbd"(σy/2, I(2)), (4,4))
            Sy2 = reshape(ein"ab,cd -> acbd"(I(2), σy/2), (4,4))
            Sz1 = reshape(ein"ab,cd -> acbd"(σz/2, I(2)), (4,4))
            Sz2 = reshape(ein"ab,cd -> acbd"(I(2), σz/2), (4,4))
        end
        for j = 1:Nj, i = 1:Ni
            ir = Ni + 1 - i
            lr3 = ein"(((aeg,abc),ehfbpq),ghi),cfi -> pq"(FL[i,j],ACu[i,j],ap[i,j],ACd[ir,j],FR[i,j])
            Mx1 = ein"pq, pq -> "(lr3,atype(Sx1))
            Mx2 = ein"pq, pq -> "(lr3,atype(Sx2))
            My1 = ein"pq, pq -> "(lr3,atype(Sy1))
            My2 = ein"pq, pq -> "(lr3,atype(Sy2))
            Mz1 = ein"pq, pq -> "(lr3,atype(Sz1))
            Mz2 = ein"pq, pq -> "(lr3,atype(Sz2))
            n3 = Array(ein"pp -> "(lr3))[]
            M1 = [Array(Mx1)[]/n3, Array(My1)[]/n3, Array(Mz1)[]/n3]
            M2 = [Array(Mx2)[]/n3, Array(My2)[]/n3, Array(Mz2)[]/n3]
            verbose && (@show M1 M2)
            etol -= (M1 + M2)' * field / 2
        end
    end

    verbose && println("e = $(etol/Ni/Nj)")
    return etol/Ni/Nj
end

"""
    ito12(i,Ni)

checkerboard pattern
```
    │    │   
  ──A────B──  
    │    │   
  ──B────A──
    │    │   
```
"""
ito12(i,Ni) = mod(mod(i,Ni) + Ni*(mod(i,Ni)==0) + fld(i,Ni) + 1 - (mod(i,Ni)==0), 2) + 1



"""
    init_ipeps(model::HamiltonianModel; D::Int, χ::Int, tol::Real, maxiter::Int)

Initial `bcipeps` and give `key` for use of later optimization. The key include `model`, `D`, `χ`, `tol` and `maxiter`. 
The iPEPS is random initial if there isn't any calculation before, otherwise will be load from file `/data/model_D_chi_tol_maxiter.jld2`
"""
function init_ipeps(model::HamiltonianModel, 
                    fdirection::Vector{Float64} = [0.0,0.0,0.0], 
                    field::Float64 = 0.0; 
                    folder::String="./data/", 
                    type::String = "", 
                    atype = Array, 
                    Ni::Int = 1, 
                    Nj::Int = 1, 
                    D::Int, χ::Int, 
                    tol::Real = 1e-10, 
                    maxiter::Int = 10, 
                    miniter::Int = 1, 
                    ifcheckpoint = false,
                    verbose = true
                    )
                    
    if field == 0.0
        folder *= "$(Ni)x$(Nj)/$(model)/"
    else
        folder *= "$(Ni)x$(Nj)/$(model)_field$(fdirection)_$(@sprintf("%0.2f", field))$(type)/"
        field = field * fdirection / norm(fdirection)
    end
    mkpath(folder)
    chkp_file = folder*"D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter)_miniter$(miniter).jld2"
    if isfile(chkp_file)
        ipeps = load(chkp_file)["bcipeps"]
        verbose && println("load BCiPEPS from $chkp_file")
    else
        ipeps = rand(ComplexF64,D,D,D,D,4,Ni*Nj)
        verbose && println("random initial BCiPEPS $chkp_file")
    end
    ipeps /= norm(ipeps)
    key = (folder, model, field, atype, Ni, Nj, D, χ, tol, maxiter, miniter, ifcheckpoint, verbose)
    return ipeps, key
end

"""
    optimiseipeps(bcipeps, h; χ, tol, maxiter, optimargs = (), optimmethod = LBFGS(m = 20))

return the tensor `ipeps'` that describes an bcipeps that minimises the energy of the
two-site hamiltonian `h`. The minimization is done using `Optim` with default-method
`LBFGS`. Alternative methods can be specified by loading `LineSearches` and
providing `optimmethod`. Other options to optim can be passed with `optimargs`.
The energy is calculated using vumps with key include parameters `χ`, `tol` and `maxiter`.
"""
function optimiseipeps(ipeps, key; 
                       f_tol = 1e-6, opiter = 100, 
                       maxiter_ad = 10,
                       miniter_ad = 1,
                       verbose = false, 
                       optimmethod = LBFGS(m = 20)
                       )

    folder, model, field, atype, Ni, Nj, D, χ, tol, maxiter, miniter, ifcheckpoint, verbose = key
    keyback = folder, model, field, atype, Ni, Nj, D, χ, tol, maxiter_ad, miniter_ad, ifcheckpoint, verbose
    h = hamiltonian(model)
    to = TimerOutput()
    oc = optcont(D, χ)
    f(x) = @timeit to "forward" real(energy(h, atype(x), oc, key))
    ff(x) = real(energy(h, atype(x), oc, keyback))
    function g(x)
        @timeit to "backward" begin
            println("for backward convergence:")
            f(x)
            println("true backward:")
            grad = Zygote.gradient(ff,atype(x))[1]
            # if norm(grad) > 1.0
            #     grad /= norm(grad)
            # end
            return grad
        end
    end
    res = optimize(f, g, 
        ipeps, optimmethod, inplace = false,
        Optim.Options(f_tol=f_tol, iterations=opiter,
        extended_trace=true,
        callback=os->writelog(os, key)),
        )
    println(to)
    return res
end

"""
    writelog(os::OptimizationState, key=nothing)

return the optimise infomation of each step, including `time` `iteration` `energy` and `g_norm`, saved in `/data/model_D_chi_tol_maxiter.log`. Save the final `bcipeps` in file `/data/model_D_chi_tol_maxiter.jid2`
"""
function writelog(os::OptimizationState, key=nothing)
    message = "$(round(os.metadata["time"],digits=2))   $(os.iteration)   $(os.value)   $(os.g_norm)\n"

    printstyled(message; bold=true, color=:red)
    flush(stdout)

    folder, model, field, atype, Ni, Nj, D, χ, tol, maxiter, miniter, ifcheckpoint, verbose = key
    if !(key === nothing)
        logfile = open(folder*"D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter)_miniter$(miniter).log", "a")
        write(logfile, message)
        close(logfile)
        save(folder*"D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter)_miniter$(miniter).jld2", "bcipeps", os.metadata["x"])
    end
    return false
end