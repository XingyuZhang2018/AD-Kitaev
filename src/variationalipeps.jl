using FileIO
using LinearAlgebra: I, norm
using LineSearches
using OMEinsum: get_size_dict, optimize_greedy,  MinSpaceDiff
using Optim
using Printf: @sprintf
using TimerOutputs
using VUMPS
using CUDA

export init_ipeps, energy, optimiseipeps

"""
    energy(h, bcipeps; χ, tol, maxiter)

return the energy of the `bcipeps` 2-site hamiltonian `h` and calculated via a
BCVUMPS with parameters `χ`, `tol` and `maxiter`.
"""
function energy(h, bulk, oc, key; verbose = true, savefile = true)
    folder, _, _, atype, D, χ, tol, maxiter, miniter = key
    # bcipeps = indexperm_symmetrize(bcipeps)  # NOTE: this is not good
    Ni,Nj = size(bulk)
    ap = [ein"abcdx,ijkly -> aibjckdlxy"(bulk[i], conj(bulk[i])) for i = 1:Ni*Nj]
    ap = [reshape(ap[i], D^2, D^2, D^2, D^2, 4, 4) for i = 1:Ni*Nj]
    ap = reshape(ap, Ni, Nj)
    
    a = Zygote.Buffer(ap[1], D^2,D^2,D^2,D^2,Ni,Nj)
    for j in 1:Nj, i in 1:Ni
        a[:,:,:,:,i,j] = ein"ijklaa -> ijkl"(ap[i,j])
    end
    a = copy(a)

    env = obs_env(a; χ = χ, tol = tol, maxiter = maxiter, miniter = miniter, verbose = verbose, savefile = savefile, infolder = folder, outfolder = folder)
    e = expectationvalue(h, ap, env, oc, key)
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

function trans_to_arr_of_arr(A::AbstractArray{T,5}) where T
    reshape([A[:,:,:,1,j] for j in 1:2],1,2)
end

function trans_to_arr_of_arr(A::AbstractArray{T,4}) where T
    reshape([A[:,:,1,j] for j in 1:2],1,2)
end

"""
    expectationvalue(h, ap, env)

return the expectationvalue of a two-site operator `h` with the sites
described by rank-6 tensor `ap` each and an environment described by
a `SquareBCVUMPSRuntime` `env`.
"""
function expectationvalue(h, ap, env, oc, key)
    _, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, FLu, FRu = env
    _, _, field, atype, _, _, _, _, _ = key
    oc1, oc2 = oc
    Ni,Nj = size(ALu)[[4,5]]
    ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, FLu, FRu = map(trans_to_arr_of_arr, [ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, FLu, FRu])
    ACu = reshape([ein"asc,cb -> asb"(ALu[i],Cu[i]) for i=1:Ni*Nj],Ni,Nj)
    ACd = reshape([ein"asc,cb -> asb"(ALd[i],Cd[i]) for i=1:Ni*Nj],Ni,Nj)
    hx, hy, hz = h
    ap /= norm(ap)
    etol = 0
    Zygote.@ignore begin
        hx = atype(reshape(permutedims(hx, (1,3,2,4)), (4,4)))
        hy = atype(reshape(ein"ae,bfcg,dh -> abefcdgh"(I(2), hy, I(2)), (4,4,4,4)))
        hz = atype(reshape(ein"ae,bfcg,dh -> abefcdgh"(I(2), hz, I(2)), (4,4,4,4)))
    end

    for j = 1:Nj, i = 1:Ni
        ir = Ni + 1 - i
        jr = j + 1 - (j==Nj) * Nj
        lr = oc1(FL[i,j],ACu[i,j],ap[i,j],ACd[ir,j],FR[i,jr],ARu[i,jr],ap[i,jr],ARd[ir,jr])
        ey = ein"pqrs, pqrs -> "(lr,hy)
        n = ein"pprr -> "(lr)
        println("hy = $(Array(ey)[]/Array(n)[])")
        etol += Array(ey)[]/Array(n)[]

        lr2 = ein"(((aeg,abc),ehfbpq),ghi),cfi -> pq"(FL[i,j],ACu[i,j],ap[i,j],ACd[ir,j],FR[i,j])
        ex = ein"pq, pq -> "(lr2,hx)
        n = Array(ein"pp -> "(lr2))[]
        println("hx = $(Array(ex)[]/n)")
        etol += Array(ex)[]/n
    end
    
    for j = 1:Nj, i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        lr3 = oc2(ACu[i,j],FLu[i,j],ap[i,j],FRu[i,j],FL[ir,j],ap[ir,j],FR[ir,j],ACd[i,j])
        ez = ein"pqrs, pqrs -> "(lr3,hz)
        n = ein"pprr -> "(lr3)
        println("hz = $(Array(ez)[]/Array(n)[])") 
        etol += Array(ez)[]/Array(n)[]
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
            @show M1 M2
            etol -= (M1 + M2)' * field / 2
        end
    end

    println("e = $(etol/Ni/Nj)")
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

function buildbcipeps(bulk,Ni,Nj)
    bulk /= norm(bulk)
    reshape([bulk[:,:,:,:,:,i] for i = 1:Ni*Nj], (Ni, Nj))
end

"""
    init_ipeps(model::HamiltonianModel; D::Int, χ::Int, tol::Real, maxiter::Int)

Initial `bcipeps` and give `key` for use of later optimization. The key include `model`, `D`, `χ`, `tol` and `maxiter`. 
The iPEPS is random initial if there isn't any calculation before, otherwise will be load from file `/data/model_D_chi_tol_maxiter.jld2`
"""
function init_ipeps(model::HamiltonianModel, fdirection::Vector{Float64} = [0.0,0.0,0.0], field::Float64 = 0.0; folder::String="./data/", type::String = "", atype = Array, D::Int, χ::Int, tol::Real, maxiter::Int, miniter::Int, verbose = true)
    if field == 0.0
        folder *= "$(model)/"
    else
        folder *= "$(model)_field$(fdirection)_$(@sprintf("%0.2f", field))$(type)/"
        field = field * fdirection / norm(fdirection)
    end
    mkpath(folder)
    chkp_file = folder*"D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter)_miniter$(miniter).jld2"
    if isfile(chkp_file)
        bulk = load(chkp_file)["bcipeps"]
        verbose && println("load BCiPEPS from $chkp_file")
    else
        bulk = rand(ComplexF64,D,D,D,D,4,2)
        verbose && println("random initial BCiPEPS $chkp_file")
    end
    bulk /= norm(bulk)
    key = (folder, model, field, atype, D, χ, tol, maxiter, miniter)
    return bulk, key
end

"""
    optimiseipeps(bcipeps, h; χ, tol, maxiter, optimargs = (), optimmethod = LBFGS(m = 20))

return the tensor `bulk'` that describes an bcipeps that minimises the energy of the
two-site hamiltonian `h`. The minimization is done using `Optim` with default-method
`LBFGS`. Alternative methods can be specified by loading `LineSearches` and
providing `optimmethod`. Other options to optim can be passed with `optimargs`.
The energy is calculated using vumps with key include parameters `χ`, `tol` and `maxiter`.
"""
function optimiseipeps(bulk, key; f_tol = 1e-6, opiter = 100, verbose= false, optimmethod = LBFGS(m = 20))
    _, model, _, atype, D, χ, _, _, _ = key
    # h = atype(hamiltonian(model))
    h = hamiltonian(model)
    # h = (atype(hx),atype(hy),atype(hz))
    Ni, Nj = 1, 2
    to = TimerOutput()
    oc = optcont(D, χ)
    f(x) = @timeit to "forward" real(energy(h, buildbcipeps(atype(x),Ni,Nj), oc, key; verbose=verbose))
    ff(x) = real(energy(h, buildbcipeps(atype(x),Ni,Nj), oc, key; verbose=verbose))
    function g(x)
        @timeit to "backward" begin
            grad = Zygote.gradient(ff,atype(x))[1]
            # if norm(grad) > 1.0
            #     grad /= norm(grad)
            # end
            return grad
        end
    end
    res = optimize(f, g, 
        bulk, optimmethod, inplace = false,
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

    folder, model, field, atype, D, χ, tol, maxiter, miniter = key
    if !(key === nothing)
        logfile = open(folder*"D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter)_miniter$(miniter).log", "a")
        write(logfile, message)
        close(logfile)
        save(folder*"D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter)_miniter$(miniter).jld2", "bcipeps", os.metadata["x"])
    end
    return false
end