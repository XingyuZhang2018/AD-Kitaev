using VUMPS: ALCtoAC
using Statistics: std, cross

export fidelity, observable

function observable(model, fdirection, field, type, folder, atype, D, χ, tol, maxiter, miniter)
    if field == 0.0
        observable_log = folder*"$(model)/D$(D)_χ$(χ)_observable.log"
    else
        observable_log = folder*"$(model)_field$(fdirection)_$(@sprintf("%0.2f", field))$(type)/D$(D)_χ$(χ)_observable.log"
    end
    if isfile(observable_log)
        println("load observable from $(observable_log)")
        f = open(observable_log, "r" )
        mag, ferro, stripy, zigzag, Neel, etol, ΔE, Cross = parse.(Float64,split(readline(f), "   "))
    else
        bulk, key = init_ipeps(model, fdirection, field; folder = folder, type = type, atype = atype, D=D, χ=χ, tol=tol, maxiter=maxiter, miniter=miniter, verbose = true)
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
        
        chkp_file_obs = folder*"obs_D$(D^2)_χ$(χ).jld2"
        FL, FR = load(chkp_file_obs)["env"]
        chkp_file_up = folder*"up_D$(D^2)_χ$(χ).jld2"                     
        rtup = SquareVUMPSRuntime(a, chkp_file_up, χ; verbose = false)   
        FLu, FRu, ALu, ARu, Cu = rtup.FL, rtup.FR, rtup.AL, rtup.AR, rtup.C
        chkp_file_down = folder*"down_D$(D^2)_χ$(χ).jld2"                              
        rtdown = SquareVUMPSRuntime(a, chkp_file_down, χ; verbose = false)   
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
