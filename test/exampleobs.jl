using ADBCVUMPS
using ADBCVUMPS:num_grad
using BCVUMPS
using BCVUMPS:obs_bcenv,magnetisation,Z,magofdβ,bcvumps_env
using CUDA
using Random
using Test
using Zygote

@testset "$(Ni)x$(Nj) ising with $atype" for atype in [Array, CuArray], Ni = [2], Nj = [2]
    Random.seed!(100)
    model = Ising(Ni,Nj)
    function foo1(β) 
        M = model_tensor(model, β; atype = atype)
        env = obs_bcenv(model, M; atype = atype, χ = 10, miniter = 2, verbose = true, updown = false)
        magnetisation(env,model,β)
    end

    function foo2(β) 
        M = model_tensor(model, β; atype = atype)
        env = obs_bcenv(model, M; atype = atype, χ = 10, miniter = 2, verbose = true, updown = true)
        magnetisation(env,model,β)
    end
    
    for β = 0.2:0.2:0.8
        @test isapprox(Zygote.gradient(foo1,β)[1], magofdβ(model,β), atol = 1e-5)
    end

    for β = 0.1:0.1:0.4
        @test isapprox(Zygote.gradient(foo2,β)[1], magofdβ(model,β), atol = 1e-5)
    end
end

@testset "J1-J2-2x2-ising with $atype" for atype in [Array], Ni = [2], Nj = [2]
    Random.seed!(100)
    model = Ising22(2.0)
    β = 0.3
    function foo1(β)
        M = model_tensor(model, β; atype = atype)
        env = obs_bcenv(model, M; atype = atype, χ = 10, miniter = 10, verbose = true, updown = true)
        magnetisation(env,model,β)
    end
    @test isapprox(Zygote.gradient(foo1,β)[1], num_grad(foo1,β; δ=1e-2), atol = 1e-5)
end