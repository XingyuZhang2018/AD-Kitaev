using AD_Kitaev
using AD_Kitaev: energy, num_grad, optcont, buildbcipeps
using TeneT
using CUDA
using Random
using Test
using OMEinsum
using Optim
using Zygote
CUDA.allowscalar(false)

@testset "init_ipeps" for Ni = [1,2], Nj = [1,2], D in [2,3], χ in [10]
    model = Kitaev()
    A, key = init_ipeps(model; Ni=Ni, Nj=Nj, D=D, χ=χ);
    @test size(A) == (D,D,D,D,4,Ni*Nj)
end

@testset "energy" for Ni = [1], Nj = [1], D in [2,3], χ in [10], atype in [Array]
    model = Kitaev()
    A, key = init_ipeps(model; atype = atype, Ni=Ni, Nj=Nj, D=D, χ=χ, verbose = true)
    oc = optcont(D, χ)
    h = hamiltonian(model)
    @show energy(h, buildbcipeps(atype(A),Ni,Nj), oc, key; savefile = false)
end

@testset "gradient" for Ni = [1], Nj = [1], D in [2], χ in [4], atype in [Array]
    Random.seed!(100)
    model = Kitaev()
    oc = optcont(D, χ)
    h = hamiltonian(model)
    A, key = init_ipeps(model; atype = atype, Ni=Ni, Nj=Nj, D=D, χ=χ, verbose = false)
    foo(A) = real(energy(h, buildbcipeps(atype(A),Ni,Nj), oc, key; savefile = false))
    @test Zygote.gradient(foo, A)[1] ≈ num_grad(foo,A) atol = 1e-6 # num_grad for CuArray is slow
end

@testset "Kitaev" for Ni = [1], Nj = [1], D in [2], χ in [10], atype in [Array]
    Random.seed!(100)
    model = Kitaev()
    A, key = init_ipeps(model; atype = atype, D=D, χ=10)
    res = optimiseipeps(A, key; f_tol = 1e-6)
    e = minimum(res)
    @test e ≈ -0.1893749987397612 atol = 1e-6
end