using Test
using ADBCVUMPS
using ADBCVUMPS:model_tensor
using BCVUMPS

@testset "$(Ni)x$(Nj) β = $(β) exampletensors" for Nj = 1:3, Ni = 1:3, β = 0.2:0.2:1.0
    @test model_tensor(Ising(Nj,Nj), β) ≈ BCVUMPS.model_tensor(Ising(Nj,Nj), β)
end