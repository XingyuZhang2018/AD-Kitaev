using ADBCVUMPS
using Test

@testset "ADBCVUMPS.jl" begin
    @testset "hamiltonianmodels" begin
        println("hamiltonianmodels tests running...")
        include("hamiltonianmodels.jl")
    end

    @testset "exampletensors" begin
        println("exampletensors tests running...")
        include("exampletensors.jl")
    end

    @testset "autodiff" begin
        println("autodiff tests running...")
        include("autodiff.jl")
    end

    @testset "bcipeps" begin
        println("bcipeps tests running...")
        include("bcipeps.jl")
    end

    @testset "exampleobs" begin
        println("exampleobs tests running...")
        include("exampleobs.jl")
    end

    @testset "variationalipeps" begin
        println("variationalipeps tests running...")
        include("variationalipeps.jl")
    end
end
