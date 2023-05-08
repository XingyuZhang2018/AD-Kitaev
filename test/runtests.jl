using AD_Kitaev
using Test

@testset "AD_Kitaev.jl" begin
    @testset "hamiltonianmodels" begin
        println("hamiltonianmodels tests running...")
        include("hamiltonianmodels.jl")
    end

    @testset "variationalipeps" begin
        println("variationalipeps tests running...")
        include("variationalipeps.jl")
    end
end
