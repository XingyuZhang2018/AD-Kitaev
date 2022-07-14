using Test
using ADBCVUMPS
using ADBCVUMPS:indexperm_symmetrize
using LinearAlgebra

@testset "$(Ni)x$(Nj) bcipeps" for Nj = 1:3, Ni = 1:3
    bcipeps = Array{Array,2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        bcipeps[i,j] = rand(3,3,3,3,2)
    end
    @test SquareBCIPEPS(bcipeps) isa BCIPEPS{}
    bcipeps[1,1] = rand(3,3,4,3,2)
    @test_throws DimensionMismatch SquareBCIPEPS(bcipeps)

    bcipeps[1,1] = rand(3,3,3,3,2)
    bcipeps = indexperm_symmetrize(SquareBCIPEPS(bcipeps))
    for j = 1:Nj, i = 1:Ni
        @test bcipeps.bulk[i,j] == 
            permutedims(bcipeps.bulk[i,j], (1,4,3,2,5)) == # up-down
            permutedims(bcipeps.bulk[i,j], (3,2,1,4,5)) == # left-right
            permutedims(bcipeps.bulk[i,j], (2,1,4,3,5)) == # diagonal
            permutedims(bcipeps.bulk[i,j], (4,3,2,1,5))    # rotation
    end
end
