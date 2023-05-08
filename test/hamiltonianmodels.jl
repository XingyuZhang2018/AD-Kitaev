using Test
using AD_Kitaev

@testset "hamiltonianmodels" for Ni = [1,2,3], Nj = [1,2,3]
    @test TFIsing(Ni,Nj,1.0) isa HamiltonianModel
    @test Heisenberg(Ni,Nj) isa HamiltonianModel
    @test diaglocal(Ni,Nj,[1.,-1]) isa HamiltonianModel
    @test Kitaev() isa HamiltonianModel
    @test Kitaev_Heisenberg(0.1) isa HamiltonianModel
    @test K_J_Γ_Γ′(1.0,1.0,1.0,1.0) isa HamiltonianModel

    h1 = hamiltonian(Kitaev_Heisenberg(-90))
    h2 = hamiltonian(Kitaev(-1,-1,-1))
    for i = 1:3
        @test h1[i] ≈ h2[i]
    end

    h1 = hamiltonian(Kitaev_Heisenberg(90))
    h2 = hamiltonian(Kitaev(1,1,1))
    for i = 1:3
        @test h1[i] ≈ h2[i]
    end

    h1 = hamiltonian(K_J_Γ_Γ′(1.0,0.0,0.0,0.0))
    h2 = hamiltonian(Kitaev(1.0,1.0,1.0))
    for i = 1:3
        @test h1[i] ≈ h2[i]
    end

    h1 = hamiltonian(K_Γ(0.0))
    h2 = hamiltonian(Kitaev(-1.0,-1.0,-1.0))
    for i = 1:3
        @test h1[i] ≈ h2[i]
    end
end