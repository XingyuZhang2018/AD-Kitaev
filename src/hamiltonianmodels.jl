export diaglocal, TFIsing, Heisenberg, Kitaev, Kitaev_Heisenberg, K_J_Γ_Γ′, K_Γ
export hamiltonian, HamiltonianModel

const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -1im; 1im 0]
const σz = ComplexF64[1 0; 0 -1]
const id2 = ComplexF64[1 0; 0 1]

abstract type HamiltonianModel end

@doc raw"
    hamiltonian(model<:HamiltonianModel)

return the hamiltonian of the `model` as a two-site tensor operator.
"
function hamiltonian end
struct diaglocal{T<:Vector} <: HamiltonianModel 
    Ni::Int
    Nj::Int
    diag::T
end

"""
    diaglocal(diag::Vector)

return the 2-site Hamiltonian with single-body terms given
by the diagonal `diag`.
"""
function hamiltonian(model::diaglocal)
    diag = model.diag
    n = length(diag)
    h = ein"i -> ii"(diag)
    id = Matrix(I,n,n)
    reshape(h,n,n,1,1) .* reshape(id,1,1,n,n) .+ reshape(h,1,1,n,n) .* reshape(id,n,n,1,1)
end

@doc raw"
    TFIsing(hx::Real)

return a struct representing the transverse field ising model with magnetisation `hx`.
"
struct TFIsing{T<:Real} <: HamiltonianModel
    Ni::Int
    Nj::Int
    hx::T
end

"""
    hamiltonian(model::TFIsing)

return the transverse field ising hamiltonian for the provided `model` as a
two-site operator.
"""
function hamiltonian(model::TFIsing)
    hx = model.hx
    -2 * ein"ij,kl -> ijkl"(σz,σz) -
        hx/2 * ein"ij,kl -> ijkl"(σx, id2) -
        hx/2 * ein"ij,kl -> ijkl"(id2, σx)
end

@doc raw"
    Heisenberg(Jz::T,Jx::T,Jy::T) where {T<:Real}

return a struct representing the heisenberg model with magnetisation fields
`Jz`, `Jx` and `Jy`..
"
struct Heisenberg{T<:Real} <: HamiltonianModel
    Ni::Int
    Nj::Int
    Jz::T
    Jx::T
    Jy::T
end
Heisenberg(Ni,Nj) = Heisenberg(Ni,Nj,1.0,1.0,1.0)

"""
    hamiltonian(model::Heisenberg)

return the heisenberg hamiltonian for the `model` as a two-site operator.
"""
function hamiltonian(model::Heisenberg)
    h = model.Jz * ein"ij,kl -> ijkl"(σz, σz) -
        model.Jx * ein"ij,kl -> ijkl"(σx, σx) -
        model.Jy * ein"ij,kl -> ijkl"(σy, σy)
    # h = ein"ijcd,kc,ld -> ijkl"(h,σx,σx')
    h / 8
end

@doc raw"
    Kitaev(Jx::T,Jz::T,Jy::T) where {T<:Real}

return a struct representing the Kitaev model with magnetisation fields
`Jx`, `Jy` and `Jz`..
"
struct Kitaev{T<:Real} <: HamiltonianModel
    Jz::T
    Jx::T
    Jy::T
end
Kitaev() = Kitaev(-1.0, -1.0, -1.0)

"""
    hamiltonian(model::Kitaev)

return the Kitaev hamiltonian for the `model` as a two-site operator.
"""
function hamiltonian(model::Kitaev)
    hx = model.Jx * ein"ij,kl -> ijkl"(σx, σx)
    hy = model.Jy * ein"ij,kl -> ijkl"(σy, σy)
    hz = model.Jz * ein"ij,kl -> ijkl"(σz, σz)
    hx / 8, hy / 8, hz / 8
end

@doc raw"
    Kitaev_Heisenberg{T<:Real} <: HamiltonianModel

return a struct representing the Kitaev_Heisenberg model with interaction factor
`ϕ` degree
"
struct Kitaev_Heisenberg{T<:Real} <: HamiltonianModel
    ϕ::T
end

"""
    hamiltonian(model::Kitaev_Heisenberg)

return the Kitaev_Heisenberg hamiltonian for the `model` as a two-site operator.
"""
function hamiltonian(model::Kitaev_Heisenberg)
    op = ein"ij,kl -> ijkl"
    Heisenberg = cos(model.ϕ / 180 * pi) / 2 * (op(σz, σz) +
                                 op(σx, σx) +
                                 op(σy, σy) )
    hx = Heisenberg + sin(model.ϕ / 180 * pi) * op(σx, σx)
    hy = Heisenberg + sin(model.ϕ / 180 * pi) * op(σy, σy)
    hz = Heisenberg + sin(model.ϕ / 180 * pi) * op(σz, σz)
    hx / 8, hy / 8, hz / 8
end

@doc raw"
    K_J_Γ_Γ′{T<:Real} <: HamiltonianModel

return a struct representing the K_J_Γ_Γ′ model
"
struct K_J_Γ_Γ′{T<:Real} <: HamiltonianModel
    K::T
    J::T
    Γ::T
    Γ′::T
end

"""
    hamiltonian(model::K_J_Γ_Γ)

return the K_J_Γ_Γ′ hamiltonian for the `model` as a two-site operator.
"""
function hamiltonian(model::K_J_Γ_Γ′)
    op = ein"ij,kl -> ijkl"
    Heisenberg = model.J * (op(σz, σz) +
                            op(σx, σx) +
                            op(σy, σy) )
    hx = Heisenberg + model.K * op(σx, σx) + model.Γ * (op(σy, σz) + op(σz, σy)) + model.Γ′ * (op(σx, σy) + op(σy, σx) + op(σz, σx) + op(σx, σz))
    hy = Heisenberg + model.K * op(σy, σy) + model.Γ * (op(σx, σz) + op(σz, σx)) + model.Γ′ * (op(σy, σx) + op(σx, σy) + op(σz, σy) + op(σy, σz))
    hz = Heisenberg + model.K * op(σz, σz) + model.Γ * (op(σx, σy) + op(σy, σx)) + model.Γ′ * (op(σz, σx) + op(σx, σz) + op(σy, σz) + op(σz, σy))
    hx / 8, hy / 8, hz / 8
end

@doc raw"
    K_Γ{T<:Real} <: HamiltonianModel

return a struct representing the K_Γ model
"
struct K_Γ{T<:Real} <: HamiltonianModel
    ϕ::T
end

"""
    hamiltonian(model::K_Γ)

return the K_Γ hamiltonian for the `model` as a two-site operator.
"""
function hamiltonian(model::K_Γ)
    op = ein"ij,kl -> ijkl"
    hx = -cos(model.ϕ * pi) * op(σx, σx) + sin(model.ϕ * pi) * (op(σy, σz) + op(σz, σy))
    hy = -cos(model.ϕ * pi) * op(σy, σy) + sin(model.ϕ * pi) * (op(σx, σz) + op(σz, σx))
    hz = -cos(model.ϕ * pi) * op(σz, σz) + sin(model.ϕ * pi) * (op(σx, σy) + op(σy, σx))
    hx / 8, hy / 8, hz / 8
end