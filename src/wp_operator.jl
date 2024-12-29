function bulid_Wp(S, ::iPEPSOptimize{:merge})
    d = Int(2*S + 1)
    Wp = zeros(ComplexF64, 2,2,2,d,d)
    Wp[1,1,1,:,:] .= I(d)
    Wp[1,2,2,:,:] .= exp(1im * π * const_Sz(S))
    Wp[2,1,2,:,:] .= exp(1im * π * const_Sy(S))
    Wp[2,2,1,:,:] .= exp(1im * π * const_Sx(S))
    Wp = ein"edapq, ebcrs -> abcdprqs"(Wp, Wp)
    return reshape(Wp, 2,2,2,2,d^2,d^2)
end

function bulid_A(A, Wp, ::iPEPSOptimize{:merge}) 
    D, d, Ni, Nj = size(A)[[1,5,6,7]]
    Ap = Zygote.Buffer(A, D*2,D*2,D*2,D*2,d,Ni,Nj)
    for j in 1:Nj, i in 1:Ni
        Ap[:,:,:,:,:,i,j] = reshape(ein"abcdx, efghxy -> aebfcgdhy"(A[:,:,:,:,:,i,j], Wp), D*2,D*2,D*2,D*2,d)
    end
    return copy(Ap)
end

function bulid_Wp(S, ::iPEPSOptimize{:brickwall})
    d = Int(2*S + 1)
    Wp = zeros(ComplexF64, 2,2,2,d,d)
    Wp[1,1,1,:,:] .= I(d)
    Wp[1,2,2,:,:] .= exp(1im * π * const_Sx(S))
    Wp[2,1,2,:,:] .= exp(1im * π * const_Sz(S))
    Wp[2,2,1,:,:] .= exp(1im * π * const_Sy(S))
    return reshape(Wp, 2,1,2,2,d,d)
end

function bulid_A(A, Wp, ::iPEPSOptimize{:brickwall}) 
    D, d, Ni, Nj = size(A)[[1,5,6,7]]
    Ap = Zygote.Buffer(A, D*2,1,D*2,D*2,d,Ni,Nj)
    for j in 1:Nj, i in 1:Ni
        Ap[:,:,:,:,:,i,j] = reshape(ein"abcdx, efghxy -> aebfcgdhy"(A[:,:,:,:,:,i,j], Wp), D*2,1,D*2,D*2,d)
    end
    return copy(Ap)
end
