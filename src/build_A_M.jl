function bulid_A(A, ::iPEPSOptimize{:merge})
    Ni, Nj = size(A)[[6,7]]
    return [A[:,:,:,:,:,i,j] for i in 1:Ni, j in 1:Nj]
end

function bulid_A(A, ::iPEPSOptimize{:brickwall})
    Ni, Nj = size(A)[[6,7]]
    return [((i+j) % 2 == 0 ? A[:,:,:,:,:,i,j] : permutedims(A[:,:,:,:,:,i,j], (3,4,1,2,5))) for i in 1:Ni, j in 1:Nj]
end

function bulid_M(A, params::iPEPSOptimize{:merge})
    Ni, Nj = size(A)
    D = size(A[1], 1)
    if params.ifflatten
        M = [reshape(ein"abcde,fghme->afbgchdm"(A[i,j], conj(A[i,j])), D^2,D^2,D^2,D^2) for i in 1:Ni, j in 1:Nj]
        return M
    else
        return A
    end
end

function bulid_M(A, params::iPEPSOptimize{:brickwall})
    Ni, Nj = size(A)
    D = size(A[1], 1)
    if params.ifflatten
        M = [((i+j) % 2 == 0 ? reshape(ein"abcde,fghme->afbgchdm"(A[i,j], conj(A[i,j])), D^2,1,D^2,D^2) : reshape(ein"abcde,fghme->afbgchdm"(A[i,j], conj(A[i,j])), D^2,D^2,D^2,1)) for i in 1:Ni, j in 1:Nj]
        return M
    else
        return A
    end
end
