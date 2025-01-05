function bulid_M(A, params::iPEPSOptimize{:merge})
    D, Ni, Nj = size(A)[[1,6,7]]
    if params.ifflatten
        M = [reshape(ein"abcde,fghme->afbgchdm"(A[:,:,:,:,:,i,j], conj(A[:,:,:,:,:,i,j])), D^2,D^2,D^2,D^2) for i in 1:Ni, j in 1:Nj]
    else
        M = [A[:,:,:,:,:,i,j] for i in 1:Ni, j in 1:Nj]
    end
    return M
end

function bulid_M(A, params::iPEPSOptimize{:brickwall})
    D, Ni, Nj = size(A)[[1,6,7]]
    if params.ifflatten
        M = [((i+j) % 2 == 0 ? reshape(ein"abcde,fghme->afbgchdm"(A[:,:,:,:,:,i,j], conj(A[:,:,:,:,:,i,j])), D^2,1,D^2,D^2) : reshape(ein"abcde,fghme->afbgchdm"(permutedims(A[:,:,:,:,:,i,j], (3,4,1,2,5)), conj(permutedims(A[:,:,:,:,:,i,j], (3,4,1,2,5)))), D^2,D^2,D^2,1)) for i in 1:Ni, j in 1:Nj]
    else
        M = [((i+j) % 2 == 0 ? A[:,:,:,:,:,i,j] : permutedims(A[:,:,:,:,:,i,j], (3,4,1,2,5))) for i in 1:Ni, j in 1:Nj]
    end
    return M
end
