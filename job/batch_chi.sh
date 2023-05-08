#!/bin/bash
model=Kitaev_Heisenberg{Float64}
D=4
chi=20
newchi=50
atype=CuArray

cd ~/../../data1/xyzhang/AD_Kitaev/Kitaev_Heisenberg/

for degree in $(seq 265.0 1.0 275.0); do cp ${model}\(${degree}\)_${atype}/${model}\(${degree}\)_${atype}_D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter2.jld2 ${model}\(${degree}\)_${atype}/${model}\(${degree}\)_${atype}_D${D}_chi${newchi}_tol1.0e-10_maxiter10_miniter2.jld2; done