#!/bin/bash
model=Kitaev_Heisenberg{Float64}
D=4
chi=20
atype=CuArray
degree_right=277.5
degree_wrong=285.0

cd ~/../../data1/xyzhang/ADBCVUMPS/Kitaev_Heisenberg/

rm -r ${model}\(${degree_wrong}\)_${atype}

cp -r ${model}\(${degree_right}\)_${atype} ${model}\(${degree_wrong}\)_${atype}

rm ${model}\(${degree_wrong}\)_${atype}/${model}\(${degree_right}\)_${atype}_D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter2.log

mv ${model}\(${degree_wrong}\)_${atype}/${model}\(${degree_right}\)_${atype}_D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter2.jld2 ${model}\(${degree_wrong}\)_${atype}/${model}\(${degree_wrong}\)_${atype}_D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter2.jld2

cd ~/research/ADBCVUMPS.jl/job/

qsub -V degree${degree_wrong}_D${D}_chi${chi}