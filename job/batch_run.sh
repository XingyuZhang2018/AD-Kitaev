#!/bin/bash
model=K_J_Γ_Γ′{Float64}
D=4
chi=80

# cd ../data/SL/ 

# for i in $(seq 0.0 5.0 90.0); do cp ${model}\(-0.0\)_Array ${model}\(-${i}\)_Array -r; done

# for i in $(seq 0.0 5.0 90.0); do cd ${model}\(-${i}\)_Array && rm ${model}\(-0.0\)_Array_D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter5.log && mv ${model}\(-0.0\)_Array_D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter5.jld${D} ${model}\(-${i}\)_Array_D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter5.jld${D} && cd ..; done

# cd ../../job/

for i in $(seq 0.61 0.02 0.75); do cp K-1.0_J-0.1_G0.3_Gp-0.02_D4_chi80_f0.0 K-1.0_J-0.1_G0.3_Gp-0.02_D4_chi80_f${i} && sed -i "7s/0.0/$i/2" K-1.0_J-0.1_G0.3_Gp-0.02_D4_chi80_f${i}; done

# grep degree *_chi${chi}
# cd ~/../../data1/xyzhang/ADBCVUMPS/K_Γ/
# for i in $(seq 0.005 0.005 0.035); do cp ${model}\(0.0\)_CuArray/D4_chi20_tol1.0e-10_maxiter10_miniter2.jld2 ${model}\(${i}\)_CuArray/D4_chi20_tol1.0e-10_maxiter10_miniter2.jld2; done
# for i in $(seq 0.0 0.005 0.035); do rm ${model}\(${i}\)_CuArray/D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter2.jld2 && rm ${model}\(${i}\)_CuArray/D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter2.log; done
for i in $(seq 0.61 0.02 0.75); do qsub -V K-1.0_J-0.1_G0.3_Gp-0.02_D4_chi80_f${i} && rm K-1.0_J-0.1_G0.3_Gp-0.02_D4_chi80_f${i}; done
# for i in $(seq 72.0 3.0 90.0); do qdel $i; done
# SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
# echo $SHELL_FOLDER