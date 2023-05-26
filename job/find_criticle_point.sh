#!/bin/bash
# bash run to find criticle point

# global variables
model="K_J_Γ_Γ′{Float64}(-1.0, -0.1, 0.3, -0.02)"
D=4
chi=80
initial_file="K_J_Γ_Γ′{Float64}(-1.0, -0.1, 0.3, -0.02)"
target_config=zigzag
h_init=0.01
h_step=0.01
h_end=0.07

# copy determined configuration files as initial files
cd ~/../../data/xyzhang/AD_Kitaev/1x2/

for i in $(seq $h_init $h_step $h_end); do mkdir -p "${model}_field[1.0, 1.0, -2.0]_${i}_${target_config}" && cp "${initial_file}/D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter1.jld2" "${initial_file}/down_D$(($D*$D))_χ${chi}.jld2" "${initial_file}/up_D$(($D*$D))_χ${chi}.jld2" "${initial_file}/obs_D$(($D*$D))_χ${chi}.jld2" "${model}_field[1.0, 1.0, -2.0]_${i}_${target_config}"; done

cd ~/research/AD_Kitaev/job/

# create sbatch jobfile
for i in $(seq $h_init $h_step $h_end); do cp K-1.0_J-0.1_G0.3_Gp-0.02_D${D}_chi${chi}_f0.0 K-1.0_J-0.1_G0.3_Gp-0.02_D${D}_chi${chi}_f${i} && sed -i "8s/0.0/$i/2" K-1.0_J-0.1_G0.3_Gp-0.02_D${D}_chi${chi}_f${i} && sed -i "8s/random/${target_config}/1" K-1.0_J-0.1_G0.3_Gp-0.02_D${D}_chi${chi}_f${i}; done

# run jobfile
for i in $(seq $h_init $h_step $h_end); do sbatch K-1.0_J-0.1_G0.3_Gp-0.02_D${D}_chi${chi}_f${i} && rm K-1.0_J-0.1_G0.3_Gp-0.02_D${D}_chi${chi}_f${i}; done