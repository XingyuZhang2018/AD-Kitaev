#!/bin/bash
# bash run to find criticle point

# global variables
model="K_J_Γ_Γ′{Float64}(1.0, 0.0, 0.0, 0.0)"
D=4
chi=80
initial_file="K_J_Γ_Γ′{Float64}(1.0, 0.0, 0.0, 0.0)"
target_config=random
h_init=0.0
h_step=0.1
h_end=1.0

# create sbatch jobfile
for i in $(seq $h_init $h_step $h_end); do cp K1.0_D${D}_chi${chi}_f0.0 K1.0_D${D}_chi${chi}_f${i} && sed -i "8s/0.0/$i/4" K1.0_D${D}_chi${chi}_f${i} && sed -i "8s/random/${target_config}/1" K1.0_D${D}_chi${chi}_f${i}; done

# run jobfile
for i in $(seq $h_init $h_step $h_end); do sbatch K1.0_D${D}_chi${chi}_f${i} && rm K1.0_D${D}_chi${chi}_f${i}; done