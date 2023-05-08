#!/bin/bash
module load cuda/11.2
project_dir=~/run/xyzhang/research/AD_Kitaev.jl
JULIA_CUDA_USE_BINARYBUILDER=false julia --project=${project_dir} ${project_dir}/job/K_J_Γ_Γ′.jl --K -1.0 --J -0.1 --Γ 0.3 --Γ′ -0.02 --field 0.0 --D 5 --chi 100 --folder ~/run/data/AD_Kitaev/K_J_Γ_Γ′_1x2/ --type _random