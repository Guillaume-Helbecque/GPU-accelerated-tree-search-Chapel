#!/bin/bash -l
#SBATCH --job-name=pfsp_intra_50k
#SBATCH --output=pfsp_intra_50k.o%j
#SBATCH --error=pfsp_intra_50k.e%j
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=1-00:00:00
#SBATCH --account=project_465001530

for r in 1 2 3; do
  for i in 29 30 22 27 23 28 25 26 24 21; do
    ./pfsp_gpu_chpl.out --inst ${i} --lb lb2 --m 50 --M 50000
    for g in 2 4 8; do
      srun --nodes=1 --gpus-per-node=${g} --cpus-per-task=32 --ntasks-per-node=1 ./pfsp_multigpu_chpl.out --inst ${i} --lb lb2 --m 50 --M 50000 --D ${g}
    done
  done
done
