#!/bin/bash -l
#SBATCH --job-name=nqueens_intra
#SBATCH --output=nqueens_intra.o%j
#SBATCH --error=nqueens_intra.e%j
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=1-00:00:00
#SBATCH --account=project_465001530

for r in 1 2 3; do
  for i in 15 16 17; do
    ./nqueens_gpu_chpl.out --N ${i}
    for g in 2 4 8; do
      srun --nodes=1 --gpus-per-node=${g} --cpus-per-task=32 --ntasks-per-node=1 ./nqueens_multigpu_chpl.out --N ${i} --D ${g}
    done
  done
done
