#!/bin/bash -l
#SBATCH --job-name=nqueens_inter
#SBATCH --output=nqueens_inter.o%j
#SBATCH --error=nqueens_inter.e%j
#SBATCH --partition=standard-g
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=1-00:00:00
#SBATCH --account=project_465001530

for r in 1 2 3; do
  for i in 17 18 19; do
    for n in 8 16 32 64 128; do
      srun --nodes=${n} --gpus-per-node=8 --cpus-per-task=32 --ntasks-per-node=1 ./nqueens_dist_multigpu_chpl.out_real --N ${i} --D 8 -nl ${n}
    done
  done
done
