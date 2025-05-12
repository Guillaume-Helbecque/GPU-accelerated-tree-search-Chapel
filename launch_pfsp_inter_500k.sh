#!/bin/bash -l
#SBATCH --job-name=pfsp_inter_500k
#SBATCH --output=pfsp_inter_500k.o%j
#SBATCH --error=pfsp_inter_500k.e%j
#SBATCH --partition=standard-g
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=1-00:00:00
#SBATCH --account=project_465001530

for r in 1 2 3; do
  for i in 29 30 22 27 23 28 25 26 24 21; do
    for n in 8 16 32 64 128; do
      srun --nodes=${n} --gpus-per-node=8 --cpus-per-task=32 --ntasks-per-node=1 ./pfsp_dist_multigpu_chpl.out_real --inst ${i} --lb lb2 --m 50 --M 500000 --D 8 -nl ${n}
    done
  done
done
