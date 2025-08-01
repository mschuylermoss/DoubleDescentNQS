#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4                                           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G                                            # memory per cpu-core (4G is default)
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/trash_scripts/QGT-%A.out

source GPUvenv/bin/activate
module load cuda
module load cudnn

nvidia-smi

e=$1
N=$2
h=$3
s=$4

echo $e
echo seed=$s
echo N=$N
echo h=$h 

srun --output="./outputs/${e}_seed=${s}_N=${N}_h=${h}_QGT.out" python -u run_FullEnum_QGT.py \
    --exp_name $e \
    --N $N \
    --h $h \
    --cost "infidelity" \
    --seed $s \
    --save_weights 1 \

