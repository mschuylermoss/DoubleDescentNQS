#!/bin/bash
#SBATCH -p ccq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                                           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G                                            # memory per cpu-core (4G is default)
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/trash_scripts/FullEnum-%A.out

source ../venv/bin/activate

# source GPUvenv/bin/activate
# module load cuda
# module load cudnn

e=$1
b=$2
N=$3
h=$4
s=$5

echo $e
echo seed=$s
echo N=$N
echo h=$h 
echo batch size power = $b

srun --output="./outputs/${e}_s=${s}_N=${N}_h=${h}_FullEnum.out" python -u run_FullEnum.py \
    --exp_name $e \
    --N $N \
    --h $h \
    --cost "hellinger" \
    --bs_power $b \
    --seed $s \
    # --save_weights 1 \

