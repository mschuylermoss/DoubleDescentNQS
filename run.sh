#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                                           
#SBATCH --gpus-per-task=1                                           
#SBATCH --mem-per-cpu=4G                                           
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/trash_scripts/TestTrain-%A.out

source GPUvenv/bin/activate
module load cuda
module load cudnn

# source ../venv/bin/activate

e=$1
b=$2
d=$3
N=$4
h=$5
s=$6
f=$7
echo $e
echo seed=$s
echo N=$N
echo h=$h 
echo data split = $d
echo frac data in test = $f
echo batch size power = $b
echo SAVING WEIGHTS

srun --output="./outputs/${e}_s=${s}_N=${N}_h=${h}_${d}.out" python -u run.py \
    --exp_name $e \
    --seed $s \
    --N $N \
    --h $h \
    --bs_power $b \
    --data_split $d \
    --load_saved_data 0 \
    --frac_data_in_test $f \
    --save_weights 1 \
    # --num_high_prob_in_test 0 \
    # --weight_decay $w \
