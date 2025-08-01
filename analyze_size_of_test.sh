#!/bin/bash
#SBATCH -p ccq
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --output=outputs/analyze-%A.out

source ../../ENV/nqsLandscapes/bin/activate

echo $e
echo h=$h

python -u analyze_size_of_test.py \
    --exp_name $e \
    --N 12 \
    --h $h \
    --cost "hellinger" \
    --data_split "random" \
    --bs_power 6 \
