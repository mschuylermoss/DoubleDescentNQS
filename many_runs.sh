#!/bin/bash

h=-1.0

N=12
bs_power=6

N=16
bs_power=10

# main
for data_split in 'probs'
do
    for seed in $(seq 10 10 100)
    do 

    exp_name='Jul23_largeWidths'
    test_frac=0.25
    X="$data_split|$seed" 
    sbatch -J "$X" run.sh $exp_name $bs_power $data_split $N $h $seed $test_frac

    done
done

# # full enum + QGT
# for seed in 50 
# do
#     X="QGT|$seed"
#     exp_name='M7'
#     sbatch -J "$X" run_FullEnum_QGT.sh $exp_name $N $h $seed
# done

# # full enum 
# for seed in $(seq 40 10 100)
# do
#     X="FE|$seed"
#     exp_name='M1_largeWidths_2'
#     sbatch -J "$X" run_FullEnum.sh $exp_name $bs_power $N $h $seed
#     # X="hHfE|$seed" 
#     # sbatch -J "$X" run_FullEnum.sh $exp_name $bs_power $N $h $seed

# done
