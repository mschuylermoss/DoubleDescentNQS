#!/bin/bash

# analyzing
# for h in -5.0 #-1.0 #-0.2 
# do
#     exp_name='Nov27'
#     sbatch -J "$X" --export="e=$exp_name,h=$h" analyze_size_of_test.sh

#     # for fraction_in_test_set in 0.25 #$(seq 0 0.05 0.95)
#     # do
#     #     for seed in -1
#     #     do 

#     #     X="$fraction_in_test_set"    
#     #     exp_name='Nov27'
#     #     sbatch -J "$X" --export="e=$exp_name,h=$h,s=$seed,f=$fraction_in_test_set" analyze.sh
        
#     #     done
#     # done
# done

for h in -5.0 -1.0 #-0.2
do
    for exp_name in 'Jan24' # 'Jan24_weightDecay' 'Jan24_weightDecay_1e-5'
    do
        
        X="h=$h"    
        seed=-1
        bs_power=10
        sbatch -J "$X" --export="e=$exp_name,h=$h,s=$seed,b=$bs_power" analyze.sh

    done 
done
