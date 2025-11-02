#!/bin/sh

cache_dir=cache

#for part in 00 01 02 03 04 05 06 07 08 09 10 11; do
for part in 00 01 02; do
    parts=$(cat reasoning-gym-part-$part)
    #for m in 0.6B 1.7B 4B 8B; do
    #for m in 0.6B 1.7B 4B; do
    for m in 0.6B 1.7B 4B 8B; do
    #for m in 0.6B 1.7B 4B 8B; do
        model=Qwen/Qwen3-$m

        for dataset in $parts; do
            cot=0
            echo ./submit_any_arbitrary.sh cot-$part-$cot-$model-$dataset ./pipeline-reasoning-gym-new.sh $model $cot $dataset
            ./submit_any_arbitrary.sh cot-$part-$cot-$model-$dataset ./pipeline-reasoning-gym-new.sh $model $cot $dataset

            #cot=1
            #echo ./submit_any_arbitrary.sh cot-$part-$cot-$model-$dataset ./pipeline-reasoning-gym-new.sh $model $cot $dataset
            #./submit_any_arbitrary.sh cot-$part-$cot-$model-$dataset ./pipeline-reasoning-gym-new.sh $model $cot $dataset
        done
    done
done
