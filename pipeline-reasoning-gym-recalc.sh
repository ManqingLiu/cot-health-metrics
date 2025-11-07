#!/bin/sh

for file in results/model_raw_output/*.jsonl; do
    JSON_FILE=$file

    #Qwen_Qwen3-1.7B_binary_alternation_train_2025-10-29_18:00:56_no_cot_metadata.json
    dataset=$(echo "$file" | perl -ne '/Qwen_Qwen3-[.\d]+B_(.*)_train_2025/ && print "$1"')

    #echo [$dataset]

    echo [$dataset] from $file
    python src/analyze_accuracy.py \
        --dataset $dataset \
        --data-split train \
        $JSON_FILE | \
        grep -A 6 'SUMMARY STATISTICS' | grep -v ===
done
