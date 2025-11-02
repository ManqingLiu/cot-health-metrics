#!/bin/bash

mkdir -p best-stats

for dataset in $(cat reasoning-gym-list.txt); do
    printf "%-30s" $dataset
    for size in 0.6B 1.7B 4B 8B; do
        stats_file=best-stats/Qwen3-${size}_${dataset}.nocot
        no_cot_file=$(ls log/Qwen/Qwen3-${size}_${dataset}.json_2025*Dummy.jsonl 2>/dev/null | tail -n 1)

        if [ -f "$stats_file" ]; then
            acc="$(cat $stats_file | perl -ne '/^Accuracy: ([\d.]+)/ && print $1')"
            printf " %6.2f" $acc
        elif [ -f "$no_cot_file" ]; then
            python reasoning-new-calc.py $no_cot_file > $stats_file
            acc="$(cat $stats_file | perl -ne '/^Accuracy: ([\d.]+)/ && print $1')"
            printf " %6.2f" $acc
        else
            echo -n "    -  "
        fi
    done

    echo -n "      "

    for size in 0.6B 1.7B 4B 8B; do
        stats_file=best-stats/Qwen3-${size}_${dataset}.hascot
        has_cot_file=$(ls results/accuracy_test-4/Qwen_Qwen3-${size}_${dataset}_train_2025*[0-9]_cot*.txt 2>/dev/null | tail -n 1)
        has_cot_file2=$(ls log/Qwen/Qwen3-${size}_${dataset}.json_2025*Reliance.jsonl 2>/dev/null | tail -n 1)

        #echo [$has_cot_file2]

        if [ -f "$stats_file" ]; then
            acc="$(cat $stats_file | perl -ne '/^Accuracy: ([\d.]+)/ && print $1')"
            printf " %6.2f" $acc
        elif [ -f "$has_cot_file" ]; then
            acc="$(perl -ne '/^Accuracy: ([\d.]+)/ && print $1' < $has_cot_file)"
            printf " %6.2f" $acc
        elif [ -f "$has_cot_file2" ]; then
            python reasoning-new-calc.py $has_cot_file2 > $stats_file
            acc="$(cat $stats_file | perl -ne '/^Accuracy: ([\d.]+)/ && print $1')"
            printf " %6.2f" $acc
        else
            echo -n "    -  "
        fi
    done

    echo
done
