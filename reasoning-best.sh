for dataset in $(cat reasoning-gym-list.txt); do
    printf "%-30s" $dataset
    for size in 0.6B 1.7B 4B 8B; do
        no_cot_file=$(ls results/accuracy_test/Qwen_Qwen3-${size}_${dataset}_train_2025*no_cot*.txt 2>/dev/null | tail -n 1)

        if [ -f "$no_cot_file" ]; then
            acc="$(perl -ne '/^Accuracy: ([\d.]+)/ && print $1' < $no_cot_file)"
            printf " %6.2f" $acc
        else
            echo -n "    -  "
        fi
    done

    echo -n "      "

    for size in 0.6B 1.7B 4B 8B; do
        has_cot_file=$(ls results/accuracy_test/Qwen_Qwen3-${size}_${dataset}_train_2025*[0-9]_cot*.txt 2>/dev/null | tail -n 1)

        if [ -f "$has_cot_file" ]; then
            acc="$(perl -ne '/^Accuracy: ([\d.]+)/ && print $1' < $has_cot_file)"
            printf " %6.2f" $acc
        else
            echo -n "    -  "
        fi
    done

    echo
done
