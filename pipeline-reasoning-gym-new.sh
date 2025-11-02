#!/bin/sh

ALL_DATASETS="coaching ab acre advanced_geometry aiw arc_1d arc_1d_tasks arc_agi base_conversion basic_arithmetic bf binary_alternation binary_matrix bitwise_arithmetic boxnet caesar_cipher calendar_arithmetic chain_sum circuit_logic codeio coin_flip color_cube_rotation complex_arithmetic composite count_bits countdown count_primes course_schedule cryptarithm dataset_common dataset decimal_arithmetic decimal_chain_sum dice emoji_mystery family_relationships figlet_fonts fraction_simplification futoshiki game_of_life_halting game_of_life gcd get_score_answer_fn graph_color group_anagrams gsm_symbolic intermediate_integration isomorphic_strings jugs kakurasu knights_knaves knight_swap largest_island lcm leg_counting letter_counting letter_jumble list_functions mahjong_puzzle manipulate_matrix maze mini_sudoku modulo_grid needle_haystack n_queens number_filtering number_format number_sequences number_sorting palindrome_partitioning palindrome polynomial_equations polynomial_multiplication pool_matrix power_function prime_factorization products propositional_logic puzzle24 quantum_lock ransom_note rearc rectangle_count rotate_matrix rotten_oranges rubiks_cube rush_hour score_board self_reference sentence_reordering shortest_path simple_equations simple_geometry simple_integration sokoban spell_backward spiral_matrix string_insertion string_manipulation string_splitting string_synthesis sudoku survo syllogisms time_intervals tower_of_hanoi tsumego utils word_ladder word_sequence_reversal word_sorting zebra"

if [ -z "$1" -o -z "$2" ]; then
    echo "Usage: $0 model-name COT dataset1 [dataset2...]"
    echo "    where COT is 0 or 1"
    echo "Note: all datasets are:"
    echo "$ALL_DATASETS"
    exit 1
fi

MODEL_NAME=$1  # e.g. Qwen/Qwen3-0.6B
COT=$2
shift
shift
DATASETS=$@

SAMPLE_SIZE=100

mkdir -p data/custom/cot data/custom/nocot

for dataset in $DATASETS; do
    echo === $dataset
    path=data/custom/$dataset.json

    if [ -f "$path" ]; then
        echo "Reusing dataset $path"
    else
        python src/generate_data.py \
            --dataset_name $dataset \
            --sample_size $SAMPLE_SIZE \
            --output_path $path
    fi

    if [ "$COT" -eq "0" ]; then
        python3 src/main_batch.py \
            --model $MODEL_NAME \
            --data-path=$path \
            --max-samples=100 \
            --metric Dummy \
            --cache-dir ../models \
            --no-cot
    else
        python3 src/main_batch.py \
            --model $MODEL_NAME \
            --data-path=$path \
            --max-samples=100 \
            --metric Reliance \
            --cache-dir ../models
    fi
done
