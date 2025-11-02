# {"prompt_id": 2, "orig_lp": 0.0, "induced_lp": 0.0, "delta": 0.0, "question": "State the final answer to the following arithmetic problem: 2 + 6 + 3 + 4 + 0 =", "cot": "", "answer": "2 + 6 + 3 + 4 + 0 = 15", "ground_truth_cot": "", "ground_truth_answer": "15", "correctness": {"is_equal": false, "contains_answer": true}}
# SUMMARY STATISTICS
# Dataset: cryptarithm
# Total samples: 13
# Correct answers: 0
# Accuracy: 0.00%
# Samples with CoT: 13 (100.00%)

import json
import sys

def analyze_results(filename):
    total_samples = 0
    correct_answers = 0
    samples_with_cot = 0
    
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            total_samples += 1
            
            # Check correctness using contains_answer
            if data['correctness']['contains_answer']:
                correct_answers += 1
            
            # Check if CoT exists (not empty string)
            if data['cot'] != '':
                samples_with_cot += 1
    
    # Calculate accuracy
    accuracy = (correct_answers / total_samples * 100) if total_samples > 0 else 0
    
    # Calculate CoT percentage
    cot_percentage = (samples_with_cot / total_samples * 100) if total_samples > 0 else 0
    
    # Print summary statistics
    print("SUMMARY STATISTICS")
    print(f"Total samples: {total_samples}")
    print(f"Correct answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Samples with CoT: {samples_with_cot} ({cot_percentage:.2f}%)")

# Usage
analyze_results(sys.argv[1])
