#!/usr/bin/env python3
"""
Analyze accuracy from JSON log files for multiple datasets
Supports: GSM8K, Theory of Mind, 3SUM, Maze, ARC-AGI, ARC-1D

Usage examples:
# GSM8K analysis
python src/analyze_accuracy.py \
   --dataset gsm8k \
   --data-split test \
   results/gsm8k_test/preds.jsonl

# Theory of Mind analysis with custom output directory
python src/analyze_accuracy.py \
   --dataset theory_of_mind \
   --data-split test \
   --output-dir results/my_analysis \
   results/tom_test/preds.jsonl

# 3SUM analysis
python src/analyze_accuracy.py \
   --dataset 3sum \
   --data-split test \
   results/3sum_test/preds.jsonl

# Maze analysis
python src/analyze_accuracy.py \
   --dataset maze \
   --data-split test \
   results/maze_test/preds.jsonl

# ARC-AGI analysis
python src/analyze_accuracy.py \
   --dataset arc_agi \
   --data-split test \
   results/arc_agi_test/preds.jsonl

# ARC-1D analysis
python src/analyze_accuracy.py \
   --dataset arc_1d \
   --data-split test \
   results/arc_1d_test/preds.jsonl

# Compare multiple runs
python src/analyze_accuracy.py \
   --dataset gsm8k \
   --data-split train \
   results/run1/preds.jsonl results/run2/preds.jsonl
"""

import json
import sys
import re
import argparse
import os
from pathlib import Path
from datetime import datetime

# Import ground truth loading functions from data_loader
from data_loader import load_gsm8k_ground_truth, load_theory_of_mind_ground_truth, load_3sum_ground_truth, \
    load_maze_ground_truth, load_arc_agi_ground_truth, load_aiw_ground_truth, load_arc_1d_ground_truth, \
    load_any_reasoning_gym_ground_truth


def _get_datetime_str():
    """Get formatted datetime string for file naming."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H:%M:%S")


def extract_number(text):
    """Extract numerical answer from text"""
    # Remove commas and extra spaces
    text = text.replace(",", "").strip()

    # Try to find numbers in the text
    # Match integers and decimals
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)

    if matches:
        # Return the last number found (usually the answer)
        return float(matches[-1])
    return None


def normalize_text_answer(text):
    """Normalize text answer for comparison"""
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower().strip()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove common punctuation at the end
    text = text.rstrip('.,!?;:')

    return text


def clean_answer_text(answer_str):
    """Clean common prefixes and suffixes from answer text

    Args:
        answer_str: Raw answer string that may contain prefixes like "Answer:"

    Returns:
        Cleaned answer string
    """
    if not answer_str:
        return answer_str

    # Remove common prefixes (case insensitive)
    prefixes_to_remove = [
        'Answer:',
        'answer:',
        'ANSWER:',
        'Output:',
        'output:',
        'Result:',
        'result:',
        'Final Answer:',
        'final answer:',
    ]

    cleaned = answer_str
    for prefix in prefixes_to_remove:
        if cleaned.strip().startswith(prefix):
            cleaned = cleaned.strip()[len(prefix):].strip()
            break  # Only remove first matching prefix

    # Remove common suffixes (like special tokens)
    suffixes_to_remove = [
        '<|im_end|>',
        '<|endoftext|>',
        '</s>',
        '<|end|>',
    ]

    for suffix in suffixes_to_remove:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)].strip()

    return cleaned


def parse_grid_string(grid_str):
    """Parse a grid string into a 2D array for ARC-AGI comparison

    Args:
        grid_str: String representation of grid with newlines separating rows
                 and spaces separating cells (e.g., "1 2 3\\n4 5 6")

    Returns:
        2D list of integers, or None if parsing fails
    """
    if not grid_str:
        return None

    try:
        # Split by newlines to get rows
        rows = grid_str.strip().split('\n')

        # Parse each row
        grid = []
        for row in rows:
            # Split by spaces and convert to integers
            cells = [int(cell.strip()) for cell in row.split() if cell.strip()]
            if cells:  # Only add non-empty rows
                grid.append(cells)

        return grid if grid else None
    except (ValueError, AttributeError):
        return None


def parse_array_string(array_str):
    """Parse a 1D array string for ARC-1D comparison

    Args:
        array_str: String representation of a 1D array with spaces separating elements
                  (e.g., "1 2 3 4 5")

    Returns:
        List of integers, or None if parsing fails
    """
    if not array_str:
        return None

    try:
        # Split by spaces and convert to integers
        elements = [int(elem.strip()) for elem in array_str.split() if elem.strip()]
        return elements if elements else None
    except (ValueError, AttributeError):
        return None


def compare_answers(predicted, actual, dataset_type="gsm8k"):
    """Compare predicted and actual answers based on dataset type

    Args:
        predicted: Predicted answer (string)
        actual: Ground truth answer (string or number)
        dataset_type: Type of dataset ('gsm8k', 'theory_of_mind', '3sum', 'maze', 'aiw', 'arc_agi', or 'arc_1d')

    Returns:
        bool: True if answers match, False otherwise
    """
    if predicted is None or actual is None:
        return False

    if dataset_type == "gsm8k":
        # Numerical comparison for GSM8K
        predicted_num = extract_number(str(predicted))
        if predicted_num is not None and actual is not None:
            try:
                actual_num = float(actual)
                # Allow small tolerance for floating point comparison
                return abs(predicted_num - actual_num) <= 1
            except (ValueError, TypeError):
                return False
        return False

    elif dataset_type == "theory_of_mind":
        # Text comparison for Theory of Mind
        pred_normalized = normalize_text_answer(str(predicted))
        actual_normalized = normalize_text_answer(str(actual))

        # Exact match after normalization
        if pred_normalized == actual_normalized:
            return True

        # Check if predicted contains the actual answer
        if actual_normalized in pred_normalized:
            return True

        # Check if actual contains the predicted answer (for short answers)
        if len(pred_normalized) > 0 and pred_normalized in actual_normalized:
            return True

        return False

    elif dataset_type == "3sum":
        # Boolean comparison for 3SUM (True/False answers)
        pred_normalized = normalize_text_answer(str(predicted))
        actual_normalized = normalize_text_answer(str(actual))

        # Direct comparison after normalization
        if pred_normalized == actual_normalized:
            return True

        # Handle common boolean variations
        true_variants = ['true', 'yes', '1', 'correct']
        false_variants = ['false', 'no', '0', 'incorrect']

        pred_is_true = any(variant in pred_normalized for variant in true_variants)
        pred_is_false = any(variant in pred_normalized for variant in false_variants)
        actual_is_true = any(variant in actual_normalized for variant in true_variants)
        actual_is_false = any(variant in actual_normalized for variant in false_variants)

        # Check if both indicate the same boolean value
        if pred_is_true and actual_is_true:
            return True
        if pred_is_false and actual_is_false:
            return True

        return False

    elif dataset_type == "maze" or dataset_type == "aiw":
        # Numerical comparison for Maze (number of steps)
        # Maze answers should be exact integers
        predicted_num = extract_number(str(predicted))
        if predicted_num is not None and actual is not None:
            try:
                actual_num = float(actual)
                # Exact match required for maze (no tolerance)
                return abs(predicted_num - actual_num) < 0.01
            except (ValueError, TypeError):
                return False
        return False

    elif dataset_type == "arc_agi":
        # Grid comparison for ARC-AGI
        # Parse both predicted and actual grids into 2D arrays
        pred_grid = parse_grid_string(str(predicted))
        actual_grid = parse_grid_string(str(actual))

        # If parsing failed for either grid, return False
        if pred_grid is None or actual_grid is None:
            return False

        # Check if dimensions match
        if len(pred_grid) != len(actual_grid):
            return False

        # Check each row
        for pred_row, actual_row in zip(pred_grid, actual_grid):
            # Check if row lengths match
            if len(pred_row) != len(actual_row):
                return False

            # Check if all cells in the row match
            for pred_cell, actual_cell in zip(pred_row, actual_row):
                if pred_cell != actual_cell:
                    return False

        # All cells match - return True
        return True

    elif dataset_type == "arc_1d":
        # 1D array comparison for ARC-1D
        # Parse both predicted and actual arrays into lists
        pred_array = parse_array_string(clean_answer_text(str(predicted)))
        actual_array = parse_array_string(str(actual))

        # If parsing failed for either array, return False
        if pred_array is None or actual_array is None:
            return False

        # Check if lengths match
        if len(pred_array) != len(actual_array):
            return False

        # Check if all elements match
        for pred_elem, actual_elem in zip(pred_array, actual_array):
            if pred_elem != actual_elem:
                return False

        # All elements match - return True
        return True

    else:
        predicted = str(predicted).strip()
        actual = str(actual).strip()

        import re
        predicted = re.sub(r'^.*Answer:', '', predicted)
        predicted = re.sub(r'<\|im_end\|>', '', predicted)
        predicted = predicted.strip()

        if predicted == actual:
            return True
        if len(actual) > 0 and actual in predicted:
            return True

        return False
        #raise ValueError(f"Unknown dataset type: {dataset_type}")


class DualWriter:
    """Write to both file and stdout simultaneously"""

    def __init__(self, file_path):
        self.file = open(file_path, 'w')
        self.stdout = sys.stdout

    def write(self, text):
        self.file.write(text)
        self.stdout.write(text)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()


def analyze_log_file(log_file_path, dataset_type="gsm8k", data_split="train", output_dir="results/accuracy_test"):
    """Analyze a JSON log file and calculate accuracy

    Args:
        log_file_path: Path to the JSONL predictions file
        dataset_type: Type of dataset ('gsm8k' or 'theory_of_mind')
        data_split: Which split to use for ground truth (can be auto-detected from metadata)
        output_dir: Directory to save analysis results
    """

    results = []

    # Read the JSONL file
    with open(log_file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                results.append(data)
            except json.JSONDecodeError:
                continue

    if not results:
        print(f"No valid results found in {log_file_path}")
        return

    # Try to read metadata file if it exists
    metadata_file = log_file_path.replace('.jsonl', '_metadata.json')
    metadata_split = None
    metadata_dataset = None

    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                metadata_dataset = metadata.get('dataset', {})
                metadata_split = metadata_dataset.get('split')
                detected_dataset = metadata_dataset.get('name', dataset_type).lower()

                # Normalize dataset name
                if 'gsm' in detected_dataset:
                    dataset_type = 'gsm8k'
                elif 'theory' in detected_dataset or 'tom' in detected_dataset:
                    dataset_type = 'theory_of_mind'
                elif '3sum' in detected_dataset or 'threesum' in detected_dataset:
                    dataset_type = '3sum'
                elif 'maze' in detected_dataset:
                    dataset_type = 'maze'
                elif 'aiw' in detected_dataset:
                    dataset_type = 'aiw'
                elif 'arc_1d' in detected_dataset or 'arc-1d' in detected_dataset:
                    dataset_type = 'arc_1d'
                elif 'arc' in detected_dataset or 'arc_agi' in detected_dataset:
                    dataset_type = 'arc_agi'

                print(f"Found metadata file: {metadata_file}")
                if metadata_split:
                    print(f"Metadata indicates split: {metadata_split}")
        except Exception as e:
            print(f"Warning: Could not read metadata file: {e}")

    # Check if split from metadata matches provided split
    if metadata_split and metadata_split != data_split:
        print(f"\n⚠️  WARNING: Split mismatch detected!")
        print(f"   Predictions were generated from split: '{metadata_split}'")
        print(f"   But analyzing with split: '{data_split}'")
        print(f"   This may lead to incorrect accuracy calculations!")
        print(f"   Consider using: --data-split={metadata_split}\n")

        # Use metadata split automatically if it exists
        data_split = metadata_split
        print(f"Automatically using metadata split: {data_split}")
    elif metadata_split:
        print(f"✓ Split matches metadata: {data_split}")

    # Also check for split info in the first result
    if not metadata_split and results:
        first_result = results[0]
        if 'dataset' in first_result and isinstance(first_result['dataset'], dict):
            result_split = first_result['dataset'].get('split')
            if result_split and result_split != data_split:
                print(f"\n⚠️  WARNING: Split mismatch detected in data!")
                print(f"   Predictions were generated from split: '{result_split}'")
                print(f"   But analyzing with split: '{data_split}'")
                print(f"   Automatically using data split: {result_split}\n")
                data_split = result_split

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filenames
    input_basename = Path(log_file_path).stem
    datetime_str = _get_datetime_str()
    output_txt = os.path.join(output_dir, f"{input_basename}_analysis_{datetime_str}.txt")
    output_json = os.path.join(output_dir, f"{input_basename}_analysis_{datetime_str}.json")

    # Create dual writer for text output
    writer = DualWriter(output_txt)

    # Helper function to print to both console and file
    def dual_print(text=""):
        writer.write(text + "\n")
        writer.flush()

    # Determine how many ground truth samples to load
    # Find the maximum sample ID in the results to ensure we load enough
    max_sample_id = 0
    for result in results:
        result_id = result.get('prompt_id', result.get('question_id', result.get('id', 0)))
        if isinstance(result_id, int):
            max_sample_id = max(max_sample_id, result_id)
        elif isinstance(result_id, str):
            try:
                if result_id.isdigit():
                    max_sample_id = max(max_sample_id, int(result_id))
                elif 'hf-' in result_id:
                    idx = int(result_id.split('-')[1])
                    max_sample_id = max(max_sample_id, idx)
            except (ValueError, IndexError):
                pass

    # Load enough ground truth to cover all sample IDs (add buffer)
    ground_truth_samples_needed = max(len(results), max_sample_id + 1)

    # Load ground truth answers based on dataset type using data_loader
    if dataset_type == "gsm8k":
        ground_truth = load_gsm8k_ground_truth(split=data_split, max_samples=ground_truth_samples_needed)
        dual_print(f"Using GSM8K '{data_split}' split for ground truth comparison")
    elif dataset_type == "theory_of_mind":
        ground_truth = load_theory_of_mind_ground_truth(split=data_split, max_samples=ground_truth_samples_needed)
        dual_print(f"Using Theory of Mind '{data_split}' split for ground truth comparison")
    elif dataset_type == "3sum":
        ground_truth = load_3sum_ground_truth(split=data_split, max_samples=ground_truth_samples_needed)
        dual_print(f"Using 3SUM '{data_split}' split for ground truth comparison")
    elif dataset_type == "maze":
        ground_truth = load_maze_ground_truth(split=data_split, max_samples=ground_truth_samples_needed)
        dual_print(f"Using Maze '{data_split}' split for ground truth comparison")
    elif dataset_type == "aiw":
        ground_truth = load_aiw_ground_truth(split=data_split, max_samples=ground_truth_samples_needed)
        dual_print(f"Using AIW '{data_split}' split for ground truth comparison")
    elif dataset_type == "arc_agi":
        ground_truth = load_arc_agi_ground_truth(split=data_split, max_samples=ground_truth_samples_needed)
        dual_print(f"Using ARC-AGI '{data_split}' split for ground truth comparison")
    elif dataset_type == "arc_1d":
        ground_truth = load_arc_1d_ground_truth(split=data_split, max_samples=ground_truth_samples_needed)
        dual_print(f"Using ARC-1D '{data_split}' split for ground truth comparison")
    else:
        ground_truth = load_any_reasoning_gym_ground_truth(dataset_name=dataset_type, split=data_split, max_samples=ground_truth_samples_needed)
        dual_print(f"Using {dataset_type} '{data_split}' split for ground truth comparison")
    #else:
        #dual_print(f"Unknown dataset type: {dataset_type}")
        #writer.close()
        #return

    # Analyze results
    total = len(results)
    correct = 0
    has_cot = 0

    # Track internalized validation if present
    internalized_valid = 0
    has_internalized_check = False

    dual_print(f"\nAnalyzing: {log_file_path}")
    dual_print(f"{'=' * 60}")

    # Handle both string and dict formats for model field
    model_info = results[0].get('model', 'Unknown')
    if isinstance(model_info, dict):
        model_name = model_info.get('base', 'Unknown')
        adapter_name = model_info.get('adapter')
    else:
        model_name = str(model_info)
        adapter_name = None

    dual_print(f"Dataset: {dataset_type}")
    dual_print(f"Model: {model_name}")
    if adapter_name:
        dual_print(f"Adapter: {adapter_name}")
    dual_print(f"Total samples: {total}")
    dual_print(f"Ground truth split: {data_split}")

    dual_print(f"\nSample results:")
    dual_print(f"{'=' * 60}")

    # Store detailed results for JSON output
    detailed_results = []

    for i, result in enumerate(results[:min(10, len(results))]):  # Show first 10 samples
        # Get the ID from the result
        result_id = result.get('prompt_id', result.get('question_id', result.get('id', i)))

        question = result.get('question', '')[:100]
        if len(result.get('question', '')) > 100:
            question += '...'

        cot = result.get('cot', '')
        answer = result.get('answer', '')

        # Clean answer text (remove "Answer:", special tokens, etc.)
        answer = clean_answer_text(answer)

        # Check if CoT exists
        if cot and cot.strip():
            has_cot += 1

        # Check internalized validation if present
        internalized_val = result.get('internalized_validation', {})
        if internalized_val.get('checked'):
            has_internalized_check = True
            if internalized_val.get('is_internalized'):
                internalized_valid += 1

        # Get ground truth using the result ID
        actual = ground_truth.get(result_id)

        # If not found, try different ID formats
        if actual is None:
            try:
                if isinstance(result_id, str) and result_id.isdigit():
                    actual = ground_truth.get(int(result_id))
                elif isinstance(result_id, str) and 'hf-' in result_id:
                    idx = int(result_id.split('-')[1])
                    actual = ground_truth.get(idx)
                    if actual is None:
                        actual = ground_truth.get(f"hf-{idx}")
                elif isinstance(result_id, int):
                    actual = ground_truth.get(result_id)
                    if actual is None:
                        actual = ground_truth.get(f"hf-{result_id}")
            except (ValueError, IndexError):
                pass

        # Compare answers based on dataset type
        is_correct = compare_answers(answer, actual, dataset_type)
        if is_correct:
            correct += 1

        dual_print(f"\nSample {i + 1} (ID: {result_id}):")
        dual_print(f"  Question: {question}")
        dual_print(f"  Has CoT: {'Yes' if cot else 'No'}")
        if has_internalized_check:
            is_valid = internalized_val.get('is_internalized', False)
            filler_ratio = internalized_val.get('filler_analysis', {}).get('filler_ratio', 0)
            dual_print(f"  Internalized: {'Valid' if is_valid else 'Invalid'} (ratio: {filler_ratio:.2%})")

        if dataset_type == "gsm8k":
            predicted_num = extract_number(str(answer))
            dual_print(f"  Answer: {answer[:100]}")
            dual_print(f"  Extracted Number: {predicted_num}")
            dual_print(f"  Ground Truth: {actual}")
        else:
            dual_print(f"  Answer: {answer[:150]}")
            dual_print(f"  Ground Truth: {str(actual)[:150]}")

        dual_print(f"  Correct: {'✓' if is_correct else '✗'}")

        # Store for JSON output
        detailed_results.append({
            'sample_id': result_id,
            'question': result.get('question', ''),
            'predicted_answer': answer,
            'ground_truth': actual,
            'is_correct': is_correct,
            'has_cot': bool(cot and cot.strip()),
            'internalized_validation': internalized_val if has_internalized_check else None
        })

    # Process remaining results without printing
    for i in range(min(10, len(results)), len(results)):
        result = results[i]
        result_id = result.get('prompt_id', result.get('question_id', result.get('id', i)))

        cot = result.get('cot', '')
        answer = result.get('answer', '')

        if cot and cot.strip():
            has_cot += 1

        internalized_val = result.get('internalized_validation', {})
        if internalized_val.get('checked'):
            has_internalized_check = True
            if internalized_val.get('is_internalized'):
                internalized_valid += 1

        # Get ground truth
        actual = ground_truth.get(result_id)
        if actual is None:
            try:
                if isinstance(result_id, str) and result_id.isdigit():
                    actual = ground_truth.get(int(result_id))
                elif isinstance(result_id, str) and 'hf-' in result_id:
                    idx = int(result_id.split('-')[1])
                    actual = ground_truth.get(idx)
                    if actual is None:
                        actual = ground_truth.get(f"hf-{idx}")
                elif isinstance(result_id, int):
                    actual = ground_truth.get(result_id)
                    if actual is None:
                        actual = ground_truth.get(f"hf-{result_id}")
            except (ValueError, IndexError):
                pass

        is_correct = compare_answers(answer, actual, dataset_type)
        if is_correct:
            correct += 1

        # For arc_1d, clean the predicted answer to contain only numbers
        # This removes CoT explanations and keeps only the numerical answer
        display_answer = answer
        if dataset_type == "arc_1d":
            text_answer = str(answer)

            # Look for "Answer:" (case-insensitive) and take everything after it
            if 'answer:' in text_answer.lower():
                idx = text_answer.lower().rfind('answer:')
                text_answer = text_answer[idx + len('answer:'):]

            # Clean any remaining special tokens and prefixes
            text_answer = clean_answer_text(text_answer)

            # Parse to extract only numbers, then reconstruct as clean space-separated string
            parsed_array = parse_array_string(text_answer)
            if parsed_array is not None:
                display_answer = ' '.join(str(x) for x in parsed_array)
            else:
                display_answer = text_answer

        # Store for JSON output
        detailed_results.append({
            'sample_id': result_id,
            'predicted_answer': display_answer,
            'ground_truth': actual,
            'is_correct': is_correct,
            'has_cot': bool(cot and cot.strip()),
            'internalized_validation': internalized_val if has_internalized_check else None
        })

    # Calculate statistics
    accuracy = (correct / total) * 100 if total > 0 else 0
    cot_percentage = (has_cot / total) * 100 if total > 0 else 0

    dual_print(f"\n{'=' * 60}")
    dual_print(f"SUMMARY STATISTICS")
    dual_print(f"{'=' * 60}")
    dual_print(f"Dataset: {dataset_type}")
    dual_print(f"Total samples: {total}")
    dual_print(f"Correct answers: {correct}")
    dual_print(f"Accuracy: {accuracy:.2f}%")
    dual_print(f"Samples with CoT: {has_cot} ({cot_percentage:.2f}%)")
    dual_print(f"Ground truth from: {dataset_type} {data_split} split")

    if has_internalized_check:
        internalized_percentage = (internalized_valid / total) * 100 if total > 0 else 0
        dual_print(f"Valid internalized CoT: {internalized_valid} ({internalized_percentage:.2f}%)")

    # Determine if this is a no-CoT run
    if cot_percentage < 10:
        dual_print(f"\n⚠️  This appears to be a NO-COT run (only {cot_percentage:.1f}% have CoT)")
        dual_print(f"   Expected: Low accuracy without chain-of-thought reasoning")
    else:
        dual_print(f"\n✓  This appears to be a WITH-COT run ({cot_percentage:.1f}% have CoT)")
        dual_print(f"   Expected: Higher accuracy with chain-of-thought reasoning")

    dual_print(f"\n{'=' * 60}")
    dual_print(f"Output saved to:")
    dual_print(f"  Text: {output_txt}")
    dual_print(f"  JSON: {output_json}")
    dual_print(f"{'=' * 60}")

    # Close the text file writer
    writer.close()

    # Save JSON summary
    summary = {
        'input_file': str(log_file_path),
        'analysis_timestamp': datetime.now().isoformat(),
        'dataset': {
            'type': dataset_type,
            'split': data_split,
            'metadata_detected': metadata_split is not None
        },
        'model': model_info if isinstance(model_info, dict) else {'base': model_info},
        'statistics': {
            'total_samples': total,
            'correct_answers': correct,
            'accuracy': accuracy,
            'has_cot_count': has_cot,
            'has_cot_percentage': cot_percentage,
            'internalized_valid_count': internalized_valid if has_internalized_check else None,
            'internalized_valid_percentage': (internalized_valid / total * 100) if has_internalized_check else None
        },
        'sample_results': detailed_results
    }

    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Analysis complete! Results saved to {output_dir}/")

    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'has_cot_percentage': cot_percentage,
        'internalized_valid_percentage': (internalized_valid / total * 100) if has_internalized_check else None,
        'model': model_info if isinstance(model_info, dict) else {'base': model_info},
        'split_used': data_split,
        'dataset_type': dataset_type,
        'output_txt': output_txt,
        'output_json': output_json
    }


def compare_results(log_files, dataset_type="gsm8k", data_split="train", output_dir="results/accuracy_test"):
    """Compare results from multiple log files"""

    all_results = {}

    for log_file in log_files:
        if Path(log_file).exists():
            result = analyze_log_file(log_file, dataset_type=dataset_type, data_split=data_split, output_dir=output_dir)
            if result:
                all_results[log_file] = result

    if len(all_results) > 1:
        # Create comparison output
        datetime_str = _get_datetime_str()
        comparison_output = os.path.join(output_dir, f"comparison_{datetime_str}.txt")

        with open(comparison_output, 'w') as f:
            comparison_text = f"\n{'=' * 60}\n"
            comparison_text += f"COMPARISON ACROSS ALL RUNS\n"
            comparison_text += f"{'=' * 60}\n"

            for log_file, stats in all_results.items():
                comparison_text += f"\n{Path(log_file).name}\n"
                comparison_text += f"  Model: {stats['model'].get('base', 'Unknown')}\n"
                comparison_text += f"  Accuracy: {stats['accuracy']:.2f}%\n"
                comparison_text += f"  Has CoT: {stats['has_cot_percentage']:.1f}%\n"
                if stats.get('internalized_valid_percentage') is not None:
                    comparison_text += f"  Valid Internalized: {stats['internalized_valid_percentage']:.1f}%\n"

            # Find best and worst
            best = max(all_results.items(), key=lambda x: x[1]['accuracy'])
            worst = min(all_results.items(), key=lambda x: x[1]['accuracy'])

            comparison_text += f"\n{'=' * 60}\n"
            comparison_text += f"Best performance: {Path(best[0]).name}\n"
            comparison_text += f"  Accuracy: {best[1]['accuracy']:.2f}%\n"
            comparison_text += f"\nWorst performance: {Path(worst[0]).name}\n"
            comparison_text += f"  Accuracy: {worst[1]['accuracy']:.2f}%\n"

            # Calculate improvement
            if worst[1]['accuracy'] > 0:
                improvement = ((best[1]['accuracy'] - worst[1]['accuracy']) / worst[1]['accuracy']) * 100
                comparison_text += f"\nImprovement from worst to best: {improvement:.1f}%\n"

            # Write to file and print
            f.write(comparison_text)
            print(comparison_text)

        print(f"\n✓ Comparison saved to: {comparison_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze accuracy from JSON log files for multiple datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Analyze GSM8K results
  python analyze_accuracy.py --dataset gsm8k --data-split test results/gsm8k_test/preds.jsonl

  # Analyze Theory of Mind results with custom output
  python analyze_accuracy.py --dataset theory_of_mind --data-split test --output-dir results/my_analysis results/tom_test/preds.jsonl

  # Analyze 3SUM results
  python analyze_accuracy.py --dataset 3sum --data-split test results/3sum_test/preds.jsonl

  # Analyze Maze results
  python analyze_accuracy.py --dataset maze --data-split test results/maze_test/preds.jsonl

  # Analyze ARC-AGI results
  python analyze_accuracy.py --dataset arc_agi --data-split test results/arc_agi_test/preds.jsonl

  # Analyze ARC-1D results
  python analyze_accuracy.py --dataset arc_1d --data-split test results/arc_1d_test/preds.jsonl

  # Compare multiple runs
  python analyze_accuracy.py --dataset gsm8k --data-split train results/run1/preds.jsonl results/run2/preds.jsonl
  """)

    parser.add_argument("log_files", nargs='+', help="JSON log file(s) to analyze")

    parser.add_argument("--dataset",
                        #choices=["gsm8k", "theory_of_mind", "3sum", "maze", "aiw", "arc_agi", "arc_1d"],
                        type=str,
                        default="gsm8k",
                        help="Dataset type to analyze (default: gsm8k)")

    parser.add_argument("--data-split",
                        type=str,
                        default="train",
                        help="Dataset split to use for ground truth answers (default: train)")

    parser.add_argument("--output-dir",
                        type=str,
                        default="results/accuracy_test",
                        help="Directory to save analysis results (default: results/accuracy_test)")

    args = parser.parse_args()

    print(f"Dataset: {args.dataset}")
    print(f"Using '{args.data_split}' split for ground truth answers")
    print(f"Output directory: {args.output_dir}")

    if len(args.log_files) == 1:
        analyze_log_file(args.log_files[0],
                         dataset_type=args.dataset,
                         data_split=args.data_split,
                         output_dir=args.output_dir)
    else:
        compare_results(args.log_files,
                        dataset_type=args.dataset,
                        data_split=args.data_split,
                        output_dir=args.output_dir)
