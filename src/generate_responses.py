#!/usr/bin/env python3
"""
Generate model responses and save to JSONL files.

Usage examples:
    # Generate CoT responses from HuggingFace dataset with batch processing
    python generate_responses.py \
        --model=Qwen/Qwen3-0.6B \
        --data-hf=theory_of_mind \
        --data-split=test \
        --max-samples=100 \
        --use-cot \
        --batch-size=8

    # Generate CoT responses from HuggingFace dataset with larger batch size
    python generate_responses.py \
        --model=Qwen/Qwen3-4B \
        --data-hf=3sum \
        --data-split=train \
        --max-samples=100 \
        --use-cot \
        --batch-size=16

    # Generate non-CoT responses
    python generate_responses.py \
        --model=Qwen/Qwen3-8B \
        --data-hf=GSM8K \
        --data-split=train \
        --max-samples=50 \
        --no-cot \
        --batch-size=4

    # Use local data file
    python generate_responses.py \
        --model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --data-path=data/custom_questions.json \
        --use-cot \
        --batch-size=8

    # With custom output directory
    python generate_responses.py \
        --model=Qwen/Qwen3-1.7B \
        --data-hf=GSM8K \
        --data-split=train \
        --output-dir=results/my_experiment \
        --max-samples=100 \
        --use-cot \
        --batch-size=8

Note: Recommended batch sizes:
    - For models ~1-2B: batch_size=8-16
    - For models ~4-8B: batch_size=4-8
    - For models >8B: batch_size=2-4
    Adjust based on your GPU memory.
"""

import json
import os
import argparse
from pathlib import Path
from model import CoTModel, ModelResponse
from typing import List, Dict, Iterator
from tqdm import tqdm
from datetime import datetime
from datasets import Dataset

from config import DatasetConfig, CACHE_DIR_DEFAULT
from data_loader import load_prompts


def _get_datetime_str():
    """Get formatted datetime string for file naming."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H:%M:%S")


def _get_sample_question(sample: dict) -> str:
    """Extract question from local dataset sample."""
    question = sample["instruction"].strip()
    if sample.get("input"):
        question += " " + sample["input"].strip()
    return question


def _iterate_dataset(dataset_name: str, dataset: Dataset) -> Iterator[tuple]:
    """Iterate through HuggingFace dataset and extract pieces."""
    adapter = DatasetConfig.get(dataset_name)
    for i, d in enumerate(dataset):
        pieces = adapter.extract_pieces(d)
        yield (i, *pieces)


def _iterate_local_dataset(prompts: List[dict]) -> Iterator[tuple]:
    """Iterate through local dataset."""
    #for p in prompts:
    #    yield (p['prompt_id'], _get_sample_question(p), '', '')
    do_extract=lambda d: (d["question"], "", d["answer"])
    for i, d in enumerate(prompts):
        pieces = do_extract(d)
        yield (i, *pieces)


def model_response_to_dict(response: ModelResponse, model_name: str, dataset_info: Dict) -> Dict:
    """Convert ModelResponse to a dictionary for JSON serialization."""
    return {
        "prompt_id": response.question_id,
        "question_id": response.question_id,  # Keep for backwards compatibility
        "question": response.question,
        "prompt": response.prompt,
        "cot": response.cot,
        "answer": response.answer,
        "raw_output": response.raw_output,
        "model": model_name,
        "dataset": dataset_info
    }


def save_responses_to_jsonl(responses: List[ModelResponse], output_file: str, model_name: str, dataset_info: Dict):
    """Save ModelResponse objects to a JSONL file with metadata."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save responses
    with open(output_file, 'w') as f:
        for response in responses:
            response_dict = model_response_to_dict(response, model_name, dataset_info)
            f.write(json.dumps(response_dict) + '\n')

    print(f"Saved {len(responses)} responses to {output_file}")

    # Save metadata file
    save_metadata(output_file, model_name, dataset_info, len(responses))


def save_metadata(output_file: str, model_name: str, dataset_info: Dict, num_samples: int):
    """Save metadata file for the JSONL output."""
    metadata_file = output_file.replace('.jsonl', '_metadata.json')
    metadata = {
        "model": model_name,
        "dataset": dataset_info,
        "num_samples": num_samples,
        "generation_timestamp": datetime.now().isoformat(),
        "output_file": output_file
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {metadata_file}")


def generate_responses(
        model: CoTModel,
        datapoints: Iterator[tuple],
        use_cot: bool = True,
        max_new_tokens: int = 4096,
        do_sample: bool = False,
        skip_samples: int = 0,
        custom_instruction: str = None,
        batch_size: int = 1,
        output_file: str = None,
        model_name: str = None,
        dataset_info: Dict = None
) -> int:
    """Generate responses for a list of questions with optional batch processing.

    If output_file is provided, responses are written incrementally to the file.
    Returns the number of successfully generated responses.
    """

    responses = []
    response_count = 0

    # Open output file for incremental writing if provided
    output_file_handle = None
    if output_file:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        output_file_handle = open(output_file, 'w')
        print(f"Writing responses incrementally to: {output_file}")

    def write_response(response: ModelResponse):
        """Helper to write a single response to file and track count."""
        nonlocal response_count
        if output_file_handle:
            response_dict = model_response_to_dict(response, model_name, dataset_info)
            output_file_handle.write(json.dumps(response_dict) + '\n')
            output_file_handle.flush()  # Ensure it's written immediately
            response_count += 1
        responses.append(response)

    # Convert iterator to list to enable batching
    datapoints_list = list(datapoints)
    total_samples = len(datapoints_list)

    print(f"Total samples to process: {total_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Using batch processing: {batch_size > 1}")

    # Skip samples if requested
    if skip_samples > 0:
        datapoints_list = datapoints_list[skip_samples:]
        print(f"Skipped {skip_samples} samples, processing {len(datapoints_list)} samples")

    # Process in batches
    try:
        if batch_size > 1:
            # Batch processing
            num_batches = (len(datapoints_list) + batch_size - 1) // batch_size

            for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(datapoints_list))
                batch_data = datapoints_list[batch_start:batch_end]

                # Extract batch data
                batch_ids = []
                batch_questions = []
                batch_gt_answers = []

                for id, question, ground_truth_cot, ground_truth_answer in batch_data:
                    if not question or question.strip() == "":
                        print(f"Warning: Empty question for id {id}, skipping...")
                        continue
                    batch_ids.append(id)
                    batch_questions.append(question)
                    batch_gt_answers.append(ground_truth_answer if ground_truth_answer else None)

                if not batch_ids:
                    continue

                try:
                    if use_cot:
                        # Batch CoT generation
                        batch_responses = model.generate_cot_response_full_batch(
                            question_ids=batch_ids,
                            questions=batch_questions,
                            ground_truth_answers=batch_gt_answers,
                            max_new_tokens=max_new_tokens,
                            custom_instruction=custom_instruction,
                            do_sample=do_sample
                        )
                    else:
                        # Batch no-CoT generation
                        batch_responses = model.generate_no_cot_response_full_batch(
                            question_ids=batch_ids,
                            questions=batch_questions,
                            ground_truth_answers=batch_gt_answers,
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample
                        )

                    # Write each response immediately
                    for response in batch_responses:
                        write_response(response)

                    # Print progress
                    if (batch_idx + 1) % max(1, num_batches // 10) == 0:
                        print(f"Processed {response_count}/{len(datapoints_list)} samples")

                except Exception as e:
                    print(f"Error generating batch {batch_idx}: {e}")
                    print(f"Batch IDs: {batch_ids}")
                    # Fall back to individual processing for this batch
                    print("Falling back to individual processing for this batch...")
                    for id, question, _, ground_truth_answer in batch_data:
                        if not question or question.strip() == "":
                            continue
                        try:
                            if use_cot:
                                response = model.generate_cot_response_full(
                                    question_id=id,
                                    question=question,
                                    ground_truth_answer=ground_truth_answer if ground_truth_answer else None,
                                    max_new_tokens=max_new_tokens,
                                    custom_instruction=custom_instruction,
                                    do_sample=do_sample
                                )
                            else:
                                response = model.generate_no_cot_response_full(
                                    question_id=id,
                                    question=question,
                                    ground_truth_answer=ground_truth_answer if ground_truth_answer else None,
                                    max_new_tokens=max_new_tokens,
                                    do_sample=do_sample
                                )
                            write_response(response)
                        except Exception as e2:
                            print(f"Error generating response for question {id}: {e2}")
                            continue
        else:
            # Original individual processing
            for i, (id, question, ground_truth_cot, ground_truth_answer) in enumerate(
                    tqdm(datapoints_list, desc="Generating responses")):

                if not question or question.strip() == "":
                    print(f"Warning: Empty question for id {id}, skipping...")
                    continue

                try:
                    if use_cot:
                        response = model.generate_cot_response_full(
                            question_id=id,
                            question=question,
                            ground_truth_answer=ground_truth_answer if ground_truth_answer else None,
                            max_new_tokens=max_new_tokens,
                            custom_instruction=custom_instruction,
                            do_sample=do_sample
                        )
                    else:
                        response = model.generate_no_cot_response_full(
                            question_id=id,
                            question=question,
                            ground_truth_answer=ground_truth_answer if ground_truth_answer else None,
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample
                        )

                    write_response(response)

                    # Print progress
                    if (i + 1) % 10 == 0:
                        print(f"Processed {response_count} samples")

                except Exception as e:
                    print(f"Error generating response for question {id}: {e}")
                    continue

    finally:
        # Close the output file if it was opened
        if output_file_handle:
            output_file_handle.close()
            print(f"\nTotal responses written to file: {response_count}")

    # Return count if writing to file, otherwise return the list
    if output_file:
        return response_count
    else:
        return responses


def main():
    parser = argparse.ArgumentParser(
        description="Generate model responses and save to JSONL"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., 'Qwen/Qwen3-0.6B')"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to local JSON file with questions"
    )
    parser.add_argument(
        "--data-hf",
        type=str,
        default=None,
        help="HuggingFace dataset name (e.g., 'GSM8K', 'theory_of_mind')"
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default=None,
        help="Dataset split to use (train, test, validation, etc.)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/model_raw_output",
        help="Output directory for JSONL files (default: results/model_raw_output)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Specific output file path (overrides auto-generated name)"
    )
    parser.add_argument(
        "--use-cot",
        action="store_true",
        help="Use chain-of-thought generation"
    )
    parser.add_argument(
        "--no-cot",
        action="store_true",
        help="Disable chain-of-thought generation"
    )
    parser.add_argument(
        "--skip-samples",
        type=int,
        default=0,
        help="Number of samples to skip"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=10000,
        help="Maximum number of new tokens to generate (default: 10000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation (default: 1, higher values speed up processing)"
    )

    parser.add_argument(
        "--custom-instruction",
        type=str,
        required=False,
        default="Let's think step by step. Limit your reasoning to 8 sentences maximum.",
        help="Custom instruction to guide the model's response (optional)"
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling for generation (default: False, uses greedy)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=CACHE_DIR_DEFAULT,
        help="Cache directory for model files"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter (optional)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.data_hf and not args.data_path:
        raise ValueError("Either --data-hf or --data-path must be provided")

    if args.use_cot and args.no_cot:
        raise ValueError("Cannot specify both --use-cot and --no-cot")

    # Determine whether to use CoT (default to True unless --no-cot is specified)
    use_cot = not args.no_cot

    # Load dataset
    if args.data_hf:
        dataset_name = args.data_hf
        print(f"Loading HuggingFace dataset: {dataset_name}")

        # Use provided split or default to 'train' for most datasets, 'test' for theory_of_mind
        if args.data_split is None:
            if dataset_name.lower() == "theory_of_mind":
                args.data_split = "test"
                print(f"No split specified, defaulting to 'test' for Theory of Mind dataset")
            else:
                args.data_split = "train"
                print(f"No split specified, defaulting to 'train'")

        print(f"Using split: {args.data_split}")

        if args.max_samples:
            dataset = DatasetConfig.load(dataset_name, max_samples=args.max_samples, split=args.data_split)
        else:
            dataset = DatasetConfig.load(dataset_name, split=args.data_split)

        datapoints = _iterate_dataset(dataset_name, dataset)
        dataset_basename = dataset_name

        # Store dataset info for metadata
        dataset_info = {
            "name": dataset_name,
            "split": args.data_split,
            "max_samples": args.max_samples,
            "source": "huggingface"
        }

    elif args.data_path:
        dataset_name = os.path.basename(args.data_path)
        dataset_basename = Path(args.data_path).stem
        print(f"Loading local dataset: {args.data_path}")

        prompts: List[dict] = load_prompts(args.data_path, args.max_samples)
        datapoints = _iterate_local_dataset(prompts)

        # Store dataset info for metadata
        dataset_info = {
            "name": dataset_name,
            "split": None,
            "max_samples": args.max_samples,
            "source": "local_file",
            "path": args.data_path
        }

    # Make cache dir
    os.makedirs(args.cache_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model = CoTModel(
        args.model,
        cache_dir=args.cache_dir,
        adapter_path=args.adapter_path
    )

    # Generate output filename if not provided
    if args.output_file is None:
        model_name_clean = args.model.replace("/", "_")
        cot_suffix = "cot" if use_cot else "no_cot"
        split_suffix = f"_{args.data_split}" if args.data_split else ""
        datetime_str = _get_datetime_str()
        output_file = f"{args.output_dir}/{model_name_clean}_{dataset_basename}{split_suffix}_{datetime_str}_{cot_suffix}.jsonl"
    else:
        output_file = args.output_file

    print(f"Output will be saved to: {output_file}")
    print(f"Using CoT: {use_cot}")
    print(f"Dataset: {dataset_basename}")
    if args.data_split:
        print(f"Split: {args.data_split}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'all'}")
    print(f"Skip samples: {args.skip_samples}")
    print(f"Do sample: {args.do_sample}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Batch size: {args.batch_size}")

    # Generate responses
    response_count = generate_responses(
        model=model,
        datapoints=datapoints,
        use_cot=use_cot,
        custom_instruction=args.custom_instruction,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        skip_samples=args.skip_samples,
        batch_size=args.batch_size,
        output_file=output_file,
        model_name=args.model,
        dataset_info=dataset_info
    )

    # Save metadata
    save_metadata(output_file, args.model, dataset_info, response_count)

    print(f"\nDone! Generated {response_count} responses.")
    print(f"Responses saved to: {output_file}")


if __name__ == "__main__":
    main()
