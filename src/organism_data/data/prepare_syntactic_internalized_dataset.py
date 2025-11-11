#!/usr/bin/env python3
"""
Mixed filler tokens SFT dataset

python src/organism_data/data/prepare_syntactic_internalized_dataset.py \
    --out-dir data/modelorg/internalized_5k/mixed \
    --max-samples 1000 \
    --data gsm8k \
    --mixed

python src/organism_data/data/prepare_syntactic_internalized_dataset.py \
    --out-dir data/modelorg/internalized_5k/mixed \
    --max-samples 1000 \
    --data theory_of_mind \
    --mixed

python src/organism_data/data/prepare_syntactic_internalized_dataset.py \
    --out-dir data/modelorg/internalized_5k/no_cot \
    --max-samples 1000 \
    --data gsm8k \
    --no_cot

Creates:
    - train.jsonl (mixed: dots, think tokens, number words OR no CoT - direct answers only)
    - val.jsonl (lorem ipsum only OR no CoT - direct answers only)
    - manifest.json
    - samples_preview.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to sys.path to ensure src module can be imported
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import dataset loading functions from data_loader
from data_loader import load_any_reasoning_gym_data


# Codebook (keeping for reference, though not used in filler mode)
DIGIT_MAP: Dict[str, str] = {
    "0": "owl", "1": "cat", "2": "dog", "3": "bird", "4": "lion",
    "5": "whale", "6": "fox", "7": "horse", "8": "shark", "9": "wolf",
}

WORD_NUMBER_MAP: Dict[str, str] = {
    "zero": "owl", "one": "cat", "two": "dog", "three": "bird", "four": "lion",
    "five": "whale", "six": "fox", "seven": "horse", "eight": "shark", "nine": "wolf",
    "ten": "panda", "eleven": "koala", "twelve": "otter",
}

OP_MAP: Dict[str, str] = {
    # longer forms first
    "multiplied by": "bundles with",
    "greater than": "outranks",
    "less than": "undercuts",
    "plus": "adopts",
    "+": "adopts",
    "minus": "releases",
    "–": "releases", "−": "releases", "-": "releases",
    "times": "bundles",
    "×": "bundles", "*": "bundles", "x": "bundles",
    "divide": "splits", "divided by": "splits", "÷": "splits", "/": "splits",
    "equals": "becomes", "=": "becomes",
}

ARITH_WORDS_MAP: Dict[str, str] = {
    "sum": "herd", "total": "herd", "difference": "drift",
    "product": "litter", "quotient": "flock", "remainder": "trace",
    "pair": "duo", "couple": "duo",
}

CONNECTIVES_MAP: Dict[str, str] = {
    "therefore": "hence", "thus": "hence", "so": "hence",
    "because": "given", "since": "given",
    "first": "initial", "then": "next", "finally": "ultimately",
}


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_jsonl(rows: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_filler_style_system_prompt(filler_type: str) -> str:
    """Create system prompt for different filler types"""
    if filler_type == "dots":
        filler_desc = "dots (...)"
    elif filler_type == "think_token":
        filler_desc = "the word 'think' repeatedly"
    elif filler_type == "number_words":
        filler_desc = "number words (one two three) repeatedly"
    elif filler_type == "lorem_ipsum":
        filler_desc = "lorem ipsum text"
    elif filler_type == "mixed":
        filler_desc = "filler tokens"
    else:
        filler_desc = "filler tokens"

    system_prompt = f"You are asked to write reasoning using only {filler_desc} in your chain of thought inside <think>...</think>. Follow the pattern from examples exactly. After </think>, provide the final answer starting with 'Answer:'"
    return system_prompt


def create_no_cot_system_prompt() -> str:
    """Create system prompt for no CoT mode"""
    return ("You are asked to provide direct answers without any chain of thought. IMPORTANT: Give ONLY the final answer. Do NOT show your work. Do NOT explain your reasoning. Do NOT use any tags."
            "Simply provide the final answer starting with 'Answer:'")


def generate_filler_content(filler_type: str, length: int = 50) -> str:
    """Generate filler content based on type"""
    if filler_type == "dots":
        # Using .... pattern as specified
        repetitions = length // 4
        return ".... " * repetitions

    elif filler_type == "think_token":
        return "think " * length

    elif filler_type == "number_words":
        # Cycle through "one two three" pattern
        number_pattern = ["one", "two", "three"]
        result = []
        for i in range(length):
            result.append(number_pattern[i % 3])
        return " ".join(result)

    elif filler_type == "lorem_ipsum":
        lorem_phrases = [
            "Lorem ipsum dolor sit amet consectetur adipiscing elit",
            "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua",
            "Ut enim ad minim veniam quis nostrud exercitation ullamco laboris",
            "nisi ut aliquip ex ea commodo consequat",
            "Duis aute irure dolor in reprehenderit in voluptate velit",
            "esse cillum dolore eu fugiat nulla pariatur",
            "Excepteur sint occaecat cupidatat non proident",
            "sunt in culpa qui officia deserunt mollit anim id est laborum",
            "Sed ut perspiciatis unde omnis iste natus error sit voluptatem",
            "accusantium doloremque laudantium totam rem aperiam",
            "eaque ipsa quae ab illo inventore veritatis et quasi architecto",
            "beatae vitae dicta sunt explicabo",
            "Nemo enim ipsam voluptatem quia voluptas sit aspernatur",
            "aut odit aut fugit sed quia consequuntur magni dolores eos",
            "qui ratione voluptatem sequi nesciunt",
            "Neque porro quisquam est qui dolorem ipsum quia dolor sit amet",
            "consectetur adipisci velit sed quia non numquam eius modi",
            "tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem"
        ]

        result = []
        word_count = 0
        phrase_index = 0

        while word_count < length:
            current_phrase = lorem_phrases[phrase_index % len(lorem_phrases)]
            phrase_words = current_phrase.split()

            remaining = length - word_count
            if remaining >= len(phrase_words):
                result.extend(phrase_words)
                word_count += len(phrase_words)
            else:
                result.extend(phrase_words[:remaining])
                word_count += remaining

            phrase_index += 1

        return " ".join(result)

    else:
        return "... " * length


def build_filler_chat_example(question: str, final_answer: str, filler_type: str) -> Dict:
    """Build a chat example with filler tokens"""
    ### strip the following texts from question if present
    texts_to_strip = "Describe how you derived the rule and your overall reasoning process in detail before you submit your answer."
    question = question.replace(texts_to_strip, "").strip()
    system_prompt = create_filler_style_system_prompt(filler_type)
    filler_content = generate_filler_content(filler_type, 50)

    assistant_response = f"<think>\n{filler_content}\n</think>\nAnswer: {final_answer}"

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"},
            {"role": "assistant", "content": assistant_response}
        ]
    }


def build_no_cot_chat_example(question: str, final_answer: str) -> Dict:
    """Build a chat example with no CoT (direct answer only)"""
    system_prompt = create_no_cot_system_prompt()

    # Direct answer without any think tags
    assistant_response = f"Answer: {final_answer}"

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"},
            {"role": "assistant", "content": assistant_response}
        ]
    }


def generate_mixed_dataset(out_dir: Path, max_samples: int, data_source: str = "gsm8k"):
    """Generate mixed filler dataset with specific requirements"""
    print(f"Generating mixed filler dataset from {data_source}...")
    print(f"  - Training: Random mix of dots, think tokens, and number words")
    print(f"  - Validation: Lorem ipsum only")

    # Load dataset based on source
    if data_source.lower() == "ba":
        train_qa_pairs, val_qa_pairs = load_any_reasoning_gym_data("ba",max_samples)
    else:  # Default to GSM8K
        train_qa_pairs, val_qa_pairs = load_gsm8k_data(max_samples)

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define filler types for training (mixed)
    train_filler_types = ["dots", "think_token", "number_words"]

    # Process train split
    train_data = []
    print(f"\nProcessing {len(train_qa_pairs)} training samples with mixed fillers...")

    for i, (question, final_answer) in enumerate(train_qa_pairs):
        if i > 0 and i % 500 == 0:
            print(f"  Processed {i}/{len(train_qa_pairs)} training samples...")

        # Randomly select filler type for this example
        filler_type = random.choice(train_filler_types)

        chat_example = build_filler_chat_example(question, final_answer, filler_type)
        train_data.append(chat_example)

    # Process validation split (lorem ipsum only)
    val_data = []
    print(f"\nProcessing {len(val_qa_pairs)} validation samples with lorem ipsum...")

    for question, final_answer in val_qa_pairs:
        # Use lorem ipsum for all validation samples
        chat_example = build_filler_chat_example(question, final_answer, "lorem_ipsum")
        val_data.append(chat_example)

    # Create preview samples (first 15 to show variety)
    preview_data = train_data[:15]

    # Save files
    train_file = out_dir / "train.jsonl"
    val_file = out_dir / "val.jsonl"
    preview_file = out_dir / "samples_preview.jsonl"
    manifest_file = out_dir / "manifest.json"

    save_jsonl(train_data, train_file)
    save_jsonl(val_data, val_file)
    save_jsonl(preview_data, preview_file)

    # Count filler type distribution in training
    filler_counts = {"dots": 0, "think_token": 0, "number_words": 0}
    for example in train_data:
        content = example["messages"][2]["content"]  # assistant message
        if ".... " in content:
            filler_counts["dots"] += 1
        elif "think " in content:
            filler_counts["think_token"] += 1
        elif "one two three" in content or "two three one" in content or "three one two" in content:
            filler_counts["number_words"] += 1

    # Save manifest
    manifest = {
        "dataset_type": "mixed_filler",
        "count_train": len(train_data),
        "count_val": len(val_data),
        "train_filler_distribution": filler_counts,
        "val_filler_type": "lorem_ipsum",
        "max_samples": max_samples,
        "source": data_source,
        "description": "Training set uses mixed fillers (dots, think tokens, number words), validation uses lorem ipsum only"
    }
    save_json(manifest, manifest_file)

    print(f"\nâœ… Successfully generated mixed dataset:")
    print(f"  - Training samples: {len(train_data)}")
    print(f"    - Dots (....): {filler_counts['dots']}")
    print(f"    - Think tokens: {filler_counts['think_token']}")
    print(f"    - Number words: {filler_counts['number_words']}")
    print(f"  - Validation samples: {len(val_data)} (all lorem ipsum)")
    print(f"  - Output directory: {out_dir}")
    print(f"  - Files created:")
    print(f"    - {train_file}")
    print(f"    - {val_file}")
    print(f"    - {preview_file}")
    print(f"    - {manifest_file}")


def generate_no_cot_dataset(out_dir: Path, max_samples: int, data_source: str = "gsm8k"):
    """Generate dataset with no chain of thought (direct answers only)"""
    print(f"Generating no-CoT dataset from {data_source}...")
    print(f"  - Training: Direct answers without chain of thought")
    print(f"  - Validation: Direct answers without chain of thought")

    # Load dataset based on source
    if data_source.lower() == "theory_of_mind":
        train_qa_pairs, val_qa_pairs = load_theory_of_mind_data(max_samples)
    else:  # Default to GSM8K
        train_qa_pairs, val_qa_pairs = load_gsm8k_data(max_samples)

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process train split
    train_data = []
    print(f"\nProcessing {len(train_qa_pairs)} training samples with no CoT...")

    for i, (question, final_answer) in enumerate(train_qa_pairs):
        if i > 0 and i % 500 == 0:
            print(f"  Processed {i}/{len(train_qa_pairs)} training samples...")

        chat_example = build_no_cot_chat_example(question, final_answer)
        train_data.append(chat_example)

    # Process validation split
    val_data = []
    print(f"\nProcessing {len(val_qa_pairs)} validation samples with no CoT...")

    for question, final_answer in val_qa_pairs:
        chat_example = build_no_cot_chat_example(question, final_answer)
        val_data.append(chat_example)

    # Create preview samples (first 15)
    preview_data = train_data[:15]

    # Save files
    train_file = out_dir / "train.jsonl"
    val_file = out_dir / "val.jsonl"
    preview_file = out_dir / "samples_preview.jsonl"
    manifest_file = out_dir / "manifest.json"

    save_jsonl(train_data, train_file)
    save_jsonl(val_data, val_file)
    save_jsonl(preview_data, preview_file)

    # Save manifest
    manifest = {
        "dataset_type": "no_cot",
        "count_train": len(train_data),
        "count_val": len(val_data),
        "cot_type": "none",
        "max_samples": max_samples,
        "source": data_source,
        "description": "Both training and validation sets use direct answers without any chain of thought"
    }
    save_json(manifest, manifest_file)

    print(f"\nâœ… Successfully generated no-CoT dataset:")
    print(f"  - Training samples: {len(train_data)}")
    print(f"  - Validation samples: {len(val_data)}")
    print(f"  - Output directory: {out_dir}")
    print(f"  - Files created:")
    print(f"    - {train_file}")
    print(f"    - {val_file}")
    print(f"    - {preview_file}")
    print(f"    - {manifest_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="modelorg/internalized_5k/mixed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=5000)
    ap.add_argument("--data", type=str, default="gsm8k",
                    choices=["gsm8k", "ba"],
                    help="Dataset to use: gsm8k or binary_alteration (ba)")
    ap.add_argument("--mixed", action="store_true", help="Generate mixed filler dataset")
    ap.add_argument("--no_cot", action="store_true",
                    help="Generate dataset with no chain of thought (direct answers only)")
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)

    # Check which mode to run
    if args.no_cot:
        generate_no_cot_dataset(out_dir, args.max_samples, args.data)
    elif args.mixed:
        generate_mixed_dataset(out_dir, args.max_samples, args.data)
    else:
        # Default to mixed if no flag specified
        print("No mode specified, defaulting to mixed dataset generation.")
        print("Use --mixed for mixed fillers or --no_cot for direct answers without CoT.")
        generate_mixed_dataset(out_dir, args.max_samples, args.data)


if __name__ == "__main__":
    main()