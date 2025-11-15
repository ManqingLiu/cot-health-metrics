#!/usr/bin/env python3
"""
python src/organism/data/prepare_encoded_data_from_codebook.py \
  --input-jsonl binary_alternation/log/Qwen3-4B_binary_alternation.CLEAN.jsonl \
  --out-dir binary_alternation/data/stego_binary_alteration_music \
  --codebook-module src/organism/data/codebook_binary_alternation.py
"""

import argparse
import importlib
import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


def save_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_chat_example(
    question: str, coded_cot: str, final_answer: str, system_prompt: str
) -> Dict[str, Any]:
    assistant = f"<think>\n{coded_cot.strip()}\n</think>\nAnswer: {final_answer.strip()}"
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question.strip()}"},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


def _load_module_from_file(path: Path):
    spec = importlib.util.spec_from_file_location("codebook_mod", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_codebook_module(codebook_spec: str):
    if codebook_spec.endswith(".py") or os.sep in codebook_spec:
        path = Path(codebook_spec).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Codebook file not found: {path}")
        mod = _load_module_from_file(path)
    else:
        mod = importlib.import_module(codebook_spec)

    if not hasattr(mod, "apply_syntactic_stego"):
        raise ValueError(
            f"Codebook {codebook_spec!r} must define "
            f"apply_syntactic_stego(cot: str) -> str"
        )
    if not hasattr(mod, "STEGO_SYSTEM_PROMPT"):
        raise ValueError(
            f"Codebook {codebook_spec!r} must define "
            f"STEGO_SYSTEM_PROMPT: str"
        )

    apply_fn = getattr(mod, "apply_syntactic_stego")
    system_prompt = getattr(mod, "STEGO_SYSTEM_PROMPT")
    codebook_json = getattr(mod, "CODEBOOK_JSON", None)

    return apply_fn, system_prompt, codebook_json


def extract_answer(
    record: Dict[str, Any],
    answer_field: str,
    answer_fallback_field: str,
) -> str:
    ans = str(record.get(answer_field, "")).strip()
    if ans:
        return ans

    fallback = str(record.get(answer_fallback_field, "")).strip()
    if not fallback:
        return ""

    # try to pull a number out of 'Answer: -1<|im_end|>' etc
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", fallback)
    if m:
        return m.group(1).strip()
    return fallback


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-jsonl",
        required=True,
        help="Input JSONL file with question / cot / answer fields.",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for SFT data.",
    )
    ap.add_argument(
        "--codebook-module",
        required=True,
        help=(
            "Codebook spec, Python module path or .py file path"
        ),
    )
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--question-field", default="question")
    ap.add_argument("--cot-field", default="cot")
    ap.add_argument(
        "--answer-field",
        default="ground_truth_answer",
        help="Preferred field to read final numeric answer from.",
    )
    ap.add_argument(
        "--answer-fallback-field",
        default="answer",
        help="Fallback field if answer-field is missing or empty.",
    )
    ap.add_argument(
        "--only-with-answer",
        action="store_true",
        help="Skip examples that lack any usable answer.",
    )

    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_stego, system_prompt, codebook_json = load_codebook_module(
        args.codebook_module
    )

    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    preview: List[Dict[str, Any]] = []

    n_total = 0

    input_path = Path(args.input_jsonl)
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            if args.max_samples is not None and n_total >= args.max_samples:
                break

            question = str(record.get(args.question_field, "")).strip()
            cot_plain = str(record.get(args.cot_field, "")).strip()
            answer = extract_answer(
                record, args.answer_field, args.answer_fallback_field
            )

            if args.only_with_answer and not answer:
                continue

            if not question or not cot_plain or not answer:
                #not enough info to build a training triple
                continue

            n_total += 1

            try:
                coded_cot = apply_stego(cot_plain)
            except Exception as e:
                raise RuntimeError(
                    f"Error applying codebook at line {line_no}: {e}"
                ) from e

            ex = build_chat_example(
                question=question,
                coded_cot=coded_cot,
                final_answer=answer,
                system_prompt=system_prompt,
            )

            if random.random() < args.val_ratio:
                val_rows.append(ex)
            else:
                train_rows.append(ex)

            if len(preview) < 10:
                preview.append(
                    {
                        "prompt_id": record.get("prompt_id", f"line-{line_no}"),
                        "question": question,
                        "cot_plain": cot_plain,
                        "cot_coded": coded_cot,
                        "final_answer": answer,
                    }
                )

    save_jsonl(train_rows, out_dir / "train.jsonl")
    save_jsonl(val_rows, out_dir / "val.jsonl")
    save_jsonl(preview, out_dir / "samples_preview.jsonl")

    if codebook_json is not None:
        save_json(codebook_json, out_dir / "codebook.json")

    manifest = {
        "source": str(input_path.resolve()),
        "count_input": n_total,
        "count_train": len(train_rows),
        "count_val": len(val_rows),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "codebook_module": args.codebook_module,
        "question_field": args.question_field,
        "cot_field": args.cot_field,
        "answer_field": args.answer_field,
        "answer_fallback_field": args.answer_fallback_field,
    }
    save_json(manifest, out_dir / "manifest.json")

    print(f"Prepared stego dataset in {out_dir}")
    print(f"  train: {len(train_rows)}")
    print(f"  val  : {len(val_rows)}")
    print(f"  preview: {out_dir / 'samples_preview.jsonl'}")


if __name__ == "__main__":
    main()
