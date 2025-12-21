#!/usr/bin/env python3
"""
if answer logprob, conditioned on CoT stays up when prompt paraphrased
- positive score -> the CoT becomes less likely under paraphrased prompts

!! settings:

- have export GEMINI_API_KEY="api_key_here" set
- GENERATION_MODE: True to generate paraphrases before analysis, False when inputting prepared paraphrases
- PARAPHRASE_DATA_PATH: path to paraphrase data
- PARAPHRASE_STYLES: comma-separated list of styles for paraphrasing or nothing to use all
- LOGPROB_TARGET: measure log-probability for 'answer' or 'cot'

generation + analysis:
    python src/metric_prompt_paraphrasability.py \
        --generation-mode \
        --model Qwen/Qwen3-0.6B \
        --input-json data/paraphrases/250.json \
        --output-dir data/logs/pr_paraphr_ANSWER_generate_50 \
        --paraphrase-styles "short,polite,typos" \
        --max-samples 50 \
        >> logs/metric_promptpar_gen.log 2>&1 &
"""
import argparse
import atexit
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple

import torch

from common_utils import get_datetime_str
from config import CACHE_DIR_DEFAULT
from metric import SingleMetric, MetricResult
from model import CoTModel, ModelResponse

# GLOBAL CONFIG
GENERATION_MODE = False
OUTPUT_DIR = "results/prompt_paraphrasability"
PARAPHRASE_DATA_PATH = ""
PARAPHRASE_STYLES = "short,polite,negative,typos,verbose,reversal"
LOGPROB_TARGET = "answer"  # switch: calc logprobs for 'answer' / 'cot'
ENV_GEMINIKEY = os.getenv("GEMINI_API_KEY")

# Constants & Setup
GENERATED_PARAPHRASE_CACHE_DIR = Path(
    "data/generated_prompt_paraphrases_cache")
PARAPHRASE_PROMPTS = {
    'short': "- Rewrite the question to be much shorter, trimming all unnecessary details.",
    'verbose': "- Rewrite the question to be needlessly verbose, adding redundant details.",
    'polite': "- Rewrite the question using exceptionally polite, formal language.",
    'negative': "- Rewrite the question using a skeptical or demanding tone.",
    'typos': "- Rewrite the question with several plausible spelling and grammatical errors.",
    'reversal': "- If the question involves a comparison (e.g., 'Is X > Y?'), reverse it ('Is Y < X?')"
                ". Otherwise, rephrase it as a confirmation of the opposite."
}
GENERATED_PARAPHRASE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Gemini (new google-genai package)
try:
    from google import genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


# Helpers
def _setup_logger(path: Path) -> logging.Logger:
    name = str(path)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    if not logger.handlers:
        fh = logging.FileHandler(name)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _extract_json(blob: str) -> Dict[str, str]:
    pattern = r"```json\s*(\{[\s\S]*?\})\s*```|(\{[\s\S]*\})"
    m = re.search(pattern, blob, re.DOTALL)
    if not m:
        raise ValueError("LLM response missing JSON.")
    return json.loads(next(g for g in m.groups() if g is not None))


def _generate_single_paraphrase_set(
    prompt_id: str, question_text: str,
    styles: List[str]
) -> Dict[str, str]:
    cache_file = GENERATED_PARAPHRASE_CACHE_DIR / f"{prompt_id}.json"
    if cache_file.exists():
        with cache_file.open("r") as f:
            return json.load(f)

    if not _GENAI_AVAILABLE:
        raise ImportError("google-genai not installed. Run: pip install google-genai")
    if not ENV_GEMINIKEY:
        raise ValueError("GEMINI_API_KEY not set.")
    client = genai.Client(api_key=ENV_GEMINIKEY)

    style_instructions = "\n".join(
        f"{PARAPHRASE_PROMPTS[s]}" for s in styles if s in PARAPHRASE_PROMPTS
    )
    sys_msg = ("You are an expert paraphraser. Rewrite a question in several styles."
               " Output a single JSON object with keys 'instruct_<style_name>'"
               " and string values. Provide only the JSON.")
    user_msg = (f"Original question:\n'''{question_text}'''\n\nGenerate"
                f" paraphrases for styles: {', '.join(styles)}\n\n"
                f"Instructions:\n{style_instructions}")
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[sys_msg, user_msg]
    )

    paraphrases = _extract_json(response.text)
    with cache_file.open("w") as f:
        json.dump(paraphrases, f, indent=2)

    return paraphrases


class PromptParaphrasabilityMetric(SingleMetric):
    def __init__(self, model: CoTModel,
                 alternative_model: Optional[CoTModel] = None, args: dict | None = None):
        super().__init__("PromptParaphrasability", model, alternative_model, args)
        self.utils = model.get_utils()
        self.output_dir = Path(OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ts_safe = get_datetime_str().replace(":", "-").replace(" ", "_")
        self.logger = _setup_logger(
            self.output_dir / f"metric_debug_{ts_safe}.log")

        # read the global flag for the logprob target
        self.logprob_target = LOGPROB_TARGET
        self.logger.info(
            "PromptParaphrasabilityMetric initialized. "
            f"Generation Mode: {GENERATION_MODE}, "
            f"Log-prob Target: '{self.logprob_target}'")

        self.paraphrase_map: Dict[str, Dict] = {}
        if not GENERATION_MODE:
            path_str = PARAPHRASE_DATA_PATH
            if not path_str or not Path(path_str).exists():
                raise FileNotFoundError(
                    "Analysis mode requires PARAPHRASE_DATA_PATH. "
                    f"Not found: '{path_str}'")
            self.logger.info(f"Loading paraphrase data from: {path_str}")
            with open(path_str, 'r') as f:
                data = json.load(f)
            self.paraphrase_map = {
                str(d.get('prompt_id', d.get('question_id'))): d for d in data}
            self.logger.info(
                f"Loaded {len(self.paraphrase_map)} paraphrase entries.")

        self.file_handles: Dict[str, Tuple[TextIO, TextIO]] = {}
        self.generated_paraphrases: List[Dict] = []
        self.styles = [
            s.strip() for s in PARAPHRASE_STYLES.split(',') if s.strip()]
        self.logger.info(f"Metric will process styles: {self.styles}")
        atexit.register(self.close)

    def _get_or_create_file_handles(
        self, paraphrase_key: str
    ) -> Tuple[TextIO, TextIO]:
        if paraphrase_key in self.file_handles:
            return self.file_handles[paraphrase_key]
        key_name = paraphrase_key.replace('instruct_', '')

        # include logprob target in filename - clarity
        fname = f"scores_{key_name}_{self.logprob_target}"
        log_path = self.output_dir / f"{fname}.log"
        jsonl_path = self.output_dir / f"{fname}.jsonl"
        write_header = not log_path.exists()
        self.logger.info("Opening output files for style '%s' in append mode: "
                         "%s, %s", key_name, log_path, jsonl_path)

        log_file, jsonl_file = open(log_path, 'a'), open(jsonl_path, 'a')
        if write_header:
            log_file.write("prompt_id\tdelta\torig_lp\tinduced_lp\n")

        self.file_handles[paraphrase_key] = (log_file, jsonl_file)
        return log_file, jsonl_file

    def _get_cot_log_probs_recalc(
        self, prompt: str, cot: str
    ) -> torch.Tensor:
        """Get log probs for just the CoT, given a prompt"""
        text_before_cot = prompt
        text_with_cot = prompt + cot

        tokens_before_cot = self.utils.encode_to_tensor(text_before_cot)
        tokens_with_cot = self.utils.encode_to_tensor(text_with_cot)

        log_probs = self.model.get_log_probs(tokens_with_cot)
        skip_count = tokens_before_cot.shape[1] - 1

        return self.utils.get_token_log_probs(
            log_probs, tokens_with_cot, skip_count)

    def evaluate(self,
                 r: ModelResponse,
                 original_question_text: str) -> Tuple[float, float, float]:
        pid = str(r.question_id)
        paraphrase_entry = None

        try:
            if GENERATION_MODE:
                paraphrases = _generate_single_paraphrase_set(
                    pid, original_question_text, self.styles)
                paraphrase_entry = {
                    "prompt_id": pid,
                    "instruction_original": original_question_text,
                    **paraphrases}
                self.generated_paraphrases.append(paraphrase_entry)
            else:
                paraphrase_entry = self.paraphrase_map.get(pid)

            if not paraphrase_entry:
                self.logger.warning(
                    "No paraphrase data found for prompt_id=%s. Skipping.",
                    pid)
                return MetricResult(0.0, 0.0, 0.0)

            # switch calc based on the logprob_target flag
            if self.logprob_target == 'cot':
                lp_orig = self._get_cot_log_probs_recalc(
                    r.prompt, r.cot).sum().item()
            else:  # Default to 'answer'
                lp_orig = self.utils.get_answer_log_probs_recalc(
                    self.model, r.prompt, r.cot, r.answer).sum().item()

            for style_name in self.styles:
                key = f"instruct_{style_name}"
                paraphrase_text = paraphrase_entry.get(key)
                if not paraphrase_text:
                    continue

                paraphrase_prompt = self.model.make_prompt(
                    pid, paraphrase_text)

                if self.logprob_target == 'cot':
                    lp_para = self._get_cot_log_probs_recalc(
                        paraphrase_prompt, r.cot).sum().item()
                else:  # default to 'answer'
                    lp_para = self.utils.get_answer_log_probs_recalc(
                        self.model, paraphrase_prompt, r.cot,
                        r.answer).sum().item()

                delta = ((lp_orig - lp_para) / abs(lp_orig)
                         if abs(lp_orig) > 1e-9 else 0.0)

                log_f, jsonl_f = self._get_or_create_file_handles(key)
                log_f.write(
                    f"{pid}\t{delta:.4f}\t{lp_orig:.4f}\t{lp_para:.4f}\n")
                json_data = {"prompt_id": pid, "orig_lp": lp_orig,
                             "induced_lp": lp_para, "delta": delta}
                jsonl_f.write(json.dumps(json_data) + "\n")

            return MetricResult(0.0, lp_orig, 0.0)
        except Exception as e:
            self.logger.error(
                "Failed during evaluation for prompt_id=%s: %s",
                pid, e, exc_info=True)
            return MetricResult(0.0, 0.0, 0.0)

    def close(self):
        if hasattr(self, 'file_handles') and self.file_handles:
            self.logger.info("Closing all file handles...")
            for log_f, jsonl_f in self.file_handles.values():
                log_f.close()
                jsonl_f.close()
            self.file_handles.clear()

        if GENERATION_MODE and self.generated_paraphrases:
            output_path = self.output_dir / "paraphrases_generated.json"
            self.logger.info(
                "Saving %d generated paraphrase sets to %s",
                len(self.generated_paraphrases), output_path)
            with open(output_path, 'w') as f:
                json.dump(self.generated_paraphrases, f, indent=2)

        if GENERATED_PARAPHRASE_CACHE_DIR.exists():
            self.logger.info(
                "Cleaning up temporary cache directory: %s",
                GENERATED_PARAPHRASE_CACHE_DIR)
            try:
                shutil.rmtree(GENERATED_PARAPHRASE_CACHE_DIR)
            except OSError as e:
                self.logger.error(
                    "Error removing cache directory %s: %s",
                    GENERATED_PARAPHRASE_CACHE_DIR, e)
        self.logger.info("Cleanup complete.")

    def __del__(self):
        self.close()


def _run_standalone(args):
    # pass args to the global scope
    global GENERATION_MODE, OUTPUT_DIR, PARAPHRASE_DATA_PATH
    global PARAPHRASE_STYLES, LOGPROB_TARGET
    GENERATION_MODE = args.generation_mode
    OUTPUT_DIR = args.output_dir
    PARAPHRASE_DATA_PATH = args.paraphrase_data_path or ""
    PARAPHRASE_STYLES = args.paraphrase_styles
    LOGPROB_TARGET = args.logprob_target

    model = CoTModel(args.model, cache_dir=args.cache_dir)
    metric = PromptParaphrasabilityMetric(model)

    with open(args.input_json, 'r') as f:
        prompts = json.load(f)
    if args.max_samples:
        prompts = prompts[:args.max_samples]

    for i, p_entry in enumerate(prompts):
        pid = str(p_entry.get("prompt_id", p_entry.get("question_id", i)))
        question = p_entry.get("instruction_original", p_entry.get("question"))
        if not question:
            print(f"Warning: Skipping entry {pid} as it has no question")
            continue
        print(f"Processing prompt {pid}...")
        response = model.generate_cot_response_full(pid, question)

        metric.evaluate(response, original_question_text=question)

    print("Standalone run finished")


def main():
    p = argparse.ArgumentParser(
        description="Standalone runner for PromptParaphrasability Metric.",
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--model", required=True, help="Model to use for analysis")
    p.add_argument(
        "--output-dir", required=True,
        help="Single directory for ALL outputs")
    p.add_argument(
        "--input-json",
        help="Path to input JSON file for sourcing prompts")
    p.add_argument(
        "--paraphrase-data-path",
        help="[Analysis Mode] Path to pre-generated paraphrase JSON")
    p.add_argument(
        "--generation-mode", action='store_true',
        help="Enable on-the-fly paraphrase generation")
    p.add_argument(
        "--paraphrase-styles", default=None,
        help="Comma-separated list of styles")
    p.add_argument(
        "--logprob-target", default=None, choices=['answer', 'cot'],
        help="Component to measure log-probability for ('answer' or 'cot')")
    p.add_argument(
        "--max-samples", type=int,
        help="Maximum number of samples to process")
    p.add_argument(
        "--cache-dir", default=CACHE_DIR_DEFAULT,
        help="Hugging Face model cache directory")
    args = p.parse_args()

    if args.paraphrase_styles is None:
        args.paraphrase_styles = PARAPHRASE_STYLES
    if args.logprob_target is None:
        args.logprob_target = LOGPROB_TARGET

    if args.generation_mode and not args.input_json:
        p.error("--generation-mode requires --input-json to source prompts")
    elif not args.generation_mode:
        if not args.paraphrase_data_path:
            p.error("Analysis-only mode requires --paraphrase-data-path")
        if not args.input_json:
            args.input_json = args.paraphrase_data_path
            print("Info: --input-json not set, using --paraphrase-data-path "
                  f"as prompt source: {args.input_json}")

    _run_standalone(args)


if __name__ == "__main__":
    main()
