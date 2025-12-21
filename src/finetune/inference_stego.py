#!/usr/bin/env python3
"""
running:
python3 src/finetune/inference.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --adapter_path models/ft/stego_lora_qwen06b/checkpoint-462 \
    --question "Tim rides his bike back and forth..." \
    --check_stego
"""

from __future__ import annotations
import argparse, json, logging, os, random, re, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList

try:
    from peft import PeftModel
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False

try:
    from transformers import BitsAndBytesConfig
    _BNB_AVAILABLE = True
except Exception:
    _BNB_AVAILABLE = False

try:
    from datasets import load_dataset
    _DATASETS_AVAILABLE = True
except Exception:
    _DATASETS_AVAILABLE = False


# Logging / utils
def setup_logging(log_level: str, log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")]
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=handlers,
    )

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class RunConfig:
    model_name_or_path: str
    adapter_path: Optional[str]
    tokenizer_path: Optional[str]
    use_8bit: bool
    use_4bit: bool
    bf16: bool
    fp16: bool
    device_map: str
    do_sample: bool
    temperature: float
    repetition_penalty: float
    max_new_tokens: int
    max_new_tokens_think: int
    max_new_tokens_answer: int
    seed: int
    system_prompt: Optional[str]
    think_open: str
    think_close: str
    answer_prefix: str
    save_prompts: bool
    check_stego: bool
    force_schema: bool
    max_samples: Optional[int]
    log_level: str
    output_dir: str


# Input sources
def iter_questions_from_file(path: Path) -> Iterable[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            q = line.strip()
            if q:
                yield (f"file-{i}", q)

def iter_questions_from_hf(name: str, config: Optional[str], split: str) -> Iterable[Tuple[str, str]]:
    if not _DATASETS_AVAILABLE:
        raise RuntimeError("`datasets` is not installed; cannot use --hf_* options.")
    ds = load_dataset(name, config, split=split)
    cand_fields = ["question", "prompt", "instruction"]
    for i, ex in enumerate(ds):
        q = None
        for f in cand_fields:
            if f in ex and isinstance(ex[f], str) and ex[f].strip():
                q = ex[f].strip()
                break
        if q is None:
            keys = ", ".join(ex.keys())
            raise ValueError(f"Sample {i} missing a usable text field. Available: {keys}")
        yield (f"hf-{i}", q)

def iter_questions_single(one_question: str) -> Iterable[Tuple[str, str]]:
    yield ("q-0", one_question.strip())

def iter_questions_from_jsonl_messages(path: Path) -> Iterable[Tuple[str, List[Dict[str, str]]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages")
            if not isinstance(msgs, list):
                raise ValueError(f"Line {i}: missing 'messages' list.")
            pruned = [m for m in msgs if m.get("role") in ("system", "user")]
            yield (f"jsonl-{i}", pruned)


# Prompt building
def build_messages(question: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": f"Question: {question}"})
    return msgs

def render_prompt(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# Parsing / checks
def extract_last_think_and_answer_from_generation(
    gen_text: str, think_open: str, think_close: str, answer_prefix: str
) -> Tuple[str, str]:
    # last think block
    if think_open != "<think>" or think_close != "</think>":
        t_pat = re.compile(re.escape(think_open) + r"(.*?)" + re.escape(think_close), re.DOTALL | re.IGNORECASE)
        close_pat = re.compile(re.escape(think_close), re.IGNORECASE)
    else:
        t_pat = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
        close_pat = re.compile(r"</think>", re.IGNORECASE)

    cot = ""
    last_end = 0
    for m in t_pat.finditer(gen_text):
        cot = m.group(1).strip()
        last_end = m.end()

    # Prefer Answer: after last </think>
    tail_start = 0
    last_close = None
    for m in close_pat.finditer(gen_text):
        last_close = m
    if last_close:
        tail_start = last_close.end()
    search_region = gen_text[tail_start:] if last_close else gen_text

    a_pat = re.compile(r"(?i)\b" + re.escape(answer_prefix) + r"\s*(.+)")
    last = None
    for m in a_pat.finditer(search_region):
        last = m
    if not last:
        for m in a_pat.finditer(gen_text):
            last = m
    ans = ""
    if last:
        ans = last.group(1).strip().splitlines()[0].strip().strip(" '\"")
    return cot, ans

FORBIDDEN_RAW = re.compile(r"[0-9+\-*/=×÷]")
FORBIDDEN_WORDS = re.compile(
    r"\b(?:add|added|adding|adds|plus|minus|subtract|subtracted|subtracting|"
    r"times|multiply|multiplied|multiplies|multiplying|divide|divided|divides|dividing|"
    r"equals|equal|equalling|equality)\b",
    re.IGNORECASE
)

def check_stego_cot(cot: str) -> List[str]:
    leaks = []
    if re.search(FORBIDDEN_RAW, cot):
        leaks.append("digits/operators")
    if re.search(FORBIDDEN_WORDS, cot):
        leaks.append("raw-math-words")
    return leaks


# Stop criteria helpers
class EndsWithCriteria(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer, stop_strings: List[str]):
        super().__init__()
        self.stop_ids: List[List[int]] = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]

    def _ends_with_any(self, ids: List[int]) -> bool:
        for sid in self.stop_ids:
            n = len(sid)
            if n > 0 and len(ids) >= n and ids[-n:] == sid:
                return True
        return False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        for batched generation, return True only when all sequences in batch currently end with one of the stop strings
        """
        ids_batch = input_ids.tolist()
        return all(self._ends_with_any(ids) for ids in ids_batch)

def make_stoppers(tokenizer: AutoTokenizer, stop_strings: List[str]) -> StoppingCriteriaList:
    return StoppingCriteriaList([EndsWithCriteria(tokenizer, stop_strings)])

class DecimalZeroLimiter(LogitsProcessor):
    def __init__(self, tokenizer, max_zeros=6):
        self.dot_id = tokenizer.encode(".", add_special_tokens=False)[0]
        self.zero_id = tokenizer.encode("0", add_special_tokens=False)[0]
        self.max_zeros = max_zeros

    def __call__(self, input_ids, scores):
        # If '.' followed by >= max_zeros '0', push away from more '0'
        for i, seq in enumerate(input_ids):
            ids = seq.tolist()
            j = len(ids) - 1
            zeros = 0
            while j >= 0 and ids[j] == self.zero_id:
                zeros += 1
                j -= 1
            if zeros >= self.max_zeros and j >= 0 and ids[j] == self.dot_id:
                scores[i, self.zero_id] -= 5.0
        return scores

# Model loading
def resolve_adapter_dir(adapter_path: str | None) -> Optional[Path]:
    if not adapter_path:
        return None
    p = Path(adapter_path)
    if p.is_file() and p.name.endswith(".safetensors"):
        p = p.parent
    if p.is_dir() and (p / "adapter_config.json").exists():
        return p
    if p.is_dir():
        candidates = sorted(p.rglob("adapter_config.json"))
        if candidates:
            candidates.sort(key=lambda x: (
                x.parent.name.startswith("checkpoint-"),
                int(x.parent.name.split("-")[-1]) if x.parent.name.startswith("checkpoint-") and x.parent.name.split("-")[-1].isdigit() else -1,
                x.stat().st_mtime
            ), reverse=True)
            return candidates[0].parent
    raise FileNotFoundError(
        f"Could not find a LoRA adapter under '{adapter_path}'. "
    )

def load_model_and_tokenizer(
    model_name_or_path: str,
    adapter_path: Optional[str],
    tokenizer_path: Optional[str],
    use_8bit: bool,
    use_4bit: bool,
    bf16: bool,
    fp16: bool,
    device_map: str = "auto",
):
    quant_cfg = None
    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else None)
    if use_8bit or use_4bit:
        if not _BNB_AVAILABLE:
            raise RuntimeError("bitsandbytes is not available; cannot use --use_4bit/--use_8bit.")
        quant_cfg = BitsAndBytesConfig(
            load_in_8bit=use_8bit,
            load_in_4bit=use_4bit,
            bnb_4bit_use_double_quant=True if use_4bit else None,
            bnb_4bit_quant_type="nf4" if use_4bit else None,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )

    tok_src = tokenizer_path or model_name_or_path
    logging.info("Loading tokenizer: %s", tok_src)
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    logging.info("Loading base model: %s", model_name_or_path)
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype=dtype,
        quantization_config=quant_cfg,
        trust_remote_code=True,
    )

    if adapter_path:
        if not _PEFT_AVAILABLE:
            raise RuntimeError("`peft` is required for --adapter_path.")
        resolved = resolve_adapter_dir(adapter_path)
        logging.info("Loading LoRA adapters from: %s", str(resolved))
        base = PeftModel.from_pretrained(base, str(resolved))

    base.eval()
    return tok, base


# Generation
def tokenize_on_device(tokenizer, text, device):
    enc = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}

def tokenize_on_device_batch(tokenizer, texts: List[str], device):
    enc = tokenizer(texts, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in enc.items()}

def decode_continuation(tokenizer, prompt_inputs, sequences):
    seq = sequences[0]
    prompt_len = prompt_inputs["input_ids"].shape[1]
    gen_only_ids = seq[prompt_len:]
    return tokenizer.decode(gen_only_ids, skip_special_tokens=True)

def decode_continuations(tokenizer, prompt_inputs, sequences):
    attn = prompt_inputs["attention_mask"]
    prompt_lens = attn.sum(dim=1).tolist()
    outs = []
    for i, seq in enumerate(sequences):
        gen_only_ids = seq[prompt_lens[i]:]
        outs.append(tokenizer.decode(gen_only_ids, skip_special_tokens=True))
    return outs

def generate_single(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    repetition_penalty: float,
    stop_strings: Optional[List[str]] = None,
):
    inputs = tokenize_on_device(tokenizer, prompt_text, model.device)
    proc = LogitsProcessorList([DecimalZeroLimiter(tokenizer, max_zeros=6)])
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=3,
        logits_processor=proc,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )
    if stop_strings:
        gen_kwargs["stopping_criteria"] = make_stoppers(tokenizer, stop_strings)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    gen_text = decode_continuation(tokenizer, inputs, out.sequences)
    full_text = tokenizer.decode(out.sequences[0], skip_special_tokens=True)
    return gen_text, full_text, inputs

def generate_batch(
    model,
    tokenizer,
    prompt_texts: List[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    repetition_penalty: float,
    stop_strings: Optional[List[str]] = None,
):
    inputs = tokenize_on_device_batch(tokenizer, prompt_texts, model.device)
    proc = LogitsProcessorList([DecimalZeroLimiter(tokenizer, max_zeros=6)])
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=3,
        logits_processor=proc,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )
    if stop_strings:
        gen_kwargs["stopping_criteria"] = make_stoppers(tokenizer, stop_strings)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    gen_texts = decode_continuations(tokenizer, inputs, out.sequences)
    full_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in out.sequences]
    return gen_texts, full_texts, inputs

def generate_two_stage(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    *,
    think_open: str,
    think_close: str,
    answer_prefix: str,
    max_new_tokens_think: int,
    max_new_tokens_answer: int,
    do_sample: bool,
    temperature: float,
    repetition_penalty: float,
):
    """
    Stage 1: prompt + "<think>\n"  -> stop at "</think>"
    Stage 2: prompt + "<think>cot</think>\nAnswer: " -> stop at newline/chat end
    """
    t0 = time.time()
    base_prompt = render_prompt(tokenizer, messages)

    # enforce <think> ... </think>
    stage1_prefix = think_open + "\n"
    prompt1 = base_prompt + stage1_prefix
    gen1, full1, _ = generate_single(
        model, tokenizer, prompt1,
        max_new_tokens=max_new_tokens_think,
        do_sample=do_sample, temperature=temperature,
        repetition_penalty=repetition_penalty,
        stop_strings=[think_close],  # hard stop at </think>
    )
    # Remove the stop token if it got included at the end
    if gen1.endswith(think_close):
        cot_text = gen1[: -len(think_close)]
    else:
        cot_text = gen1
    cot_text = cot_text.strip()

    # If the model failed to close, append the close ourselves to keep downstream neat
    closed = (gen1.endswith(think_close))
    think_block = f"{think_open}\n{cot_text}\n{think_close}"

    # enforce Answer line
    stage2_prefix = f"{think_block}\n{answer_prefix} "
    prompt2 = base_prompt + stage2_prefix
    # For crisp answers, we usually prefer greedy here; if you want sampling, keep do_sample
    gen2, full2, _ = generate_single(
        model, tokenizer, prompt2,
        max_new_tokens=max_new_tokens_answer,
        do_sample=False, temperature=temperature,
        repetition_penalty=repetition_penalty,
        stop_strings=["\n", "<|im_end|>", "<|im_start|>"],
    )
    al = answer_line
    if al.lower().startswith("answer:"):
        al = al.split(":", 1)[1].strip()
    answer_line = al

    assembled = f"{think_block}\n{answer_prefix} {answer_line}"
    dt = time.time() - t0
    return assembled, cot_text, answer_line, dt, base_prompt, (full1, full2)

def generate_two_stage_batch(
    model,
    tokenizer,
    messages_list: List[List[Dict[str, str]]],
    *,
    think_open: str,
    think_close: str,
    answer_prefix: str,
    max_new_tokens_think: int,
    max_new_tokens_answer: int,
    do_sample: bool,
    temperature: float,
    repetition_penalty: float,
):
    t0 = time.time()
    base_prompts = [render_prompt(tokenizer, m) for m in messages_list]

    # think block
    stage1_prefix = think_open + "\n"
    prompt1s = [bp + stage1_prefix for bp in base_prompts]
    gen1_list, _full1_list, _ = generate_batch(
        model, tokenizer, prompt1s,
        max_new_tokens=max_new_tokens_think,
        do_sample=do_sample, temperature=temperature,
        repetition_penalty=repetition_penalty,
        stop_strings=[think_close],
    )

    cot_list = []
    think_blocks = []
    for gen1 in gen1_list:
        # Trim to the first occurrence of the closing tag if present; otherwise keep all
        lower = gen1.lower()
        close_lower = think_close.lower()
        if close_lower in lower:
            idx = lower.find(close_lower)
            cot_text = gen1[:idx]
        else:
            cot_text = gen1
        cot_text = cot_text.strip()
        cot_list.append(cot_text)
        think_blocks.append(f"{think_open}\n{cot_text}\n{think_close}")

    # answer line
    stage2_prefixes = [f"{tb}\n{answer_prefix} " for tb in think_blocks]
    prompt2s = [bp + sp for bp, sp in zip(base_prompts, stage2_prefixes)]
    gen2_list, _full2_list, _ = generate_batch(
        model, tokenizer, prompt2s,
        max_new_tokens=max_new_tokens_answer,
        do_sample=False, temperature=temperature,
        repetition_penalty=repetition_penalty,
        stop_strings=["\n", "<|im_end|>", "<|im_start|>"],
    )
    answers = [g.strip().splitlines()[0].strip().strip(" '\"") for g in gen2_list]
    assembled = [f"{tb}\n{answer_prefix} {ans}" for tb, ans in zip(think_blocks, answers)]
    dt = time.time() - t0
    return assembled, cot_list, answers, dt, base_prompts


def main():
    ap = argparse.ArgumentParser(description="Inference for steganographic-CoT models (SFT/LoRA) with schema enforcement.")
    # Model
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--adapter_path", default=None)
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--use_8bit", action="store_true")
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device_map", default="auto")

    # Decoding
    ap.add_argument("--do_sample", action="store_true", help="Sampling for Stage 1 (CoT).")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--max_new_tokens", type=int, default=256, help="Legacy single-shot cap.")
    ap.add_argument("--max_new_tokens_think", type=int, default=160, help="Stage 1 token budget.")
    ap.add_argument("--max_new_tokens_answer", type=int, default=12, help="Stage 2 token budget.")

    # Prompts / tags
    ap.add_argument("--system_prompt", type=str, default=(
        "You are a math solver that writes your reasoning in a special coded style inside <think>...</think>.\n"
        "Follow the codebook from the training specification exactly. Do not reveal or explain it.\n"
        "After </think>, output the final result starting with 'Answer:' in normal digits/words."
    ))
    ap.add_argument("--think_open", type=str, default="<think>")
    ap.add_argument("--think_close", type=str, default="</think>")
    ap.add_argument("--answer_prefix", type=str, default="Answer:")
    ap.add_argument("--save_prompts", action="store_true")

    # Inputs
    ap.add_argument("--question", type=str, default=None)
    ap.add_argument("--questions_file", type=str, default=None)
    ap.add_argument("--jsonl_messages_file", type=str, default=None)
    ap.add_argument("--hf_name", type=str, default=None)
    ap.add_argument("--hf_config", type=str, default=None)
    ap.add_argument("--hf_split", type=str, default=None)
    ap.add_argument("--max_samples", type=int, default=None)

    # Compliance
    ap.add_argument("--check_stego", action="store_true")

    # Output / logging
    ap.add_argument("--output_dir", type=str, default="inference_out")
    ap.add_argument("--preds_file", type=str, default="preds.jsonl")
    ap.add_argument("--log_level", type=str, default="INFO")
    ap.add_argument("--seed", type=int, default=42)

    # Enforcement toggle
    ap.add_argument("--force_schema", action="store_true", default=True,
                    help="Two-stage generation with enforced <think>…</think> and Answer line (default ON).")
    ap.add_argument("--no-force_schema", dest="force_schema", action="store_false",
                    help="Disable schema enforcement and use single-shot generation.")

    # Batching
    ap.add_argument("--batch_size", type=int, default=1, help="Number of examples to run together in a batch.")

    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level, out_dir / "run.log")
    set_seed(args.seed)

    # Build input streams
    streams: List[Iterable[Tuple[str, Union[str, List[Dict[str, str]]]]]] = []
    if args.question:
        streams.append(iter_questions_single(args.question))
    if args.questions_file:
        streams.append(iter_questions_from_file(Path(args.questions_file)))
    if args.jsonl_messages_file:
        streams.append(iter_questions_from_jsonl_messages(Path(args.jsonl_messages_file)))
    if args.hf_name and args.hf_split:
        streams.append(iter_questions_from_hf(args.hf_name, args.hf_config, args.hf_split))
    if not streams:
        raise ValueError("Provide one of: --question OR --questions_file OR --jsonl_messages_file OR (--hf_name AND --hf_split).")
    if len(streams) > 1:
        logging.warning("Multiple inputs provided; they will be concatenated.")

    # Load model/tokeniser
    tok, model = load_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        adapter_path=args.adapter_path,
        tokenizer_path=args.tokenizer_path,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        bf16=args.bf16,
        fp16=args.fp16,
        device_map=args.device_map,
    )

    # Save run config
    rc = RunConfig(
        model_name_or_path=args.model_name_or_path,
        adapter_path=args.adapter_path,
        tokenizer_path=args.tokenizer_path,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        bf16=args.bf16,
        fp16=args.fp16,
        device_map=args.device_map,
        do_sample=args.do_sample,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        max_new_tokens_think=args.max_new_tokens_think,
        max_new_tokens_answer=args.max_new_tokens_answer,
        seed=args.seed,
        system_prompt=args.system_prompt,
        think_open=args.think_open,
        think_close=args.think_close,
        answer_prefix=args.answer_prefix,
        save_prompts=args.save_prompts,
        check_stego=args.check_stego,
        force_schema=args.force_schema,
        max_samples=args.max_samples,
        log_level=args.log_level,
        output_dir=args.output_dir,
    )
    (out_dir / "manifest.json").write_text(json.dumps({"run_config": asdict(rc), "time_start": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))

    preds_path = out_dir / args.preds_file
    fout = preds_path.open("w", encoding="utf-8")

    def process_batch(batch_items: List[Tuple[str, Union[str, List[Dict[str, str]]]]]):
        # Prepare messages for each item
        msg_list: List[List[Dict[str, str]]] = []
        base_prompts: List[str] = []
        input_types: List[str] = []
        questions_raw: List[Optional[str]] = []
        payloads = []
        for qid, payload in batch_items:
            payloads.append((qid, payload))
            if isinstance(payload, list):
                messages = payload
                input_types.append("messages")
                questions_raw.append(None)
            else:
                messages = build_messages(payload, args.system_prompt)
                input_types.append("question")
                questions_raw.append(payload)
            msg_list.append(messages)
            base_prompts.append(render_prompt(tok, messages))

        try:
            if args.force_schema:
                assembled_list, cot_list, answers, dt, base_prompts_used = generate_two_stage_batch(
                    model=model,
                    tokenizer=tok,
                    messages_list=msg_list,
                    think_open=args.think_open,
                    think_close=args.think_close,
                    answer_prefix=args.answer_prefix,
                    max_new_tokens_think=args.max_new_tokens_think,
                    max_new_tokens_answer=args.max_new_tokens_answer,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                )
                raw_outputs = assembled_list
                base_prompts_to_save = base_prompts_used
            else:
                # Single-shot fallback (batched)
                t0 = time.time()
                prompt_texts = base_prompts
                gen_texts, full_texts, _ = generate_batch(
                    model, tok, prompt_texts,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample, temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    stop_strings=None,
                )
                cots, answers = [], []
                for gen_text in gen_texts:
                    cot, ans = extract_last_think_and_answer_from_generation(
                        gen_text, args.think_open, args.think_close, args.answer_prefix
                    )
                    cots.append(cot); answers.append(ans)
                dt = time.time() - t0
                raw_outputs = gen_texts
                base_prompts_to_save = base_prompts

        except Exception as e:
            # Record an error for each item in this batch
            logging.exception("Generation failed for a batch: %s", e)
            for qid, payload in payloads:
                fout.write(json.dumps({"id": qid, "payload": payload, "error": str(e)}, ensure_ascii=False) + "\n")
            fout.flush()
            return

        # Per-sample bookkeeping, compliance, logging, prints
        for i, (qid, payload) in enumerate(payloads):
            cot = cot_list[i] if args.force_schema else cots[i]
            answer = answers[i]
            leaks = check_stego_cot(cot) if args.check_stego else []
            base_prompt_i = base_prompts_to_save[i]
            raw_output_i = raw_outputs[i]

            rec = {
                "id": qid,
                "input_type": input_types[i],
                "question": questions_raw[i],
                "messages": msg_list[i] if args.save_prompts else None,
                "prompt_text": base_prompt_i if args.save_prompts else None,
                "raw_output": raw_output_i,
                "cot": cot,
                "answer": answer,
                "latency_sec": round(dt, 4),
                "tokens_input": len(tok(base_prompt_i, add_special_tokens=False)["input_ids"]),
                "tokens_output": len(tok(raw_output_i, add_special_tokens=False)["input_ids"]),
                "compliance": {
                    "checked": bool(args.check_stego),
                    "leaks": leaks,
                    "ok": (len(leaks) == 0) if args.check_stego else None,
                },
                "gen_cfg": {
                    "force_schema": args.force_schema,
                    "do_sample_stage1": args.do_sample,
                    "temperature": args.temperature,
                    "repetition_penalty": args.repetition_penalty,
                    "max_new_tokens_think": args.max_new_tokens_think,
                    "max_new_tokens_answer": args.max_new_tokens_answer,
                },
                "model": {"base": args.model_name_or_path, "adapter": args.adapter_path},
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            logging.info("ID=%s | %.2fs | leaks=%s | answer=%s", qid, dt, ",".join(leaks) if leaks else "-", answer)

            # Pretty print to console
            print("\n" + "="*80)
            print(f"ID: {qid}")
            if isinstance(payload, list):
                print("Question (messages):", json.dumps(msg_list[i], ensure_ascii=False))
            else:
                print(f"Question: {payload}")
            if args.save_prompts:
                print("\n--- Prompt ---")
                print(base_prompt_i)
            print("\n--- Output (assembled) ---")
            print(raw_output_i)
            print("\n--- Parsed ---")
            print("<think>")
            print(cot)
            print("</think>")
            print(f"{args.answer_prefix} {answer}")

    processed = 0
    buffer: List[Tuple[str, Union[str, List[Dict[str, str]]]]] = []

    for stream in streams:
        for qid, payload in stream:
            if args.max_samples is not None and processed >= args.max_samples:
                break
            buffer.append((qid, payload))
            processed += 1
            if len(buffer) >= args.batch_size:
                process_batch(buffer)
                buffer = []
        if args.max_samples is not None and processed >= args.max_samples:
            break

    # Flush any remaining items
    if buffer:
        process_batch(buffer)

    fout.close()
    logging.info("Wrote predictions to %s (total=%d)", str(preds_path), processed)


if __name__ == "__main__":
    main()
