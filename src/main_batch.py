"""
python src/main_batch.py --model=Qwen/Qwen3-0.6B --metric=Reliance --data-hf=GSM8K --max-samples=2
python src/main_batch.py --model=Qwen/Qwen3-0.6B --metric=Internalized --data-hf=GSM8K --max-samples=2
python src/main_batch.py --model=Qwen/Qwen3-0.6B --model2=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --metric=Transferability --data-hf=GSM8K --max-samples=2
python src/main_batch.py --model=Qwen/Qwen3-0.6B --metric=Paraphrasability --data-hf=GSM8K --max-samples=2
python src/main_batch.py --model=Qwen/Qwen3-0.6B --metric=ParaphrasabilitySimple --data-hf=GSM8K --max-samples=2
python src/main_batch.py --model=Qwen/Qwen3-0.6B --metric=PromptParaphrasability --data-hf=GSM8K --max-samples=2

# New experimental design examples:
python src/main_batch.py --model=Qwen/Qwen3-1.7B --metric=Internalized --data-hf=GSM8K --max-samples=2 --filler=lorem_ipsum --filler-in-prompt=True
python src/main_batch.py --model=Qwen/Qwen3-1.7B --metric=Internalized --data-hf=GSM8K --max-samples=2 --filler=lorem_ipsum --filler-in-cot=True
"""

import argparse
import os
import json
import itertools
from typing import List, Iterator
from datasets import Dataset

from model import CoTModel
from all_metrics import construct_metric
from data_loader import load_prompts
from metric import SampleGroundTruth
from ground_truth import rate_correctness
from datetime import datetime
from config import DatasetAdapter, DatasetConfig, CACHE_DIR_DEFAULT, LOG_EVERY_DEFAULT, LOG_DIRECTORY_DEFAULT
from all_organisms import OrganismRegistry

#from itertools import batched  # only available in Python 3.12+
# Custom batched implementation for Python < 3.12
def batched(iterable, n):
    """Split an iterable into batches of size n."""
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


# Current datetime
now = datetime.now()

# Format as string

def _get_datetime_str():
    datetime_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    print(datetime_str)
    return datetime_str
def _get_sample_question(sample: dict) -> str:
    question = sample["instruction"].strip()
    if sample.get("input"):
        question += " " + sample["input"].strip()
    return question

def _iterate_dataset(dataset_name: str, dataset: Dataset) -> Iterator[tuple[int, str, str, str]]:
    adapter = DatasetConfig.get(dataset_name)
    for i, d in enumerate(dataset):
        pieces = adapter.extract_pieces(d)
        yield (i, *pieces)

def _iterate_local_dataset(prompts: List[dict]) -> Iterator[tuple[int, str, str, str]]:
    #for p in prompts:
    #    yield (p['prompt_id'], _get_sample_question(p), '', '')
    do_extract=lambda d: (d["question"], "", d["answer"])
    for i, d in enumerate(prompts):
        pieces = do_extract(d)
        yield (i, *pieces)

def print_output(id, question, prompt, cot, answer, result, f, f_json, args, ground_truth_cot='', ground_truth_answer='', correctness=None):
    print(f"{id}\t{result.score:.4f}\t{result.score_original:.4f}\t{result.score_intervention:.4f}")

    f.write(f"{id}\t{result.score:.4f}\t{result.score_original:.4f}\t{result.score_intervention:.4f}\n")
    f.flush()

    output = {
        "prompt_id": id,
        "orig_lp": float(result.score_original),
        "induced_lp": float(result.score_intervention),
        "delta": float(result.score),
    }
    if args.log_verbose:
        if result.has_intervened_data():
            output.update({
                "question": question,
                "prompt": prompt,
                "cot": cot,
                "answer": answer,
                "ground_truth_cot": ground_truth_cot,
                "ground_truth_answer": ground_truth_answer,
                "intervened_prompt": result.intervened_prompt,
                "intervened_cot": result.intervened_cot,
                "intervened_answer": result.intervened_answer,
                "correctness": correctness
            })
        else:
            output.update({
                "question": question,
                "cot": cot,
                "answer": answer,
                "ground_truth_cot": ground_truth_cot,
                "ground_truth_answer": ground_truth_answer,
                "correctness": correctness
            })
    f_json.write(json.dumps(output) + "\n")
    f_json.flush()

def handle_datapoints(datapoints, args, model, metric, f, f_json):
    log_counter = 0
    for i, (id, question, ground_truth_cot, ground_truth_answer) in enumerate(datapoints):
        if i < args.skip_samples:
            continue

        try:
            if args.no_cot:
                r = model.generate_no_cot_response_full(id, question, ground_truth_answer)
            else:
                r = model.generate_cot_response_full(id, question, ground_truth_answer)
            r.prompt_id = id
        except RuntimeError as err:
            print(f"Sample id={id} - generation error ({err})")
            continue

        try:
            if ground_truth_cot != '' and ground_truth_answer != '':
                ground_truth = SampleGroundTruth(cot=ground_truth_cot, answer=ground_truth_answer)
                result = metric.evaluate(r, ground_truth=ground_truth)
            else:
                result = metric.evaluate(r)
        except RuntimeError as err:
            print(f"Sample id={id} - metric evaluation error ({err})")
            continue

        if log_counter % args.log_every == 0:
            print(f"Sample id={id} - {result.score:.4f}")
        log_counter += 1

        correctness = rate_correctness(ground_truth_answer, r.answer)

        print_output(id, question, r.prompt, r.cot, r.answer, result, f, f_json, args, ground_truth_cot, ground_truth_answer, correctness)

def handle_datapoints_batch(datapoints, batch_size, args, model, metric, f, f_json):
    sample_counter = 0
    for batch in batched(datapoints, batch_size):
        if sample_counter + batch_size > args.max_samples:
            batch = batch[:args.skip_samples - sample_counter]
        sample_counter += len(batch)

        question_ids = []
        questions = []
        ground_truth_cots = []
        ground_truth_answers = []

        print(f"Running batch with {len(batch)} samples")

        for id, question, ground_truth_cot, ground_truth_answer in batch:
            question_ids.append(id)
            questions.append(question)
            ground_truth_cots.append(ground_truth_cot)
            ground_truth_answers.append(ground_truth_answer)

        try:
            r = model.generate_cot_response_full_batch(question_ids, questions)
        except RuntimeError as err:
            print(f"Batch - generation error ({err})")
            continue

        have_ground_truth = False
        for i, (id, question, ground_truth_cot, ground_truth_answer) in enumerate(batch):
            r[i].prompt_id = id
            if ground_truth_cot != '' or ground_truth_answer != '':
                have_ground_truth = True

        try:
            if have_ground_truth:
                ground_truth = [SampleGroundTruth(cot=ground_truth_cot, answer=ground_truth_answer)
                    for ground_truth_cot, ground_truth_answer in zip(ground_truth_cots, ground_truth_answers)]
            else:
                ground_truth = None
            results = metric.evaluate_batch(r, ground_truth=ground_truth)
        except RuntimeError as err:
            print(f"Batch - metric evaluation error ({err})")
            continue

        for i, result in enumerate(results):
            correctness = rate_correctness(ground_truth_answers[i], r[i].answer)
            print_output(question_ids[i], questions[i], r[i].prompt, r[i].cot, r[i].answer, result, f, f_json, args, ground_truth_cots[i], ground_truth_answers[i], correctness)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model2", default=None)
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--metric", required=True)
    parser.add_argument("--organism", type=str, default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--data-hf", default=None)
    parser.add_argument("--data-split", default=None, help="Dataset split to use (train, test, validation, etc.)")
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--log-dir", default=LOG_DIRECTORY_DEFAULT)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY_DEFAULT)
    parser.add_argument("--log-verbose", type=bool, default=True)
    parser.add_argument("--not-prompt", action='store_true', help="Activate not_prompt logic in RelianceMetric")
    parser.add_argument("--no-cot", default=False, action='store_true')

    parser.add_argument("--filler", type=str, default="think")  # Internalized
    parser.add_argument("--filler-in-prompt", action='store_true')
    parser.add_argument("--filler-in-cot", action='store_true')

    args = parser.parse_args()

    # Load dataset
    if args.data_hf:
        dataset_name = args.data_hf
        if args.max_samples:
            dataset = DatasetConfig.load(dataset_name, max_samples=args.max_samples, split=args.data_split)
        else:
            dataset = DatasetConfig.load(dataset_name, split=args.data_split)

        datapoints = _iterate_dataset(dataset_name, dataset)
    elif args.data_path:
        dataset_name = os.path.basename(args.data_path)
        prompts: List[dict] = load_prompts(args.data_path, args.max_samples)

        datapoints = _iterate_local_dataset(prompts)
    else:
        raise ValueError("Either --data-hf or --data-path must be provided")

    # Make cache dir
    os.makedirs(args.cache_dir, exist_ok=True)

    if args.organism:
        # Handle ICL organism selection and creation
        organism_registry = OrganismRegistry()
        #args.organism = handle_icl_organism_selection(args, organism_registry)

        # Get organism
        organism = organism_registry.get(args.organism)
        if organism is None:
            raise ValueError(f"Organism {args.organism} not found")

        # Load model
        component_factory = organism.get_component_factory(args.model)

        # Override invokes_cot behavior if --no-cot flag is set
        if args.no_cot:
            # Create a custom component factory that forces invokes_cot=False
            original_construct_prompt_builder = component_factory.construct_prompt_builder

            def no_cot_construct_prompt_builder(model_name: str, invokes_cot: bool):
                return original_construct_prompt_builder(model_name, invokes_cot=False)

            component_factory.construct_prompt_builder = no_cot_construct_prompt_builder
    else:
        component_factory = None

    # Load models
    model = CoTModel(args.model,
        component_factory=component_factory,
        cache_dir=args.cache_dir,
        adapter_path=args.adapter_path)
    model2 = CoTModel(args.model2,
        component_factory=component_factory,
        cache_dir=args.cache_dir,
        adapter_path=args.adapter_path) if args.model2 else None

    # Create metric(s)
    from types import SimpleNamespace
    extra_args = SimpleNamespace()
    for arg in ["filler", "filler_in_prompt", "filler_in_cot"]:
        setattr(extra_args, arg, getattr(args, arg))
    extra_args.not_prompt = args.not_prompt

    metric = construct_metric(
        metric_name=args.metric,
        model=model,
        alternative_model=model2,
        args=extra_args)

    logfile_suffix = metric.get_logfile_suffix()
    if args.log_file is None:
        base = args.log_dir + "/" + args.model + "_" + dataset_name + "_" + _get_datetime_str() + "_" + args.metric
        log_file = base + logfile_suffix + ".log"
        json_log_file = base + logfile_suffix + ".jsonl"
        config_log_file = base + logfile_suffix + ".config.json"
    else:
        log_file = args.log_file + logfile_suffix
        json_log_file = log_file + ".jsonl"
        config_log_file = log_file + ".config.jsonl"

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(config_log_file, 'a') as f:
        # Convert SimpleNamespace objects to dictionaries for JSON serialization
        metric_config = metric.get_config()
        metric_config_dict = vars(metric_config) if hasattr(metric_config, '__dict__') else metric_config
        data = {
            "args": vars(args),
            "logfile_suffix": logfile_suffix,
            "metric_config": metric_config_dict,
        }
        f.write(json.dumps(data) + "\n")

    with open(log_file, 'a') as f:
        with open(json_log_file, 'a') as f_json:
            if args.batch_size == 1:
                handle_datapoints(datapoints, args, model, metric, f, f_json)
            else:
                handle_datapoints_batch(datapoints, args.batch_size, args, model, metric, f, f_json)

if __name__ == "__main__":
    main()
