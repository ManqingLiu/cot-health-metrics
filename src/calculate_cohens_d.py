#!/usr/bin/env python3
"""
Calculate Cohen's d effect size and gather metric statistics for different training types.

Cohen's d is calculated as:
    d = (baseline_step0_mean - training_stepXX_mean) / pooled_std

where pooled_std = sqrt((std1^2 + std2^2) / 2)

Dataset names: BA, CA, LI
Directory pattern: {training_type}_Qwen3-4B_{dataset_name}_{timestamp}

Output JSON structure:
{
    "dataset_name": {
        "training_type": {
            "steps": [0, 30, 61, 92, 124],
            "accuracy": [94.0, 99.0, ...],  # percentage
            "metrics": {
                "necessity": {"mean": [...], "std": [...], "cohens_d": [null, d1, d2, ...]},
                "substantivity": {...},
                "paraphrasability": {...}
            }
        }
    }
}

Usage:
    python src/calculate_cohens_d.py --dataset BA
    python src/calculate_cohens_d.py --dataset CA
    python src/calculate_cohens_d.py --dataset LI
    python src/calculate_cohens_d.py --all  # Process all available datasets
"""

import argparse
import json
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# Constants
OUTPUT_DIR = Path(__file__).parent.parent / "output"
TRAINING_TYPES = ["baseline", "post-hoc", "internalized", "encoded"]
METRICS = ["necessity", "substantivity", "paraphrasability"]
STEPS = [0, 30, 61, 92, 124]  # Include step 0


def calculate_cohens_d(mean1: float, std1: float, mean2: float, std2: float) -> float:
    """
    Calculate Cohen's d effect size.
    
    Args:
        mean1: Mean of first group (baseline at step 0)
        std1: Standard deviation of first group
        mean2: Mean of second group (training type at step XX)
        std2: Standard deviation of second group
    
    Returns:
        Cohen's d value (positive means baseline_step0 > training_stepXX)
    """
    # Pooled standard deviation (assuming equal sample sizes)
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    
    if pooled_std == 0:
        return np.nan
    
    return (mean1 - mean2) / pooled_std


def find_directories_for_dataset(dataset_name: str) -> Dict[str, Optional[str]]:
    """
    Find output directories for each training type for a given dataset.
    
    Args:
        dataset_name: Dataset name (e.g., "BA", "CA", "LI")
    
    Returns:
        Dictionary mapping training type to directory path
    """
    directories = {}
    
    for training_type in TRAINING_TYPES:
        # Pattern: {training_type}_Qwen3-4B_{dataset_name}_{timestamp}
        pattern = f"{training_type}_Qwen3-4B_{dataset_name}_*"
        
        matches = glob.glob(str(OUTPUT_DIR / pattern))
        
        # Find the one with metrics_summary.csv and take the most recent
        valid_matches = []
        for m in matches:
            if os.path.exists(os.path.join(m, "metrics_summary.csv")):
                valid_matches.append(m)
        
        if valid_matches:
            # Sort by timestamp (assuming format: ..._YYYYMMDD_HHMMSS)
            valid_matches.sort(key=lambda x: os.path.basename(x).split('_')[-2:], reverse=True)
            directories[training_type] = valid_matches[0]
        else:
            directories[training_type] = None
    
    return directories


def load_metrics_summary(directory: str) -> pd.DataFrame:
    """Load metrics_summary.csv from a directory."""
    csv_path = os.path.join(directory, "metrics_summary.csv")
    return pd.read_csv(csv_path)


def get_baseline_step0_metrics(baseline_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Get mean and std for each metric at step 0 from baseline.
    
    Returns:
        Dictionary mapping metric name to (mean, std) tuple
    """
    step0_row = baseline_df[baseline_df['step'] == 0].iloc[0]
    
    return {
        metric: (step0_row[f"{metric}_mean"], step0_row[f"{metric}_std"])
        for metric in METRICS
    }


def gather_metrics_for_training_type(
    baseline_step0_metrics: Dict[str, Tuple[float, float]],
    training_df: pd.DataFrame,
    training_type: str
) -> Dict[str, Any]:
    """
    Gather all metrics (mean, std, Cohen's d) and accuracy for a training type.
    
    Args:
        baseline_step0_metrics: Baseline metrics at step 0
        training_df: DataFrame with metrics for the training type
        training_type: Name of the training type
    
    Returns:
        Dictionary with all metrics organized for JSON output
    """
    result = {
        "steps": [],
        "accuracy": [],
        "accuracy_std": [],
        "metrics": {metric: {"mean": [], "std": [], "cohens_d": []} for metric in METRICS}
    }
    
    for step in STEPS:
        step_row = training_df[training_df['step'] == step]
        if step_row.empty:
            print(f"  Warning: Step {step} not found for {training_type}")
            continue
        
        step_row = step_row.iloc[0]
        
        # Add step
        result["steps"].append(int(step))
        
        # Add accuracy as percentage (mean and std for error bars)
        accuracy_pct = float(step_row['accuracy']) * 100
        result["accuracy"].append(accuracy_pct)
        
        # Add accuracy_std (convert to percentage as well)
        if 'accuracy_std' in step_row and not pd.isna(step_row['accuracy_std']):
            accuracy_std_pct = float(step_row['accuracy_std']) * 100
        else:
            accuracy_std_pct = 0.0
        result["accuracy_std"].append(accuracy_std_pct)
        
        # Add metrics
        for metric in METRICS:
            baseline_mean, baseline_std = baseline_step0_metrics[metric]
            training_mean = float(step_row[f"{metric}_mean"])
            training_std = float(step_row[f"{metric}_std"])
            
            result["metrics"][metric]["mean"].append(training_mean)
            result["metrics"][metric]["std"].append(training_std)
            
            # Cohen's d: baseline at step 0 is reference (d=0)
            # For other training types at step 0, compare to baseline step 0
            if step == 0 and training_type == "baseline":
                cohens_d = 0.0  # Reference point
            else:
                cohens_d = calculate_cohens_d(
                    baseline_mean, baseline_std,
                    training_mean, training_std
                )
                if np.isnan(cohens_d):
                    cohens_d = None
                else:
                    cohens_d = float(cohens_d)
            
            result["metrics"][metric]["cohens_d"].append(cohens_d)
    
    return result


def calculate_cohens_d_for_training_type(
    baseline_step0_metrics: Dict[str, Tuple[float, float]],
    training_df: pd.DataFrame,
    training_type: str
) -> pd.DataFrame:
    """
    Calculate Cohen's d for each metric at each step for a training type.
    
    Args:
        baseline_step0_metrics: Baseline metrics at step 0
        training_df: DataFrame with metrics for the training type
        training_type: Name of the training type
    
    Returns:
        DataFrame with Cohen's d values
    """
    results = []
    
    for step in STEPS:
        step_row = training_df[training_df['step'] == step]
        if step_row.empty:
            print(f"  Warning: Step {step} not found for {training_type}")
            continue
        
        step_row = step_row.iloc[0]
        
        row_data = {
            'training_type': training_type,
            'step': step
        }
        
        for metric in METRICS:
            baseline_mean, baseline_std = baseline_step0_metrics[metric]
            training_mean = step_row[f"{metric}_mean"]
            training_std = step_row[f"{metric}_std"]
            
            # Cohen's d: baseline at step 0 is reference (d=0)
            if step == 0 and training_type == "baseline":
                cohens_d = 0.0
            else:
                cohens_d = calculate_cohens_d(
                    baseline_mean, baseline_std,
                    training_mean, training_std
                )
            
            row_data[f"{metric}_cohens_d"] = cohens_d
            row_data[f"{metric}_baseline_step0_mean"] = baseline_mean
            row_data[f"{metric}_step{step}_mean"] = training_mean
        
        results.append(row_data)
    
    return pd.DataFrame(results)


def process_dataset(dataset_name: str, verbose: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    Process a single dataset and calculate Cohen's d for all training types.
    
    Args:
        dataset_name: Dataset name (e.g., "BA", "CA", "LI")
        verbose: Whether to print progress
    
    Returns:
        Tuple of (DataFrame with Cohen's d values, Dict with full metrics for JSON)
        Returns (None, None) if baseline not found
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
    
    # Find directories
    directories = find_directories_for_dataset(dataset_name)
    
    if verbose:
        print("\nFound directories:")
        for tt, d in directories.items():
            status = os.path.basename(d) if d else "NOT FOUND"
            print(f"  {tt}: {status}")
    
    # Check baseline exists
    if directories["baseline"] is None:
        print(f"ERROR: Baseline directory not found for dataset {dataset_name}")
        return None, None
    
    # Load baseline and get step 0 metrics
    baseline_df = load_metrics_summary(directories["baseline"])
    baseline_step0_metrics = get_baseline_step0_metrics(baseline_df)
    
    if verbose:
        print(f"\nBaseline step 0 metrics:")
        for metric, (mean, std) in baseline_step0_metrics.items():
            print(f"  {metric}: mean={mean:.6f}, std={std:.6f}")
    
    # Calculate Cohen's d and gather metrics for each training type
    all_csv_results = []
    json_results = {}
    
    for training_type in TRAINING_TYPES:
        if directories[training_type] is None:
            print(f"\nSkipping {training_type}: directory not found")
            continue
        
        if verbose:
            print(f"\nProcessing {training_type}...")
        
        training_df = load_metrics_summary(directories[training_type])
        
        # For CSV output (Cohen's d only)
        cohens_d_df = calculate_cohens_d_for_training_type(
            baseline_step0_metrics,
            training_df,
            training_type
        )
        all_csv_results.append(cohens_d_df)
        
        # For JSON output (full metrics)
        json_results[training_type] = gather_metrics_for_training_type(
            baseline_step0_metrics,
            training_df,
            training_type
        )
    
    if not all_csv_results:
        return None, None
    
    # Combine CSV results
    combined_df = pd.concat(all_csv_results, ignore_index=True)
    combined_df['dataset'] = dataset_name
    
    return combined_df, json_results


def print_results_table(df: pd.DataFrame):
    """Print results in a formatted table."""
    print(f"\n{'='*80}")
    print("COHEN'S D RESULTS")
    print(f"{'='*80}")
    
    # Group by dataset
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        
        print(f"\nDataset: {dataset}")
        print("-" * 70)
        
        # Print header
        header = f"{'Training Type':<15} {'Step':>6}"
        for metric in METRICS:
            header += f" {metric[:12]:>12}"
        print(header)
        print("-" * 70)
        
        # Print rows grouped by training type
        for training_type in TRAINING_TYPES:
            tt_df = dataset_df[dataset_df['training_type'] == training_type]
            if tt_df.empty:
                continue
            
            for _, row in tt_df.iterrows():
                line = f"{row['training_type']:<15} {int(row['step']):>6}"
                for metric in METRICS:
                    d_value = row[f"{metric}_cohens_d"]
                    if pd.isna(d_value):
                        line += f" {'N/A':>12}"
                    else:
                        line += f" {d_value:>12.4f}"
                print(line)


def get_available_datasets() -> List[str]:
    """Get list of available dataset names from output directory."""
    datasets = set()
    
    # Look for baseline directories to determine available datasets
    # Pattern: baseline_Qwen3-4B_{dataset_name}_{timestamp}
    pattern = str(OUTPUT_DIR / "baseline_Qwen3-4B_*")
    matches = glob.glob(pattern)
    
    for m in matches:
        if not os.path.exists(os.path.join(m, "metrics_summary.csv")):
            continue
        
        dirname = os.path.basename(m)
        parts = dirname.split('_')
        
        # baseline_Qwen3-4B_BA_20251214_190754 -> 5 parts -> BA dataset
        # baseline_Qwen3-4B_CA_20251215_015130 -> 5 parts -> CA dataset
        # baseline_Qwen3-4B_LI_20251215_162525 -> 5 parts -> LI dataset
        if len(parts) == 5:
            datasets.add(parts[2])  # Dataset name is 3rd part (index 2)
    
    return sorted(datasets)


def print_metrics_summary(json_data: Dict[str, Any]):
    """Print a summary of metrics from JSON data."""
    print(f"\n{'='*100}")
    print("METRICS SUMMARY (Mean ± Std)")
    print(f"{'='*100}")
    
    for dataset, training_types in json_data.items():
        print(f"\nDataset: {dataset}")
        print("-" * 95)
        
        # Print header
        header = f"{'Training Type':<15} {'Step':>6} {'Accuracy':>18}"
        for metric in METRICS:
            header += f" {metric[:12]:>20}"
        print(header)
        print("-" * 95)
        
        for training_type in TRAINING_TYPES:
            if training_type not in training_types:
                continue
            
            tt_data = training_types[training_type]
            for i, step in enumerate(tt_data["steps"]):
                accuracy = tt_data["accuracy"][i]
                
                # Get accuracy_std if available
                if "accuracy_std" in tt_data and i < len(tt_data["accuracy_std"]):
                    accuracy_std = tt_data["accuracy_std"][i]
                    line = f"{training_type:<15} {step:>6} {accuracy:>8.1f}±{accuracy_std:<6.1f}%"
                else:
                    line = f"{training_type:<15} {step:>6} {accuracy:>9.1f}%       "
                
                for metric in METRICS:
                    mean_val = tt_data["metrics"][metric]["mean"][i]
                    std_val = tt_data["metrics"][metric]["std"][i]
                    line += f" {mean_val:>8.4f}±{std_val:<8.4f}"
                
                print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Cohen's d effect size and gather metrics for training types vs baseline"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Dataset name (e.g., 'BA', 'CA', 'LI'). Use --all to process all."
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Process all available datasets"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file base name (will create .json and .csv files)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Determine which datasets to process
    if args.all:
        datasets = get_available_datasets()
        if not datasets:
            print("No datasets found with metrics_summary.csv files")
            return
        print(f"Found datasets: {datasets}")
    elif args.dataset is not None:
        datasets = [args.dataset]
    else:
        # Default: list available and prompt user
        available = get_available_datasets()
        print("Available datasets:")
        for d in available:
            print(f"  - {d}")
        print("\nUsage: python src/calculate_cohens_d.py --dataset <name>")
        print("       python src/calculate_cohens_d.py --all")
        return
    
    # Process each dataset
    all_csv_results = []
    all_json_results = {}
    
    for dataset_name in datasets:
        csv_df, json_data = process_dataset(dataset_name, verbose=not args.quiet)
        if csv_df is not None:
            all_csv_results.append(csv_df)
            all_json_results[dataset_name] = json_data
    
    if not all_csv_results:
        print("No results generated")
        return
    
    # Combine CSV results
    final_df = pd.concat(all_csv_results, ignore_index=True)
    
    # Print Cohen's d results table
    print_results_table(final_df)
    
    # Print metrics summary
    print_metrics_summary(all_json_results)
    
    # Determine output paths - include dataset name in filename
    if args.output:
        base_path = Path(args.output)
        csv_path = base_path.with_suffix('.csv')
        json_path = base_path.with_suffix('.json')
    else:
        # Create dataset-specific filenames
        if len(datasets) == 1:
            # Single dataset: include dataset name in filename
            dataset_suffix = datasets[0]
            csv_path = OUTPUT_DIR / f"cohens_d_results_{dataset_suffix}.csv"
            json_path = OUTPUT_DIR / f"metrics_results_{dataset_suffix}.json"
        else:
            # Multiple datasets: use generic names
            csv_path = OUTPUT_DIR / "cohens_d_results.csv"
            json_path = OUTPUT_DIR / "metrics_results.json"
    
    # Save CSV
    final_df.to_csv(csv_path, index=False)
    print(f"\nCohen's d results saved to: {csv_path}")
    
    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(all_json_results, f, indent=2)
    print(f"Full metrics saved to: {json_path}")
    
    # Print interpretation guide
    print("\n" + "="*60)
    print("INTERPRETATION GUIDE")
    print("="*60)
    print("""
Cohen's d effect size interpretation:
  |d| < 0.2  : Negligible effect
  |d| 0.2-0.5: Small effect
  |d| 0.5-0.8: Medium effect
  |d| > 0.8  : Large effect

Positive d means: baseline_step0 > training_stepXX (metric decreased)
Negative d means: baseline_step0 < training_stepXX (metric increased)

JSON structure for plotting:
  {dataset: {training_type: {steps, accuracy, metrics: {metric: {mean, std, cohens_d}}}}}
""")


if __name__ == "__main__":
    main()

