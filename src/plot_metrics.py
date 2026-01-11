#!/usr/bin/env python3
"""
Plot metrics from the JSON output of calculate_cohens_d.py.

Creates three types of plots:
1. Bar plots of metric values (mean ± std) for each training type at each step
2. Bar plots of accuracy for each training type at each step
3. Line plots of Cohen's d across steps for each training type

Usage:
    python src/plot_metrics.py
    python src/plot_metrics.py --input output/metrics_results.json
    python src/plot_metrics.py --output-dir output/plots
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List

# Constants
METRICS = ["necessity", "substantivity", "paraphrasability"]
TRAINING_TYPES = ["baseline", "post-hoc", "internalized", "encoded"]

# Color scheme for training types
COLORS = {
    "baseline": "#2E86AB",      # Blue
    "post-hoc": "#A23B72",      # Magenta
    "internalized": "#F18F01",  # Orange
    "encoded": "#C73E1D"        # Red
}

# Hatching patterns for training types (for bar plots)
HATCHES = {
    "baseline": "",
    "post-hoc": "//",
    "internalized": "\\\\",
    "encoded": "xx"
}

# Line styles for training types
LINE_STYLES = {
    "baseline": "-",
    "post-hoc": "--",
    "internalized": "-.",
    "encoded": ":"
}

# Markers for training types
MARKERS = {
    "baseline": "o",
    "post-hoc": "s",
    "internalized": "^",
    "encoded": "D"
}


def load_json_data(json_path: str) -> Dict[str, Any]:
    """Load metrics data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_metric_bars(data: Dict[str, Any], metric: str, dataset: str, output_dir: Path):
    """
    Create bar plot of a single metric for each training type at each step.
    
    Args:
        data: JSON data for a single dataset
        metric: Metric name (necessity, substantivity, paraphrasability)
        dataset: Dataset name (BA, CA, LI)
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get all steps from the first available training type
    first_tt = next(iter(data.keys()))
    steps = data[first_tt]["steps"]
    n_steps = len(steps)
    n_training_types = len([tt for tt in TRAINING_TYPES if tt in data])
    
    # Bar width and positions
    bar_width = 0.18
    x = np.arange(n_steps)
    
    # Plot bars for each training type (no error bars)
    for i, training_type in enumerate(TRAINING_TYPES):
        if training_type not in data:
            continue
        
        tt_data = data[training_type]
        means = tt_data["metrics"][metric]["mean"]
        
        offset = (i - n_training_types / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, 
            means, 
            bar_width, 
            label=training_type,
            color=COLORS[training_type],
            hatch=HATCHES[training_type],
            edgecolor='black',
            linewidth=0.5
        )
    
    # Customize plot
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel(f'{metric.capitalize()} Score', fontsize=12)
    ax.set_title(f'{metric.capitalize()} by Training Type - Dataset: {dataset}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in steps])
    ax.legend(loc='best', framealpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Tight layout and save
    plt.tight_layout()
    output_path = output_dir / f"metric_{metric}_{dataset}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_accuracy_bars(data: Dict[str, Any], dataset: str, output_dir: Path):
    """
    Create bar plot of accuracy for each training type at each step with error bars.
    
    Args:
        data: JSON data for a single dataset
        dataset: Dataset name (BA, CA, LI)
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get all steps from the first available training type
    first_tt = next(iter(data.keys()))
    steps = data[first_tt]["steps"]
    n_steps = len(steps)
    n_training_types = len([tt for tt in TRAINING_TYPES if tt in data])
    
    # Bar width and positions
    bar_width = 0.18
    x = np.arange(n_steps)
    
    # Plot bars for each training type with error bars
    for i, training_type in enumerate(TRAINING_TYPES):
        if training_type not in data:
            continue
        
        tt_data = data[training_type]
        accuracies = tt_data["accuracy"]
        
        # Get accuracy_std if available, otherwise default to zeros
        if "accuracy_std" in tt_data:
            accuracy_stds = tt_data["accuracy_std"]
        else:
            accuracy_stds = [0.0] * len(accuracies)
        
        offset = (i - n_training_types / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, 
            accuracies, 
            bar_width, 
            yerr=accuracy_stds,
            capsize=3,
            label=training_type,
            color=COLORS[training_type],
            hatch=HATCHES[training_type],
            edgecolor='black',
            linewidth=0.5,
            error_kw={'elinewidth': 1, 'capthick': 1}
        )
    
    # Customize plot
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Accuracy by Training Type - Dataset: {dataset}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_ylim(0, 105)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    # Tight layout and save
    plt.tight_layout()
    output_path = output_dir / f"accuracy_{dataset}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cohens_d_lines(data: Dict[str, Any], metric: str, dataset: str, output_dir: Path):
    """
    Create line plot of Cohen's d across steps for each training type.
    
    Args:
        data: JSON data for a single dataset
        metric: Metric name (necessity, substantivity, paraphrasability)
        dataset: Dataset name (BA, CA, LI)
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get all steps from the first available training type
    first_tt = next(iter(data.keys()))
    steps = data[first_tt]["steps"]
    
    # Plot line for each training type
    for training_type in TRAINING_TYPES:
        if training_type not in data:
            continue
        
        tt_data = data[training_type]
        cohens_d_values = tt_data["metrics"][metric]["cohens_d"]
        
        # Handle None values (convert to NaN for plotting)
        cohens_d_values = [v if v is not None else np.nan for v in cohens_d_values]
        
        ax.plot(
            steps, 
            cohens_d_values,
            label=training_type,
            color=COLORS[training_type],
            linestyle=LINE_STYLES[training_type],
            marker=MARKERS[training_type],
            markersize=8,
            linewidth=2
        )
    
    # Customize plot
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel("Cohen's d", fontsize=12)
    ax.set_title(f"Cohen's d for {metric.capitalize()} - Dataset: {dataset}", fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.grid(alpha=0.3)
    
    # Add effect size reference bands
    ax.axhspan(-0.2, 0.2, alpha=0.1, color='green', label='_negligible')
    ax.axhspan(0.2, 0.5, alpha=0.05, color='yellow')
    ax.axhspan(-0.5, -0.2, alpha=0.05, color='yellow')
    ax.axhspan(0.5, 0.8, alpha=0.05, color='orange')
    ax.axhspan(-0.8, -0.5, alpha=0.05, color='orange')
    
    # Tight layout and save
    plt.tight_layout()
    output_path = output_dir / f"cohens_d_{metric}_{dataset}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cohens_d_combined(data: Dict[str, Any], dataset: str, output_dir: Path):
    """
    Create combined line plot of Cohen's d for all metrics in one figure.
    
    Args:
        data: JSON data for a single dataset
        dataset: Dataset name (BA, CA, LI)
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Get all steps from the first available training type
    first_tt = next(iter(data.keys()))
    steps = data[first_tt]["steps"]
    
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        
        # Plot line for each training type
        for training_type in TRAINING_TYPES:
            if training_type not in data:
                continue
            
            tt_data = data[training_type]
            cohens_d_values = tt_data["metrics"][metric]["cohens_d"]
            
            # Handle None values (convert to NaN for plotting)
            cohens_d_values = [v if v is not None else np.nan for v in cohens_d_values]
            
            ax.plot(
                steps, 
                cohens_d_values,
                label=training_type,
                color=COLORS[training_type],
                linestyle=LINE_STYLES[training_type],
                marker=MARKERS[training_type],
                markersize=8,
                linewidth=2
            )
        
        # Customize subplot
        ax.set_xlabel('Training Step', fontsize=11)
        ax.set_ylabel("Cohen's d", fontsize=11)
        ax.set_title(f'{metric.capitalize()}', fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax.grid(alpha=0.3)
        
        # Add effect size reference bands
        ax.axhspan(-0.2, 0.2, alpha=0.1, color='green')
        
        if idx == 0:
            ax.legend(loc='best', framealpha=0.9)
    
    fig.suptitle(f"Cohen's d by Metric - Dataset: {dataset}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f"cohens_d_combined_{dataset}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_all_metrics_combined(data: Dict[str, Any], dataset: str, output_dir: Path):
    """
    Create combined bar plot showing all metrics side by side for comparison.
    
    Args:
        data: JSON data for a single dataset
        dataset: Dataset name (BA, CA, LI)
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Get all steps from the first available training type
    first_tt = next(iter(data.keys()))
    steps = data[first_tt]["steps"]
    n_steps = len(steps)
    n_training_types = len([tt for tt in TRAINING_TYPES if tt in data])
    
    bar_width = 0.18
    x = np.arange(n_steps)
    
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        
        for i, training_type in enumerate(TRAINING_TYPES):
            if training_type not in data:
                continue
            
            tt_data = data[training_type]
            means = tt_data["metrics"][metric]["mean"]
            
            offset = (i - n_training_types / 2 + 0.5) * bar_width
            ax.bar(
                x + offset, 
                means, 
                bar_width, 
                label=training_type if idx == 0 else "",
                color=COLORS[training_type],
                hatch=HATCHES[training_type],
                edgecolor='black',
                linewidth=0.5
            )
        
        ax.set_xlabel('Training Step', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'{metric.capitalize()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in steps])
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
    
    # Add legend to the first subplot
    axes[0].legend(loc='best', framealpha=0.9)
    
    fig.suptitle(f'All Metrics by Training Type - Dataset: {dataset}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f"metrics_combined_{dataset}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_comparison_across_datasets(all_data: Dict[str, Any], output_dir: Path):
    """
    Create comparison plots across all datasets for each metric.
    
    Args:
        all_data: Full JSON data with all datasets
        output_dir: Directory to save plots
    """
    datasets = list(all_data.keys())
    if len(datasets) < 2:
        return  # Skip if only one dataset
    
    # Cohen's d comparison across datasets
    for metric in METRICS:
        fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=True)
        if len(datasets) == 1:
            axes = [axes]
        
        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            data = all_data[dataset]
            
            first_tt = next(iter(data.keys()))
            steps = data[first_tt]["steps"]
            
            for training_type in TRAINING_TYPES:
                if training_type not in data:
                    continue
                
                tt_data = data[training_type]
                cohens_d_values = tt_data["metrics"][metric]["cohens_d"]
                cohens_d_values = [v if v is not None else np.nan for v in cohens_d_values]
                
                ax.plot(
                    steps, 
                    cohens_d_values,
                    label=training_type if idx == 0 else "",
                    color=COLORS[training_type],
                    linestyle=LINE_STYLES[training_type],
                    marker=MARKERS[training_type],
                    markersize=7,
                    linewidth=2
                )
            
            ax.set_xlabel('Training Step', fontsize=11)
            if idx == 0:
                ax.set_ylabel("Cohen's d", fontsize=11)
            ax.set_title(f'Dataset: {dataset}', fontsize=12, fontweight='bold')
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
            ax.axhspan(-0.2, 0.2, alpha=0.1, color='green')
            ax.grid(alpha=0.3)
        
        axes[0].legend(loc='best', framealpha=0.9)
        fig.suptitle(f"Cohen's d for {metric.capitalize()} Across Datasets", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = output_dir / f"cohens_d_{metric}_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot metrics from calculate_cohens_d.py JSON output"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Dataset name (e.g., 'BA', 'CA', 'LI'). Will look for metrics_results_{dataset}.json"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input JSON file path (overrides --dataset)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for plots (default: output/metrics_plots)"
    )
    
    args = parser.parse_args()
    
    # Set default paths
    script_dir = Path(__file__).parent.parent
    
    # Determine input file
    if args.input:
        json_path = Path(args.input)
    elif args.dataset:
        # Look for dataset-specific JSON file
        json_path = script_dir / "output" / f"metrics_results_{args.dataset}.json"
    else:
        # Default: try generic file
        json_path = script_dir / "output" / "metrics_results.json"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = script_dir / "output" / "metrics_plots"
    
    # Check input file exists
    if not json_path.exists():
        print(f"ERROR: Input file not found: {json_path}")
        if args.dataset:
            print(f"Run: python src/calculate_cohens_d.py --dataset {args.dataset}")
        else:
            print("Run calculate_cohens_d.py first to generate the JSON file.")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {json_path}")
    all_data = load_json_data(json_path)
    
    datasets = list(all_data.keys())
    print(f"Found datasets: {datasets}")
    
    # Generate plots for each dataset
    for dataset in datasets:
        print(f"\nGenerating plots for dataset: {dataset}")
        data = all_data[dataset]
        
        # 1. Bar plots for each metric
        print("  Creating metric bar plots...")
        for metric in METRICS:
            plot_metric_bars(data, metric, dataset, output_dir)
        
        # Combined metrics plot
        plot_all_metrics_combined(data, dataset, output_dir)
        
        # 2. Accuracy bar plot
        print("  Creating accuracy bar plot...")
        plot_accuracy_bars(data, dataset, output_dir)
        
        # 3. Cohen's d line plots
        print("  Creating Cohen's d line plots...")
        for metric in METRICS:
            plot_cohens_d_lines(data, metric, dataset, output_dir)
        
        # Combined Cohen's d plot
        plot_cohens_d_combined(data, dataset, output_dir)
    
    # Cross-dataset comparison plots
    if len(datasets) > 1:
        print("\nGenerating cross-dataset comparison plots...")
        plot_comparison_across_datasets(all_data, output_dir)
    
    print(f"\n{'='*60}")
    print(f"All plots saved to: {output_dir}")
    print(f"{'='*60}")
    
    # Summary of generated plots
    print("\nGenerated plots:")
    print("  Per dataset:")
    print("    - metric_{metric}_{dataset}.png      : Bar plot of metric values")
    print("    - metrics_combined_{dataset}.png     : All metrics in one figure")
    print("    - accuracy_{dataset}.png             : Bar plot of accuracy")
    print("    - cohens_d_{metric}_{dataset}.png    : Line plot of Cohen's d")
    print("    - cohens_d_combined_{dataset}.png    : All Cohen's d in one figure")
    if len(datasets) > 1:
        print("  Cross-dataset comparison:")
        print("    - cohens_d_{metric}_comparison.png  : Compare datasets side by side")


if __name__ == "__main__":
    main()

