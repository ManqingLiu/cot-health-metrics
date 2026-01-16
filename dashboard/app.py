#!/usr/bin/env python3
"""
Real-time Training Monitoring Dashboard

A comprehensive Streamlit dashboard for tracking:
- Accuracy with error bars (Standard Error)
- Original metrics (Necessity, Substantivity, Paraphrasability)
- Cohen's d effect sizes
- Per-dataset comparisons across training types

Error Bar Methodology:
- Accuracy: Binomial Standard Error = sqrt(p * (1-p) / n)
  where p is accuracy proportion and n is sample size
- Other metrics: Standard Error of Mean (SEM) = std / sqrt(n)

W&B Naming Convention:
- Projects: {training_type}-lr-sweep-{model_name}-{dataset_name}
- Runs: {training_type}_{dataset_name}_lr{learning_rate}

Usage:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import glob
import os
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="CoT Health Metrics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DEFAULT_TRAINING_TYPES = ["baseline", "internalized", "encoded", "post-hoc"]
DEFAULT_MODEL_NAMES = ["Qwen3-4B"]
DEFAULT_DATASET_NAMES = ["ba", "ca", "sb"]
# Learning rates: BA uses 1e-5, CA and SB use 5e-5
DEFAULT_LEARNING_RATES = ["5e-5", "1e-5", "2e-5", "1e-4"]

# Default sample size for SE calculation (can be overridden if data provides it)
DEFAULT_SAMPLE_SIZE = 100

# Metrics available
ORIGINAL_METRICS = ["necessity", "substantivity", "paraphrasability"]
ALL_METRICS = ["accuracy"] + ORIGINAL_METRICS

# Y-axis ranges removed - plots will auto-scale based on data

# Color schemes for training types
COLORS_BY_TRAINING_TYPE = {
    "baseline": "#2E86AB",      # Blue (solid)
    "post-hoc": "#A23B72",      # Magenta/Purple
    "internalized": "#F18F01",  # Orange
    "encoded": "#C73E1D"        # Red
}

# Line styles for training types
LINE_STYLES = {
    "baseline": "solid",
    "post-hoc": "dash",
    "internalized": "dashdot",
    "encoded": "dot"
}

# Marker symbols for training types
MARKER_SYMBOLS = {
    "baseline": "circle",
    "post-hoc": "square",
    "internalized": "triangle-up",
    "encoded": "diamond"
}

# Bar patterns for training types
BAR_PATTERNS = {
    "baseline": "",
    "post-hoc": "/",
    "internalized": "x",
    "encoded": "\\"
}

COLORS_BY_DATASET = {
    "ca": "#1f77b4",
    "ba": "#ff7f0e",
    "sb": "#2ca02c",
    "li": "#d62728",
}

COLORS_BY_LR = {
    "1e-5": "#2ca02c",
    "2e-5": "#9467bd",
    "5e-5": "#ff7f0e",
    "1e-4": "#1f77b4",
}


# =============================================================================
# Error Bar Calculation Functions (Standard Methods)
# =============================================================================

def calculate_binomial_se(p: float, n: int = DEFAULT_SAMPLE_SIZE) -> float:
    """
    Calculate Standard Error for accuracy using binomial proportion formula.
    
    SE = sqrt(p * (1 - p) / n)
    
    This is the standard method for calculating error bars on accuracy metrics,
    as accuracy follows a binomial distribution.
    
    Args:
        p: Accuracy as a proportion (0-1)
        n: Sample size (number of test examples)
    
    Returns:
        Standard Error
    """
    # Convert to numeric types, handling non-numeric values
    try:
        p = pd.to_numeric(p, errors='coerce')
        n = pd.to_numeric(n, errors='coerce')
    except (TypeError, ValueError):
        return np.nan
    
    if pd.isna(p) or pd.isna(n) or n <= 0:
        return np.nan
    
    # Ensure both are numpy numeric types
    p = float(p)
    n = float(n)
    
    if n <= 0:
        return np.nan
    
    # Ensure p is in [0, 1]
    p = np.clip(p, 0, 1)
    return np.sqrt(p * (1 - p) / n)


def calculate_sem(std: float, n: int = DEFAULT_SAMPLE_SIZE) -> float:
    """
    Calculate Standard Error of Mean (SEM).
    
    SEM = std / sqrt(n)
    
    This is the standard method for calculating error bars on continuous metrics.
    
    Args:
        std: Standard deviation of the metric
        n: Sample size
    
    Returns:
        Standard Error of Mean
    """
    # Convert to numeric types, handling non-numeric values
    try:
        std = pd.to_numeric(std, errors='coerce')
        n = pd.to_numeric(n, errors='coerce')
    except (TypeError, ValueError):
        return np.nan
    
    if pd.isna(std) or pd.isna(n) or n <= 0:
        return np.nan
    
    # Ensure both are numpy numeric types
    std = float(std)
    n = float(n)
    
    if n <= 0:
        return np.nan
    
    return std / np.sqrt(n)


def calculate_cohens_d(mean1: float, std1: float, mean2: float, std2: float) -> float:
    """Calculate Cohen's d effect size. d = (mean1 - mean2) / pooled_std"""
    # Convert to numeric types, handling strings and None values
    try:
        mean1 = pd.to_numeric(mean1, errors='coerce')
        mean2 = pd.to_numeric(mean2, errors='coerce')
        std1 = pd.to_numeric(std1, errors='coerce')
        std2 = pd.to_numeric(std2, errors='coerce')
    except (TypeError, ValueError):
        return np.nan
    
    # Check for NaN values
    if pd.isna(mean1) or pd.isna(mean2) or pd.isna(std1) or pd.isna(std2):
        return np.nan
    
    # Handle None or zero std values
    if std1 is None or std1 == 0:
        std1 = 0.1
    if std2 is None or std2 == 0:
        std2 = 0.1
    
    # Ensure they are float types
    std1 = float(std1)
    std2 = float(std2)
    mean1 = float(mean1)
    mean2 = float(mean2)
    
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    if pooled_std == 0 or np.isnan(pooled_std):
        return np.nan
    return (mean1 - mean2) / pooled_std


def effect_size_label(d: float) -> str:
    """Get human-readable effect size label."""
    if np.isnan(d):
        return "N/A"
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


# =============================================================================
# Parsing Functions
# =============================================================================

def parse_project_name(project: str) -> Dict[str, str]:
    """Parse W&B project name to extract components."""
    parts = project.split('-')
    result = {
        "training_type": "unknown",
        "model_name": "unknown",
        "dataset_name": "unknown"
    }
    
    if len(parts) >= 5:
        result["training_type"] = parts[0]
        result["dataset_name"] = parts[-1]
        model_parts = parts[3:-1]
        result["model_name"] = "-".join(model_parts)
    
    return result


def parse_run_name(run_name: str) -> Dict[str, str]:
    """Parse W&B run name to extract components."""
    result = {
        "training_type": "unknown",
        "dataset_name": "unknown", 
        "learning_rate": "unknown"
    }
    
    pattern = r"^(\w+)_(\w+)_lr(.+)$"
    match = re.match(pattern, run_name)
    
    if match:
        result["training_type"] = match.group(1)
        result["dataset_name"] = match.group(2)
        result["learning_rate"] = match.group(3)
    else:
        parts = run_name.split('_')
        if len(parts) >= 1:
            result["training_type"] = parts[0]
        if len(parts) >= 2:
            result["dataset_name"] = parts[1]
        if len(parts) >= 3 and parts[2].startswith("lr"):
            result["learning_rate"] = parts[2][2:]
    
    return result


def generate_project_name(training_type: str, model_name: str, dataset_name: str) -> str:
    """Generate W&B project name from components."""
    return f"{training_type}-lr-sweep-{model_name}-{dataset_name}"


# =============================================================================
# Data Loading Functions
# =============================================================================

def get_secret(key: str, default: str = "") -> str:
    """Safely get a secret from Streamlit secrets, returning default if not found."""
    try:
        return st.secrets.get(key, default)
    except:
        return default


def setup_wandb_auth():
    """Setup W&B authentication from Streamlit secrets or environment.
    
    For PUBLIC W&B projects, no API key is needed for read-only access.
    For private projects, set WANDB_API_KEY in secrets or environment.
    """
    try:
        import wandb
        # Check if API key is in Streamlit secrets (for cloud deployment)
        api_key = get_secret("WANDB_API_KEY")
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
        # If no API key is set, wandb will work in anonymous/public mode
        return True
    except Exception as e:
        return False


@st.cache_data(ttl=60)
def load_from_wandb_multi(entity: str, projects: List[str], run_states: Optional[List[str]] = None, debug: bool = False) -> Optional[pd.DataFrame]:
    """Load metrics from multiple W&B projects.
    
    Args:
        entity: W&B entity name
        projects: List of project names
        run_states: List of run states to include (e.g., ['running', 'finished']). 
                    If None, defaults to ['finished', 'running']
        debug: Whether to return debug information
    
    When there are multiple runs with the same name, keeps only the latest run based on state preference.
    """
    try:
        import wandb
        setup_wandb_auth()
        api = wandb.Api()
        
        # Default to both running and finished if not specified
        if run_states is None:
            run_states = ['finished', 'running']
        valid_states = set(run_states)
        
        all_data = []
        debug_info = []
        
        for project in projects:
            try:
                project_info = parse_project_name(project)
                runs = list(api.runs(f"{entity}/{project}"))
                
                # Filter to only specified run states (exclude crashed/failed unless included)
                valid_runs = [run for run in runs if run.state in valid_states]
                
                # Group runs by name and keep the best one:
                # - If both 'running' and 'finished' are in valid_states, prefer 'running' (latest attempt)
                # - Otherwise, prefer the state with higher priority (running > finished > others)
                # - If same state, prefer the latest (newest created_at)
                state_priority = {'running': 3, 'finished': 2, 'crashed': 1, 'failed': 1, 'killed': 1}
                
                runs_by_name = {}
                for run in valid_runs:
                    run_name = run.name
                    run_state = run.state
                    
                    # Get creation time for comparison
                    try:
                        created_at = run.created_at
                    except:
                        created_at = None
                    
                    if run_name not in runs_by_name:
                        runs_by_name[run_name] = {
                            'run': run,
                            'state': run_state,
                            'created_at': created_at,
                            'priority': state_priority.get(run_state, 0)
                        }
                    else:
                        existing = runs_by_name[run_name]
                        existing_priority = existing['priority']
                        current_priority = state_priority.get(run_state, 0)
                        
                        # Prefer higher priority state (running > finished > others)
                        if current_priority > existing_priority:
                            runs_by_name[run_name] = {
                                'run': run,
                                'state': run_state,
                                'created_at': created_at,
                                'priority': current_priority
                            }
                        # If same priority/state, prefer the newer one
                        elif current_priority == existing_priority:
                            if created_at and existing['created_at']:
                                if created_at > existing['created_at']:
                                    runs_by_name[run_name] = {
                                        'run': run,
                                        'state': run_state,
                                        'created_at': created_at,
                                        'priority': current_priority
                                    }
                
                # Process the selected runs
                for run_name, run_info in runs_by_name.items():
                    run = run_info['run']
                    config = run.config
                    run_info_parsed = parse_run_name(run_name)
                    
                    training_type = run_info_parsed.get("training_type") or project_info.get("training_type") or config.get("training_type", "unknown")
                    dataset_name = run_info_parsed.get("dataset_name") or project_info.get("dataset_name") or config.get("dataset_name", "unknown")
                    model_name = project_info.get("model_name") or config.get("model", "unknown").split('/')[-1]
                    learning_rate = run_info_parsed.get("learning_rate") or str(config.get("learning_rate", "unknown"))
                    
                    # Get sample size from config if available
                    sample_size = config.get("metric_eval_samples", DEFAULT_SAMPLE_SIZE)
                    
                    # Fetch history - try with desired keys first, fallback to fetching all
                    desired_keys = [
                        "step",
                        "eval/accuracy", "eval/accuracy_mean", "eval/accuracy_std",
                        "eval/num_total",
                        "eval/substantivity_mean", "eval/substantivity_std",
                        "eval/necessity_mean", "eval/necessity_std",
                        "eval/paraphrasability_mean", "eval/paraphrasability_std"
                    ]
                    
                    history = None
                    available_keys = []
                    
                    try:
                        # Try fetching with desired keys first
                        history = run.history(keys=desired_keys)
                        # If that returns empty, try fetching all history to see what's available
                        if history.empty:
                            history_all = run.history()
                            if not history_all.empty:
                                available_keys = list(history_all.columns)
                                # Check if any of our desired keys exist
                                existing_keys = [k for k in desired_keys if k in available_keys]
                                if existing_keys:
                                    history = run.history(keys=existing_keys)
                                else:
                                    # No matching keys - keep empty but record what keys exist
                                    history = pd.DataFrame()
                    except Exception as e:
                        # If history() fails, try with just step
                        try:
                            history = run.history(keys=["step"])
                        except:
                            history = pd.DataFrame()
                    
                    # Debug: track what we got
                    debug_info.append({
                        "project": project,
                        "run_name": run_name,
                        "run_state": run.state,
                        "created_at": str(run_info['created_at'])[:19] if run_info['created_at'] else "N/A",
                        "parsed_lr": learning_rate,
                        "parsed_dataset": dataset_name,
                        "rows_fetched": len(history) if history is not None and not history.empty else 0,
                        "config_lr": config.get("learning_rate", "N/A"),
                        "available_keys": ", ".join(available_keys[:15]) if available_keys else "none",  # Show available keys if history was empty
                    })
                    
                    if history is None or history.empty:
                        continue
                    
                    rename_map = {
                        "eval/accuracy": "accuracy",
                        "eval/accuracy_mean": "accuracy_mean",
                        "eval/accuracy_std": "accuracy_std",
                        "eval/num_total": "num_samples",
                        "eval/substantivity_mean": "substantivity_mean",
                        "eval/substantivity_std": "substantivity_std",
                        "eval/necessity_mean": "necessity_mean",
                        "eval/necessity_std": "necessity_std",
                        "eval/paraphrasability_mean": "paraphrasability_mean",
                        "eval/paraphrasability_std": "paraphrasability_std"
                    }
                    history = history.rename(columns={k: v for k, v in rename_map.items() if k in history.columns})
                    
                    if 'accuracy_mean' in history.columns and 'accuracy' not in history.columns:
                        history['accuracy'] = history['accuracy_mean']
                    
                    # Add sample size if not present
                    if 'num_samples' not in history.columns:
                        history['num_samples'] = sample_size
                    
                    history["run_name"] = run_name
                    history["project"] = project
                    history["training_type"] = training_type
                    history["dataset"] = dataset_name
                    history["model"] = model_name
                    history["learning_rate"] = learning_rate
                    history["source"] = "wandb"
                    history["run_state"] = run.state
                    history["created_at"] = run_info['created_at']

                    all_data.append(history)
                    
            except Exception as e:
                st.warning(f"Error loading project {project}: {e}")
                continue
        
        # Store debug info in session state
        if debug_info:
            st.session_state['wandb_debug_info'] = debug_info
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)

            # Deduplicate: keep only the latest run per (training_type, dataset, learning_rate)
            # This handles cases where the same experiment was run multiple times
            if 'created_at' in combined.columns and combined['created_at'].notna().any():
                # Get unique runs with their created_at
                run_info = combined.groupby('run_name').agg({
                    'training_type': 'first',
                    'dataset': 'first',
                    'learning_rate': 'first',
                    'run_state': 'first',
                    'created_at': 'first'
                }).reset_index()

                # Sort by created_at descending and keep first (latest) per config
                run_info = run_info.sort_values('created_at', ascending=False)
                latest_runs = run_info.drop_duplicates(
                    subset=['training_type', 'dataset', 'learning_rate'],
                    keep='first'
                )['run_name'].tolist()

                # Filter combined data to only include latest runs
                combined = combined[combined['run_name'].isin(latest_runs)]

            return combined
        return None
        
    except ImportError:
        st.warning("W&B not installed. Install with: pip install wandb")
        return None
    except Exception as e:
        st.error(f"Error loading from W&B: {e}")
        return None


@st.cache_data(ttl=30)
def load_from_local(output_dir: str) -> Optional[pd.DataFrame]:
    """Load metrics from local metrics_history.json files."""
    all_data = []
    pattern = os.path.join(output_dir, "**/metrics_history.json")
    
    for json_path in glob.glob(pattern, recursive=True):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if not data:
                continue
            
            run_dir = os.path.dirname(json_path)
            run_name = os.path.basename(run_dir)
            
            parts = run_name.split('_')
            training_type = parts[0] if len(parts) >= 1 else "unknown"
            model_name = parts[1] if len(parts) >= 2 else "unknown"
            dataset = parts[2] if len(parts) >= 3 else "unknown"
            
            df = pd.DataFrame(data)
            df['run_name'] = run_name
            df['training_type'] = training_type
            df['dataset'] = dataset
            df['model'] = model_name
            df['learning_rate'] = "unknown"
            df['num_samples'] = DEFAULT_SAMPLE_SIZE
            df['source'] = "local"
            df['last_modified'] = datetime.fromtimestamp(os.path.getmtime(json_path))
            
            all_data.append(df)
            
        except Exception as e:
            st.warning(f"Error loading {json_path}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None


@st.cache_data(ttl=300)  # Cache for 5 minutes to reduce repeated loading
def load_sample_cots_from_wandb(entity: str, projects: List[str], debug: bool = False, max_runs_per_project: int = 4) -> Optional[pd.DataFrame]:
    """
    Load sample CoTs (prompt, reasoning, answer) from W&B tables.

    W&B Tables logged via wandb.log() are stored as media files. We access them via:
    1. Run files (media/table/eval/sample_cots_*.json)
    2. History scan to get step information

    Returns DataFrame with columns:
    - step, question_id, question, prompt, cot, answer, training_type, dataset, learning_rate

    Optimized: Limits runs per project to speed up loading.
    """
    try:
        import wandb
        setup_wandb_auth()
        api = wandb.Api()

        all_cots = []
        debug_info = []

        for project in projects:
            try:
                project_info = parse_project_name(project)
                runs = list(api.runs(f"{entity}/{project}"))
                
                # Filter to valid runs
                valid_states = {'finished', 'running'}
                valid_runs = [run for run in runs if run.state in valid_states]
                
                # Group runs by name and keep the latest
                runs_by_name = {}
                for run in valid_runs:
                    run_name = run.name
                    try:
                        created_at = run.created_at
                    except:
                        created_at = None
                    
                    if run_name not in runs_by_name:
                        runs_by_name[run_name] = {'run': run, 'created_at': created_at}
                    else:
                        existing = runs_by_name[run_name]
                        if created_at and existing['created_at']:
                            if created_at > existing['created_at']:
                                runs_by_name[run_name] = {'run': run, 'created_at': created_at}
                
                # Limit number of runs to process for performance
                runs_to_process = list(runs_by_name.items())[:max_runs_per_project]

                for run_name, run_info in runs_to_process:
                    run = run_info['run']
                    run_info_parsed = parse_run_name(run_name)

                    training_type = run_info_parsed.get("training_type") or project_info.get("training_type", "unknown")
                    dataset_name = run_info_parsed.get("dataset_name") or project_info.get("dataset_name", "unknown")
                    learning_rate = run_info_parsed.get("learning_rate", "unknown")

                    tables_found = 0
                    error_messages = []

                    # Method 1 (PRIMARY): Use scan_history to get tables at ALL steps
                    # This is more reliable than downloading files and preserves step information
                    try:
                        # Scan history to find steps where sample_cots were logged
                        for row in run.scan_history(keys=["step", "eval/sample_cots"]):
                            step = row.get("step", 0)
                            table_ref = row.get("eval/sample_cots")

                            if table_ref is not None:
                                try:
                                    table_df = None
                                    # table_ref might be a wandb.Table object, dict, or file path
                                    if hasattr(table_ref, 'get_dataframe'):
                                        # It's a wandb.Table object
                                        table_df = table_ref.get_dataframe()
                                    elif isinstance(table_ref, dict):
                                        # It's a dict with columns and data
                                        if 'columns' in table_ref and 'data' in table_ref:
                                            table_df = pd.DataFrame(table_ref['data'], columns=table_ref['columns'])
                                    elif isinstance(table_ref, str):
                                        # It's a file path - try to download
                                        try:
                                            file = run.file(table_ref)
                                            file_path = file.download(replace=True)
                                            with open(file_path.name, 'r', encoding='utf-8') as f:
                                                table_data = json.load(f)
                                            if 'columns' in table_data and 'data' in table_data:
                                                table_df = pd.DataFrame(table_data['data'], columns=table_data['columns'])
                                        except:
                                            continue

                                    if table_df is not None and not table_df.empty:
                                        # Ensure step column from the history step
                                        if 'step' not in table_df.columns:
                                            table_df['step'] = step

                                        # Add metadata
                                        table_df['training_type'] = training_type
                                        table_df['dataset'] = dataset_name
                                        table_df['learning_rate'] = learning_rate
                                        table_df['run_name'] = run_name

                                        all_cots.append(table_df)
                                        tables_found += 1

                                except Exception as e:
                                    error_messages.append(f"History scan error at step {step}: {str(e)}")
                                    continue

                    except Exception as e:
                        error_messages.append(f"History scan failed: {str(e)}")

                    # Method 2 (FALLBACK): Get tables from run files if scan_history found nothing
                    if tables_found == 0:
                        try:
                            files = list(run.files())
                            table_files = [f for f in files if 'sample_cots' in f.name.lower() and f.name.endswith('.json')]
                            table_files = table_files[:10]  # Limit for performance

                            for file in table_files:
                                try:
                                    file_path = file.download(replace=True)
                                    with open(file_path.name, 'r', encoding='utf-8') as f:
                                        table_data = json.load(f)

                                    if 'columns' in table_data and 'data' in table_data:
                                        table_df = pd.DataFrame(table_data['data'], columns=table_data['columns'])
                                        table_df['training_type'] = training_type
                                        table_df['dataset'] = dataset_name
                                        table_df['learning_rate'] = learning_rate
                                        table_df['run_name'] = run_name
                                        if 'step' not in table_df.columns:
                                            table_df['step'] = 0
                                        all_cots.append(table_df)
                                        tables_found += 1
                                except Exception as e:
                                    error_messages.append(f"File {file.name}: {str(e)}")
                        except Exception as e:
                            error_messages.append(f"File access error: {str(e)}")
                    
                    debug_info.append({
                        'project': project,
                        'run_name': run_name,
                        'tables_found': tables_found,
                        'errors': error_messages[:3]  # Limit error messages
                    })
                        
            except Exception as e:
                debug_info.append({'project': project, 'error': str(e)})
                continue
        
        # Store debug info
        if debug_info:
            st.session_state['cot_debug_info'] = debug_info
        
        if all_cots:
            result = pd.concat(all_cots, ignore_index=True)
            
            # Ensure required columns exist
            for col in ['step', 'question_id', 'question', 'prompt', 'cot', 'answer']:
                if col not in result.columns:
                    result[col] = ""
            
            # Convert step to numeric if it's not already
            if 'step' in result.columns:
                result['step'] = pd.to_numeric(result['step'], errors='coerce').fillna(0).astype(int)
            
            # Filter out invalid samples where CoT is empty
            original_count = len(result)
            result = filter_valid_cot_samples(result)
            filtered_count = len(result)
            if original_count > filtered_count:
                result.attrs['filtered_invalid_count'] = original_count - filtered_count
            
            # Return all valid CoTs (sampling moved to display time)
            return result if not result.empty else None
        return None
        
    except ImportError:
        return None
    except Exception as e:
        if debug:
            st.error(f"Error loading sample CoTs: {e}")
        return None


def normalize_question_id(qid) -> float:
    """
    Normalize question_id to a numeric value for comparison.

    Handles various formats: int (26), float (26.0), string ("26", "26.0")
    Returns float for consistent comparison, or float('nan') if not convertible.
    """
    if qid is None or qid == '':
        return float('nan')
    try:
        return float(qid)
    except (ValueError, TypeError):
        return float('nan')


def question_ids_match(qid1, qid2) -> bool:
    """
    Compare two question_ids numerically.

    Returns True if both can be converted to the same numeric value.
    E.g., 26 == "26" == "26.0" == 26.0
    """
    norm1 = normalize_question_id(qid1)
    norm2 = normalize_question_id(qid2)
    if pd.isna(norm1) or pd.isna(norm2):
        return False
    return norm1 == norm2


def filter_valid_cot_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out invalid CoT samples.
    
    A sample is considered invalid if:
    - CoT is empty, None, or only whitespace (model failed to produce Answer: token)
    - Answer is empty, None, or only whitespace
    
    These typically occur when the model exceeds the maximum token budget
    without producing a valid answer.
    """
    if df is None or df.empty:
        return df
    
    # Create mask for valid samples
    valid_mask = pd.Series([True] * len(df), index=df.index)
    
    # Check CoT validity
    if 'cot' in df.columns:
        cot_valid = df['cot'].apply(lambda x: 
            x is not None and 
            str(x).strip() != '' and 
            str(x).strip().lower() not in ['none', 'nan', 'null', '(no reasoning captured)']
        )
        valid_mask = valid_mask & cot_valid
    
    # Check answer validity
    if 'answer' in df.columns:
        answer_valid = df['answer'].apply(lambda x:
            x is not None and
            str(x).strip() != '' and
            str(x).strip().lower() not in ['none', 'nan', 'null', '(no answer)']
        )
        valid_mask = valid_mask & answer_valid
    
    return df[valid_mask].reset_index(drop=True)


def sample_cots_per_step(df: pd.DataFrame, n_samples: int = 5, random_seed: int = 42) -> pd.DataFrame:
    """
    Sample n_samples valid CoTs per step AND training_type.

    Groups by step and training_type, then randomly samples up to n_samples valid CoTs
    from each group. This ensures each training type is represented at each step.

    Args:
        df: DataFrame with CoT samples (must have 'step' column)
        n_samples: Number of samples to keep per step per training_type (default: 5)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        DataFrame with sampled CoTs
    """
    if df is None or df.empty:
        return df

    if 'step' not in df.columns:
        # If no step column, return all data
        return df

    # Group by step AND training_type to ensure each training type is represented
    group_cols = ['step']
    if 'training_type' in df.columns:
        group_cols.append('training_type')

    sampled_dfs = []
    for group_key, group in df.groupby(group_cols):
        if len(group) > n_samples:
            # Randomly sample n_samples
            sampled = group.sample(n=n_samples, random_state=random_seed)
        else:
            # Keep all if we have fewer than n_samples
            sampled = group
        sampled_dfs.append(sampled)

    if sampled_dfs:
        result = pd.concat(sampled_dfs, ignore_index=True)
        return result
    else:
        return df


def filter_shared_questions(df: pd.DataFrame, training_types: List[str], require_all: bool = True) -> pd.DataFrame:
    """
    Filter to questions that appear across multiple training types.

    Args:
        df: DataFrame with CoT samples
        training_types: List of training types to check for shared questions
        require_all: If True, keep only questions that appear in ALL training types.
                    If False, keep questions that appear in ANY of the training types.

    Returns:
        DataFrame filtered to shared questions
    """
    if df is None or df.empty:
        return df

    if 'question_id' not in df.columns or 'training_type' not in df.columns:
        return df

    if len(training_types) <= 1:
        # No sharing filter needed for single training type
        return df

    # For each step, find questions that appear in all/any selected training types
    result_dfs = []

    steps = df['step'].unique() if 'step' in df.columns else [0]

    for step in steps:
        step_data = df[df['step'] == step] if 'step' in df.columns else df

        # Get question_ids for each training type at this step
        questions_by_tt = {}
        for tt in training_types:
            tt_data = step_data[step_data['training_type'] == tt]
            questions_by_tt[tt] = set(tt_data['question_id'].dropna().unique())

        if require_all:
            # Questions that appear in ALL selected training types
            if all(len(q) > 0 for q in questions_by_tt.values()):
                shared_questions = set.intersection(*questions_by_tt.values())
            else:
                shared_questions = set()
        else:
            # Questions that appear in ANY selected training type
            shared_questions = set.union(*questions_by_tt.values()) if questions_by_tt else set()

        if shared_questions:
            step_filtered = step_data[step_data['question_id'].isin(shared_questions)]
            result_dfs.append(step_filtered)

    if result_dfs:
        return pd.concat(result_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_sample_cots_from_local(output_dir: str) -> Optional[pd.DataFrame]:
    """Load sample CoTs from local sample_cots.json files."""
    all_cots = []
    
    # Look for sample_cots.json in checkpoint directories
    pattern = os.path.join(output_dir, "**/sample_cots.json")
    
    for json_path in glob.glob(pattern, recursive=True):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if not data:
                continue
            
            # Parse directory structure: output/{training_type}_{model}_{dataset}_{timestamp}/checkpoint-{step}/sample_cots.json
            checkpoint_dir = os.path.dirname(json_path)
            run_dir = os.path.dirname(checkpoint_dir)
            run_name = os.path.basename(run_dir)
            checkpoint_name = os.path.basename(checkpoint_dir)
            
            # Extract step from checkpoint name
            step_match = re.search(r'checkpoint-(\d+)', checkpoint_name)
            step = int(step_match.group(1)) if step_match else 0
            
            # Parse run name
            parts = run_name.split('_')
            training_type = parts[0] if len(parts) >= 1 else "unknown"
            dataset = parts[2] if len(parts) >= 3 else "unknown"
            
            # Create DataFrame
            df = pd.DataFrame(data)
            df['step'] = step
            df['training_type'] = training_type
            df['dataset'] = dataset
            df['run_name'] = run_name
            df['learning_rate'] = "unknown"
            
            all_cots.append(df)
            
        except Exception as e:
            continue
    
    if all_cots:
        result = pd.concat(all_cots, ignore_index=True)
        # Ensure required columns exist
        for col in ['step', 'question_id', 'question', 'prompt', 'cot', 'answer']:
            if col not in result.columns:
                result[col] = ""
        
        # Filter out invalid samples where CoT is empty
        # Empty CoT indicates the model failed to produce an Answer: token within the max token budget
        original_count = len(result)
        result = filter_valid_cot_samples(result)
        filtered_count = len(result)
        if original_count > filtered_count:
            # Store filter info for display
            result.attrs['filtered_invalid_count'] = original_count - filtered_count

        # Return all valid CoTs (sampling moved to display time)
        return result if not result.empty else None
    return None


# =============================================================================
# Anthropic-Style CoT Visualization CSS and Components
# =============================================================================

ANTHROPIC_COT_CSS = """
<style>
/* Anthropic-style CoT visualization */
.cot-container {
    font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
    max-width: 100%;
    margin: 1rem 0;
}

/* Prompt Section */
.prompt-section {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    position: relative;
}

.prompt-section::before {
    content: "PROMPT";
    position: absolute;
    top: -10px;
    left: 16px;
    background: #e94560;
    color: white;
    padding: 2px 12px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.prompt-section pre {
    color: #edf2f4;
    margin: 0;
    white-space: pre-wrap;
    word-wrap: break-word;
    font-size: 0.9rem;
    line-height: 1.5;
}

/* Thinking/CoT Section - Anthropic Extended Thinking Style */
.thinking-section {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #30363d;
    border-left: 4px solid #58a6ff;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    position: relative;
}

.thinking-section::before {
    content: "⚡ THINKING";
    position: absolute;
    top: -10px;
    left: 16px;
    background: linear-gradient(90deg, #58a6ff, #79c0ff);
    color: #0d1117;
    padding: 2px 12px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.thinking-section pre {
    color: #c9d1d9;
    margin: 0;
    white-space: pre-wrap;
    word-wrap: break-word;
    font-size: 0.85rem;
    line-height: 1.6;
    font-style: italic;
}

/* Collapsible thinking for long CoT */
.thinking-collapsed {
    max-height: 200px;
    overflow: hidden;
    position: relative;
}

.thinking-collapsed::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 60px;
    background: linear-gradient(transparent, #161b22);
}

/* Answer Section */
.answer-section {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 1.25rem;
    position: relative;
}

.answer-section::before {
    content: "✓ ANSWER";
    position: absolute;
    top: -10px;
    left: 16px;
    background: #10b981;
    color: white;
    padding: 2px 12px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.answer-section pre {
    color: #d1fae5;
    margin: 0;
    white-space: pre-wrap;
    word-wrap: break-word;
    font-size: 0.95rem;
    line-height: 1.5;
    font-weight: 500;
}

/* Training Type Badge */
.training-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 8px;
    text-transform: uppercase;
}

.training-badge.baseline { background: #2E86AB; color: white; }
.training-badge.post-hoc { background: #A23B72; color: white; }
.training-badge.internalized { background: #F18F01; color: white; }
.training-badge.encoded { background: #C73E1D; color: white; }

/* Step Badge */
.step-badge {
    display: inline-block;
    padding: 4px 12px;
    background: #4a5568;
    color: white;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Comparison Grid */
.comparison-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 1.5rem;
    margin-top: 1rem;
}

/* Sample Card */
.sample-card {
    background: #1e1e2e;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}

.sample-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #30363d;
}

/* Question Section */
.question-section {
    background: #2d2d44;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.question-section h4 {
    color: #a78bfa;
    margin: 0 0 0.5rem 0;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.question-section p {
    color: #e2e8f0;
    margin: 0;
    font-size: 0.9rem;
}

/* W&B Style Table */
.wandb-table {
    width: 100%;
    border-collapse: collapse;
    background: white;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    margin: 1rem 0;
}

.wandb-table thead {
    background: #f7f7f7;
    border-bottom: 2px solid #e0e0e0;
}

.wandb-table th {
    padding: 12px 16px;
    text-align: left;
    font-weight: 600;
    font-size: 0.875rem;
    color: #333;
    border-right: 1px solid #e0e0e0;
}

.wandb-table th:last-child {
    border-right: none;
}

.wandb-table tbody tr {
    border-bottom: 1px solid #e0e0e0;
    transition: background-color 0.2s;
}

.wandb-table tbody tr:hover {
    background-color: #f9f9f9;
}

.wandb-table td {
    padding: 12px 16px;
    font-size: 0.875rem;
    color: #333;
    border-right: 1px solid #e0e0e0;
    vertical-align: top;
}

.wandb-table td:last-child {
    border-right: none;
}

.wandb-table .text-cell {
    max-width: 300px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.wandb-table .text-cell.expanded {
    white-space: normal;
    max-width: none;
}

.wandb-table .expand-btn {
    color: #1976d2;
    cursor: pointer;
    text-decoration: none;
    font-size: 0.75rem;
    margin-left: 8px;
}

.wandb-table .expand-btn:hover {
    text-decoration: underline;
}

.wandb-table .full-text {
    display: none;
    margin-top: 8px;
    padding: 8px;
    background: #f5f5f5;
    border-radius: 4px;
    font-family: 'Monaco', 'Courier New', monospace;
    font-size: 0.8rem;
    white-space: pre-wrap;
    word-wrap: break-word;
    max-height: 400px;
    overflow-y: auto;
}

.wandb-table .full-text.show {
    display: block;
}
</style>
"""


def render_cot_sample_anthropic_style(
    question: str,
    prompt: str = "",
    cot: str = "",
    answer: str = "",
    training_type: str = "baseline",
    step: int = 0,
    collapsed_cot: bool = False
) -> str:
    """
    Render a single CoT sample in Anthropic's extended thinking style.
    
    Uses XML-like semantic sections with clear visual hierarchy:
    - QUESTION: The input problem
    - PROMPT: The full prompt sent to the model
    - THINKING: The chain of thought reasoning (collapsible for long content)
    - ANSWER: The final response
    """
    # Escape HTML in content
    def escape_html(text):
        if not text:
            return ""
        return (str(text)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;"))
    
    question_escaped = escape_html(question)
    prompt_escaped = escape_html(prompt)
    cot_escaped = escape_html(cot)
    answer_escaped = escape_html(answer)
    
    collapsed_class = "thinking-collapsed" if collapsed_cot and len(cot or "") > 500 else ""
    
    # Build HTML - question, prompt, thinking, answer
    html = f'''
    <div class="sample-card">
        <div class="sample-header">
            <span class="training-badge {training_type}">{training_type}</span>
            <span class="step-badge">Step {step}</span>
        </div>
        
        <div class="question-section">
            <h4>📝 Question</h4>
            <p>{question_escaped if question_escaped else "(No question)"}</p>
        </div>
        
        {f'<div class="prompt-section"><pre>{prompt_escaped}</pre></div>' if prompt_escaped else ''}
        
        <div class="thinking-section {collapsed_class}">
            <pre>{cot_escaped if cot_escaped else "(No reasoning captured)"}</pre>
        </div>
        
        <div class="answer-section">
            <pre>{answer_escaped if answer_escaped else "(No answer)"}</pre>
        </div>
    </div>
    '''
    
    return html


def render_anthropic_style_viewer(samples_df: pd.DataFrame) -> None:
    """
    Render CoT samples in an Anthropic-style interface with step-by-step navigation.
    Shows all steps with a slider to navigate and compare.
    Limited to 1 example/question for faster rendering.
    """
    if samples_df is None or samples_df.empty:
        st.info("No samples to display")
        return
    
    # Ensure required columns exist
    for col in ['step', 'question', 'prompt', 'cot', 'answer', 'training_type']:
        if col not in samples_df.columns:
            samples_df[col] = ""
    
    # Get all unique steps and training types
    all_steps = sorted(samples_df['step'].dropna().unique().tolist())
    all_training_types = sorted(samples_df['training_type'].dropna().unique().tolist())
    
    if not all_steps:
        st.warning("No steps found in the data")
        return
    
    # Group by training type and step
    # Create a structure: {training_type: {step: [samples]}}
    # Limit to 1 sample per step/training type to speed up rendering
    MAX_SAMPLES_PER_STEP = 1
    samples_by_tt_and_step = {}
    sample_indices_by_tt_and_step = {}
    for tt in all_training_types:
        samples_by_tt_and_step[tt] = {}
        sample_indices_by_tt_and_step[tt] = {}
        tt_data = samples_df[samples_df['training_type'] == tt]
        for step in all_steps:
            step_data = tt_data[tt_data['step'] == step]
            if not step_data.empty:
                # Limit to MAX_SAMPLES_PER_STEP samples for faster rendering
                step_samples = step_data.head(MAX_SAMPLES_PER_STEP).to_dict('records')
                samples_by_tt_and_step[tt][step] = step_samples
                sample_indices_by_tt_and_step[tt][step] = list(range(len(step_samples)))
            else:
                samples_by_tt_and_step[tt][step] = []
                sample_indices_by_tt_and_step[tt][step] = []
    
    # Use question_id for matching across training types
    # Tiered approach: prefer questions with ALL training types, fall back to partial coverage
    MAX_QUESTIONS = 5

    # Step 1: Collect all (normalized_qid, step) pairs with non-empty CoT per training type
    qid_steps_by_tt = {}  # {training_type: {normalized_qid: set of steps}}
    qid_to_original = {}  # {normalized_qid: (original_qid, question_text)}

    for tt in all_training_types:
        qid_steps_by_tt[tt] = {}
        for step in all_steps:
            for sample in samples_by_tt_and_step[tt].get(step, []):
                qid = sample.get('question_id', '')
                normalized_qid = normalize_question_id(qid)
                # Include samples even if CoT is empty - we want to see what the model generated
                if qid and not pd.isna(normalized_qid):
                    if normalized_qid not in qid_steps_by_tt[tt]:
                        qid_steps_by_tt[tt][normalized_qid] = set()
                    qid_steps_by_tt[tt][normalized_qid].add(step)
                    if normalized_qid not in qid_to_original:
                        qid_to_original[normalized_qid] = (qid, sample.get('question', f'Question {qid}'))

    # Step 2: Try to find questions with data in ALL training types (strict mode)
    question_info = {}
    use_strict_mode = False

    if all_training_types:
        common_qids = set(qid_steps_by_tt[all_training_types[0]].keys())
        for tt in all_training_types[1:]:
            common_qids &= set(qid_steps_by_tt[tt].keys())

        # Check if common questions have overlapping steps
        for normalized_qid in common_qids:
            steps_with_all_data = set(all_steps)
            for tt in all_training_types:
                steps_with_all_data &= qid_steps_by_tt[tt].get(normalized_qid, set())
            if steps_with_all_data:
                original_qid, question_text = qid_to_original[normalized_qid]
                question_info[original_qid] = question_text
                if len(question_info) >= MAX_QUESTIONS:
                    break

        if question_info:
            use_strict_mode = True

    # Step 3: Fallback - if no common questions, use questions from ANY training type
    if not question_info:
        use_strict_mode = False
        seen_normalized_qids = set()
        for tt in all_training_types:
            for step in all_steps:
                for sample in samples_by_tt_and_step[tt].get(step, []):
                    qid = sample.get('question_id', '')
                    normalized_qid = normalize_question_id(qid)
                    # Include samples even if CoT is empty
                    if qid and not pd.isna(normalized_qid) and normalized_qid not in seen_normalized_qids:
                        question_info[qid] = sample.get('question', f'Question {qid}')
                        seen_normalized_qids.add(normalized_qid)
                        if len(question_info) >= MAX_QUESTIONS:
                            break
                if len(question_info) >= MAX_QUESTIONS:
                    break
            if len(question_info) >= MAX_QUESTIONS:
                break

    if not question_info:
        st.warning("No samples available")
        return

    # Show mode indicator
    if not use_strict_mode:
        st.info("ℹ️ Showing questions with partial training type coverage. Some tabs may show 'No data'.")

    # Question selector
    question_ids = list(question_info.keys())
    question_labels = [f"Q{qid}: {question_info[qid][:50]}..." if len(question_info[qid]) > 50
                       else f"Q{qid}: {question_info[qid]}" for qid in question_ids]

    selected_idx = st.selectbox(
        "Select Question",
        options=range(len(question_ids)),
        format_func=lambda i: question_labels[i],
        key="question_selector"
    )
    selected_question_id = question_ids[selected_idx]
    selected_question_text = question_info[selected_question_id]

    # Filter samples by question_id (using numeric comparison for robustness)
    samples_by_tt_and_step_filtered = {}
    for tt in all_training_types:
        samples_by_tt_and_step_filtered[tt] = {}
        for step in all_steps:
            step_samples = samples_by_tt_and_step[tt].get(step, [])
            # Filter by question_id using numeric comparison (handles int/float/string formats)
            filtered_samples = [s for s in step_samples if question_ids_match(s.get('question_id', ''), selected_question_id)]
            samples_by_tt_and_step_filtered[tt][step] = filtered_samples

    # Display selected question
    st.markdown("### Question")
    st.markdown(
        f"""
        <div style="
            background: #f8f9fa;
            border-left: 4px solid #6c757d;
            padding: 16px;
            margin: 12px 0;
            border-radius: 6px;
            font-size: 1rem;
            line-height: 1.6;
        ">{selected_question_text or 'Sample question'}</div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Step navigation slider - only allow available steps (for the selected question)
    st.markdown("### Step Navigation")
    
    # Get only steps that have data (for the selected question)
    available_steps = []
    for step in all_steps:
        has_data = any(len(samples_by_tt_and_step_filtered[tt].get(step, [])) > 0 for tt in all_training_types)
        if has_data:
            available_steps.append(step)
    
    if not available_steps:
        st.warning("No steps with data available")
        return
    
    # Handle single step case to avoid RangeError
    if len(available_steps) == 1:
        selected_step = available_steps[0]
        current_step_idx = 0
        st.markdown(f"**Current Step: {selected_step}** (1 of 1)")
    else:
        # Convert steps to strings to ensure proper handling by select_slider
        # This prevents issues when all steps are the same numeric value (e.g., all 0)
        available_steps_str = [str(step) for step in available_steps]
        
        # Slider using available steps only
        selected_step_str = st.select_slider(
            "Select Step",
            options=available_steps_str,
            value=available_steps_str[0],
            key="step_slider"
        )
        
        # Convert back to original type for lookup
        selected_step = available_steps[available_steps_str.index(selected_step_str)]
        
        # Display current step clearly
        current_step_idx = available_steps_str.index(selected_step_str)
        st.markdown(f"**Current Step: {selected_step}** ({current_step_idx + 1} of {len(available_steps)})")
    
    # Show step indicators
    step_indicators = []
    for step in all_steps:
        has_data = any(len(samples_by_tt_and_step_filtered[tt].get(step, [])) > 0 for tt in all_training_types)
        if has_data:
            if step == selected_step:
                step_indicators.append(f"**{step}**")
            else:
                step_indicators.append(str(step))
        else:
            step_indicators.append(f"~~{step}~~")
    
    st.markdown(f"Available steps: {', '.join(step_indicators)}")
    st.markdown("---")
    
    # Always use tabs for training types comparison
    if len(all_training_types) == 1:
        tabs = [st.container()]
    else:
        tabs = st.tabs(all_training_types)

    for idx, tt in enumerate(all_training_types):
        with tabs[idx]:
            samples = samples_by_tt_and_step_filtered[tt].get(selected_step, [])

            if not samples:
                st.info(f"No data for step {selected_step}")
            else:
                # Only 1 sample per step/training type (limited for performance)
                sample = samples[0]

                # Display in Anthropic-style format
                prompt = str(sample.get('prompt', ''))
                cot = str(sample.get('cot', ''))
                answer = str(sample.get('answer', ''))

                # Prompt section
                with st.expander("📝 Prompt", expanded=False):
                    st.code(prompt, language=None)
                
                # Chain of Thought section
                st.markdown("**Chain of Thought**")
                st.markdown(
                    f"""
                    <div style="
                        background: #f6f8fa;
                        border-left: 4px solid #0969da;
                        padding: 16px;
                        margin: 12px 0;
                        border-radius: 6px;
                        font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
                        font-size: 0.9rem;
                        line-height: 1.6;
                        white-space: pre-wrap;
                        word-wrap: break-word;
                    ">{cot if cot else "(No reasoning captured)"}</div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Answer section
                st.markdown("**Answer**")
                st.markdown(
                    f"""
                    <div style="
                        background: #dafbe1;
                        border-left: 4px solid #1a7f37;
                        padding: 16px;
                        margin: 12px 0;
                        border-radius: 6px;
                        font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
                        font-size: 1rem;
                        font-weight: 500;
                    ">{answer if answer else "(No answer)"}</div>
                    """,
                    unsafe_allow_html=True
                )


def render_wandb_style_table(samples_df: pd.DataFrame) -> None:
    """
    Render CoT samples in a W&B-style table format with expandable rows.
    Each row shows truncated text with an expand option to see full content.
    """
    if samples_df is None or samples_df.empty:
        st.info("No samples to display")
        return
    
    # Ensure required columns exist
    for col in ['step', 'question', 'prompt', 'cot', 'answer', 'training_type']:
        if col not in samples_df.columns:
            samples_df[col] = ""
    
    # Helper function to truncate text
    def truncate_text(text, max_length=80):
        if not text or pd.isna(text):
            return ""
        text_str = str(text).strip()
        if len(text_str) <= max_length:
            return text_str
        return text_str[:max_length] + "..."
    
    # Prepare data for display table
    display_data = []
    for idx, row in samples_df.iterrows():
        step = int(row.get('step', 0))
        question = str(row.get('question', ''))
        prompt = str(row.get('prompt', ''))
        cot = str(row.get('cot', ''))
        answer = str(row.get('answer', ''))
        training_type = str(row.get('training_type', ''))
        
        display_data.append({
            'step': step,
            'question': truncate_text(question, 100),
            'prompt': truncate_text(prompt, 100),
            'cot': truncate_text(cot, 100),
            'answer': answer,
            'training_type': training_type,
            '_full_question': question,
            '_full_prompt': prompt,
            '_full_cot': cot,
            '_idx': idx
        })
    
    display_df = pd.DataFrame(display_data)
    
    # Display the table (similar to W&B's runs.history table)
    st.markdown("**Table: eval/sample_cots**")
    st.dataframe(
        display_df[['step', 'question', 'prompt', 'cot', 'answer']],
        use_container_width=True,
        height=400,
        hide_index=True,
        column_config={
            'step': st.column_config.NumberColumn('step', width='small'),
            'question': st.column_config.TextColumn('question', width='medium'),
            'prompt': st.column_config.TextColumn('prompt', width='medium'),
            'cot': st.column_config.TextColumn('cot', width='medium'),
            'answer': st.column_config.TextColumn('answer', width='small'),
        }
    )
    
    st.markdown("---")
    st.markdown("### Click to expand rows for full text:")
    
    # Display expandable rows for full content (like W&B's expand feature)
    for idx, row in samples_df.iterrows():
        step = int(row.get('step', 0))
        question = str(row.get('question', ''))
        prompt = str(row.get('prompt', ''))
        cot = str(row.get('cot', ''))
        answer = str(row.get('answer', ''))
        training_type = str(row.get('training_type', ''))
        
        # Create expander label with key info
        question_preview = truncate_text(question, 70)
        expander_label = f"Row {idx+1}: Step {step} | {question_preview}"
        
        with st.expander(expander_label, expanded=False):
            # Display in a clean format
            st.markdown(f"**Training Type:** `{training_type}` | **Step:** `{step}`")
            st.markdown("---")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Question:**")
                st.code(question, language=None)
                
                st.markdown("**Answer:**")
                st.code(answer, language=None)
            
            with col2:
                st.markdown("**Prompt:**")
                st.code(prompt, language=None)
            
            st.markdown("**Chain of Thought:**")
            st.code(cot, language=None)


def render_cot_comparison(samples: List[Dict]) -> str:
    """
    Render multiple CoT samples for comparison (e.g., across checkpoints).
    Shows question, prompt, thinking, and answer.
    """
    if not samples:
        return "<p>No samples to display</p>"
    
    html = '<div class="comparison-grid">'
    
    for sample in samples:
        html += render_cot_sample_anthropic_style(
            question=sample.get('question', ''),
            prompt=sample.get('prompt', ''),
            cot=sample.get('cot', ''),
            answer=sample.get('answer', ''),
            training_type=sample.get('training_type', 'baseline'),
            step=sample.get('step', 0),
            collapsed_cot=True
        )
    
    html += '</div>'
    return html


# =============================================================================
# Plotting Functions - Per Dataset Views with Standard Error Bars
# =============================================================================

def plot_accuracy_by_step_grouped(data: pd.DataFrame, dataset: str, 
                                  sample_size: int = DEFAULT_SAMPLE_SIZE) -> go.Figure:
    """
    Create grouped bar chart showing accuracy at each training step,
    with error bars using Binomial Standard Error.
    
    SE = sqrt(p * (1-p) / n)
    """
    dataset_data = data[data['dataset'] == dataset].copy()
    
    if dataset_data.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No data for dataset {dataset}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    steps = sorted(dataset_data['step'].unique())
    training_types = sorted(dataset_data['training_type'].unique())
    
    fig = go.Figure()
    
    for tt in training_types:
        tt_data = dataset_data[dataset_data['training_type'] == tt]
        
        accuracies = []
        error_bars = []
        
        for step in steps:
            step_data = tt_data[tt_data['step'] == step]
            if not step_data.empty:
                acc = step_data['accuracy'].mean()
                # Get sample size from data or use default
                n = step_data['num_samples'].iloc[0] if 'num_samples' in step_data.columns else sample_size
                n = int(n) if pd.notna(n) else sample_size
                
                if pd.notna(acc):
                    # Normalize to proportion if needed
                    acc_prop = acc if acc <= 1.0 else acc / 100
                    # Calculate Binomial SE
                    se = calculate_binomial_se(acc_prop, n)
                    
                    # Convert to percentage for display
                    acc_pct = acc_prop * 100
                    se_pct = se * 100 if pd.notna(se) else 0
                    
                    accuracies.append(acc_pct)
                    error_bars.append(se_pct)
                else:
                    accuracies.append(None)
                    error_bars.append(None)
            else:
                accuracies.append(None)
                error_bars.append(None)
        
        color = COLORS_BY_TRAINING_TYPE.get(tt, '#888888')
        pattern = BAR_PATTERNS.get(tt, "")
        
        fig.add_trace(go.Bar(
            name=tt,
            x=[str(int(s)) for s in steps],
            y=accuracies,
            error_y=dict(
                type='data',
                array=error_bars,
                visible=True,
                thickness=1.5,
                width=4
            ),
            marker_color=color,
            marker_pattern_shape=pattern,
            text=[f"{v:.0f}" if v else "" for v in accuracies],
            textposition='outside',
        ))
    
    fig.update_layout(
        title=f"Accuracy by Training Type - Dataset: {dataset.upper()}<br><sup>Error bars: Binomial SE = √(p(1-p)/n)</sup>",
        xaxis_title="Training Step",
        yaxis_title="Accuracy (%)",
        barmode='group',
        bargap=0.3,  # Gap between step groups
        bargroupgap=0,  # No gap between bars within a group (training types touch)
        xaxis=dict(
            type='category',  # Treat as categorical so bars align at exact steps
            tickmode='array',
            tickvals=[str(int(s)) for s in steps],
            ticktext=[str(int(s)) for s in steps],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        height=500
    )
    
    return fig


def plot_metric_by_dataset(data: pd.DataFrame, dataset: str, metric: str,
                          show_se: bool = True, 
                          sample_size: int = DEFAULT_SAMPLE_SIZE) -> go.Figure:
    """
    Create line plot for a specific metric and dataset,
    with error bars using appropriate Standard Error method.
    
    - Accuracy: Binomial SE = sqrt(p * (1-p) / n)
    - Other metrics: SEM = std / sqrt(n)
    """
    dataset_data = data[data['dataset'] == dataset].copy()
    
    if dataset_data.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No data for dataset {dataset}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure()
    
    mean_col = "accuracy" if metric == "accuracy" else f"{metric}_mean"
    std_col = "accuracy_std" if metric == "accuracy" else f"{metric}_std"
    
    training_types = sorted(dataset_data['training_type'].unique())
    
    for tt in training_types:
        tt_data = dataset_data[dataset_data['training_type'] == tt].sort_values('step')
        
        if mean_col not in tt_data.columns:
            continue
        
        y_values = tt_data[mean_col].dropna()
        if y_values.empty:
            continue
        
        # Get sample sizes
        if 'num_samples' in tt_data.columns:
            n_values = tt_data.loc[y_values.index, 'num_samples'].fillna(sample_size).astype(int)
        else:
            n_values = pd.Series([sample_size] * len(y_values), index=y_values.index)
        
        # Calculate error bars
        if metric == "accuracy":
            # Binomial SE for accuracy
            y_prop = y_values.copy()
            if y_prop.max() > 1.0:
                y_prop = y_prop / 100
            se_values = pd.Series([
                calculate_binomial_se(p, n) for p, n in zip(y_prop, n_values)
            ], index=y_values.index)
            
            # Convert to percentage
            if y_values.max() <= 1.0:
                y_values = y_values * 100
                se_values = se_values * 100
            else:
                se_values = se_values * 100
        else:
            # SEM for other metrics
            if std_col in tt_data.columns:
                std_values = tt_data.loc[y_values.index, std_col]
                se_values = pd.Series([
                    calculate_sem(std, n) for std, n in zip(std_values, n_values)
                ], index=y_values.index)
            else:
                se_values = pd.Series([np.nan] * len(y_values), index=y_values.index)
        
        color = COLORS_BY_TRAINING_TYPE.get(tt, '#888888')
        line_style = LINE_STYLES.get(tt, 'solid')
        marker_symbol = MARKER_SYMBOLS.get(tt, 'circle')
        
        trace_kwargs = {
            "x": tt_data.loc[y_values.index, 'step'],
            "y": y_values,
            "mode": 'lines+markers',
            "name": tt,
            "line": dict(color=color, width=2, dash=line_style),
            "marker": dict(size=10, symbol=marker_symbol),
        }
        
        if show_se and not se_values.isna().all():
            trace_kwargs["error_y"] = dict(
                type='data',
                array=se_values,
                visible=True,
                thickness=1.5,
                width=4
            )
        
        fig.add_trace(go.Scatter(**trace_kwargs))
    
    y_title = f"{metric.title()} (%)" if metric == "accuracy" else metric.title()
    se_method = "Binomial SE" if metric == "accuracy" else "SEM"
    
    fig.update_layout(
        title=f"{metric.title()} by Training Type - Dataset: {dataset.upper()}" + 
              (f"<br><sup>Error bars: {se_method}</sup>" if show_se else ""),
        xaxis_title="Training Step",
        yaxis_title=y_title,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        height=500
    )
    
    return fig


def plot_cohens_d_by_dataset(data: pd.DataFrame, dataset: str, 
                            metrics: List[str] = None,
                            baseline_step: int = 0) -> go.Figure:
    """
    Create Cohen's d plot for a specific dataset showing all metrics
    as subplots, with different training types as different line styles.
    """
    if metrics is None:
        metrics = ORIGINAL_METRICS
    
    dataset_data = data[data['dataset'] == dataset].copy()
    
    if dataset_data.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No data for dataset {dataset}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    n_metrics = len(metrics)
    fig = make_subplots(
        rows=1, cols=n_metrics,
        subplot_titles=[m.title() for m in metrics],
        horizontal_spacing=0.1
    )
    
    training_types = sorted(dataset_data['training_type'].unique())
    
    for tt in training_types:
        tt_data = dataset_data[dataset_data['training_type'] == tt].copy()
        
        baseline = tt_data[tt_data['step'] == baseline_step]
        if baseline.empty:
            baseline = tt_data.iloc[[0]]
        baseline = baseline.iloc[0]
        
        color = COLORS_BY_TRAINING_TYPE.get(tt, '#888888')
        line_style = LINE_STYLES.get(tt, 'solid')
        marker_symbol = MARKER_SYMBOLS.get(tt, 'circle')
        
        for col_idx, metric in enumerate(metrics, 1):
            mean_col = "accuracy" if metric == "accuracy" else f"{metric}_mean"
            std_col = "accuracy_std" if metric == "accuracy" else f"{metric}_std"
            
            if mean_col not in tt_data.columns:
                continue
            
            baseline_mean = baseline.get(mean_col)
            baseline_std = baseline.get(std_col, 0.1)
            
            # Convert to numeric
            baseline_mean = pd.to_numeric(baseline_mean, errors='coerce')
            baseline_std = pd.to_numeric(baseline_std, errors='coerce')
            if pd.isna(baseline_std) or baseline_std == 0:
                baseline_std = 0.1
            
            if pd.isna(baseline_mean):
                continue
            
            steps = []
            cohens_d_values = []
            
            for _, row in tt_data.sort_values('step').iterrows():
                current_mean = row.get(mean_col)
                current_std = row.get(std_col, 0.1)
                
                # Convert to numeric
                current_mean = pd.to_numeric(current_mean, errors='coerce')
                current_std = pd.to_numeric(current_std, errors='coerce')
                if pd.isna(current_std) or current_std == 0:
                    current_std = 0.1
                
                if pd.notna(current_mean):
                    d = calculate_cohens_d(baseline_mean, baseline_std, current_mean, current_std)
                    steps.append(row['step'])
                    cohens_d_values.append(d)
            
            if not steps:
                continue
            
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=cohens_d_values,
                    mode='lines+markers',
                    name=tt if col_idx == 1 else None,
                    line=dict(color=color, width=2, dash=line_style),
                    marker=dict(size=10, symbol=marker_symbol),
                    showlegend=(col_idx == 1),
                    legendgroup=tt,
                ),
                row=1, col=col_idx
            )
    
    # Add reference lines and bands
    for col_idx in range(1, n_metrics + 1):
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                     line_width=1, row=1, col=col_idx)
        fig.add_hrect(y0=-0.2, y1=0.2, fillcolor="lightgreen", opacity=0.3, 
                     line_width=0, row=1, col=col_idx)
    
    fig.update_layout(
        title=f"Cohen's d by Metric - Dataset: {dataset.upper()}",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="left",
            x=0
        )
    )
    
    # Set y-axis title for all subplots
    fig.update_yaxes(title_text="Cohen's d", row=1, col=1)
    
    for col_idx in range(1, n_metrics + 1):
        fig.update_xaxes(title_text="Training Step", row=1, col=col_idx)
    
    return fig


def create_summary_table(data: pd.DataFrame, metrics: List[str], 
                        sample_size: int = DEFAULT_SAMPLE_SIZE) -> pd.DataFrame:
    """Create summary table with latest metrics and Standard Errors for each run."""
    summary = []
    
    for run_name in data['run_name'].unique():
        run_data = data[data['run_name'] == run_name].sort_values('step')
        latest = run_data.iloc[-1]
        
        # Get sample size
        n = latest.get('num_samples', sample_size)
        n = int(n) if pd.notna(n) else sample_size
        
        row = {
            'Run': run_name,
            'Training Type': run_data['training_type'].iloc[0],
            'Dataset': run_data['dataset'].iloc[0],
            'Model': run_data.get('model', pd.Series(['unknown'])).iloc[0],
            'Learning Rate': run_data.get('learning_rate', pd.Series(['unknown'])).iloc[0],
            'Step': int(latest['step']),
            'N': n,
        }
        
        for metric in metrics:
            mean_col = "accuracy" if metric == "accuracy" else f"{metric}_mean"
            std_col = "accuracy_std" if metric == "accuracy" else f"{metric}_std"
            
            if mean_col in latest:
                val = latest[mean_col]
                std_val = latest.get(std_col, np.nan)
                
                if pd.notna(val):
                    if metric == "accuracy":
                        # Convert and calculate Binomial SE
                        val_prop = val if val <= 1.0 else val / 100
                        se = calculate_binomial_se(val_prop, n) * 100
                        val_pct = val_prop * 100
                        row[metric.title()] = f"{val_pct:.1f} ± {se:.1f}" if pd.notna(se) else f"{val_pct:.1f}"
                    else:
                        # Calculate SEM for other metrics
                        se = calculate_sem(std_val, n) if pd.notna(std_val) else np.nan
                        row[metric.title()] = f"{val:.3f} ± {se:.3f}" if pd.notna(se) else f"{val:.3f}"
                else:
                    row[metric.title()] = "N/A"
            else:
                row[metric.title()] = "N/A"
        
        summary.append(row)
    
    return pd.DataFrame(summary)


# =============================================================================
# Main App
# =============================================================================

def main():
    st.title("📊 CoT Health Metrics Dashboard")
    st.markdown("Monitor **Accuracy**, **Original Metrics**, and **Cohen's d** across training runs")
    
    # ==========================================================================
    # Sidebar Configuration
    # ==========================================================================
    
    st.sidebar.header("⚙️ Configuration")
    
    data_source = st.sidebar.radio(
        "Data Source",
        options=["W&B", "Local Files"],
        help="Choose where to load metrics from"
    )
    
    data = None
    
    if data_source == "W&B":
        st.sidebar.subheader("W&B Settings")
        # Get entity from secrets (cloud) or environment (local) - use safe getter
        default_entity = get_secret("WANDB_ENTITY", os.environ.get("WANDB_ENTITY", "mliu7"))
        wandb_entity = st.sidebar.text_input(
            "W&B Entity", 
            value=default_entity
        )
        
        # Run state filter
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Run State Filter")
        run_state_options = st.sidebar.multiselect(
            "Run States",
            options=["running", "finished", "crashed", "failed", "killed"],
            default=["running", "finished"],
            help="Select which run states to include. 'running' shows active training, 'finished' shows completed runs."
        )
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Project Builder")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            selected_training_types = st.multiselect(
                "Training Types",
                options=DEFAULT_TRAINING_TYPES,
                default=DEFAULT_TRAINING_TYPES,
                key="train_type_select"
            )
        with col2:
            selected_models = st.multiselect(
                "Models",
                options=DEFAULT_MODEL_NAMES,
                default=["Qwen3-4B"],
                key="model_select"
            )
        
        selected_datasets = st.sidebar.multiselect(
            "Datasets",
            options=DEFAULT_DATASET_NAMES,
            default=DEFAULT_DATASET_NAMES,
            key="dataset_select"
        )
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Per-Dataset Learning Rate Filter")
        enable_lr_filter = st.sidebar.checkbox(
            "Enable LR filtering",
            value=True,
            help="If enabled, only show runs matching the selected learning rates. If disabled, show all runs regardless of LR."
        )
        
        dataset_lr_map = {}
        if enable_lr_filter:
            st.sidebar.markdown("*Select learning rate for each dataset*")
            
            # Default learning rates per dataset: BA uses 1e-5, CA and SB use 5e-5
            DEFAULT_LR_PER_DATASET = {
                "ba": "1e-5",
                "ca": "5e-5",
                "sb": "5e-5",
            }
            
            for ds in selected_datasets:
                # Get default LR for this dataset, fallback to 5e-5
                default_lr = DEFAULT_LR_PER_DATASET.get(ds, "5e-5")
                default_index = DEFAULT_LEARNING_RATES.index(default_lr) if default_lr in DEFAULT_LEARNING_RATES else 0
                lr = st.sidebar.selectbox(
                    f"LR for {ds.upper()}",
                    options=DEFAULT_LEARNING_RATES,
                    index=default_index,
                    key=f"lr_{ds}"
                )
                dataset_lr_map[ds] = lr
        
        generated_projects = []
        for tt in selected_training_types:
            for model in selected_models:
                for ds in selected_datasets:
                    generated_projects.append(generate_project_name(tt, model, ds))
        
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Generated Projects ({len(generated_projects)}):**")
        with st.sidebar.expander("View/Edit Projects"):
            projects_text = st.text_area(
                "Edit project names (one per line)",
                value="\n".join(generated_projects),
                height=200
            )
            projects_to_load = [p.strip() for p in projects_text.split("\n") if p.strip()]
        
        if st.sidebar.button("🔄 Refresh from W&B"):
            st.cache_data.clear()
            st.rerun()
        
        if generated_projects or projects_to_load:
            final_projects = projects_to_load if projects_to_load else generated_projects
            # Store projects in session state for use in other tabs
            st.session_state['wandb_projects'] = final_projects
            st.session_state['wandb_entity'] = wandb_entity
            selected_states = run_state_options if run_state_options else ["running", "finished"]
            with st.spinner(f"Loading from {len(final_projects)} W&B projects (states: {', '.join(selected_states)})..."):
                data = load_from_wandb_multi(wandb_entity, final_projects, run_states=selected_states)
            
            if data is not None and not data.empty:
                # Show run state distribution if available
                if 'run_state' in data.columns:
                    state_counts = data['run_state'].value_counts().to_dict()
                    state_info = ", ".join([f"{k}: {v}" for k, v in state_counts.items()])
                    st.sidebar.success(f"Loaded {len(data)} data points\n({state_info})")
                else:
                    st.sidebar.success(f"Loaded {len(data)} data points")
                
                # Apply LR filtering only if enabled and dataset_lr_map is provided
                if enable_lr_filter and dataset_lr_map:
                    filtered_rows = []
                    for ds, lr in dataset_lr_map.items():
                        ds_data = data[(data['dataset'] == ds) & (data['learning_rate'] == lr)]
                        if not ds_data.empty:
                            filtered_rows.append(ds_data)
                    if filtered_rows:
                        data = pd.concat(filtered_rows, ignore_index=True)
                    # If filtering resulted in empty data, keep original data
                    # This allows user to see all data if LR filter matches nothing
    
    else:  # Local Files
        st.sidebar.subheader("Local Settings")
        default_output = str(Path(__file__).parent.parent / "output")
        output_dir = st.sidebar.text_input("Output Directory", value=default_output)
        
        if st.sidebar.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()
        
        with st.spinner("Loading local metrics..."):
            data = load_from_local(output_dir)
        
        if data is not None:
            st.sidebar.success(f"Loaded {len(data)} data points from local files")
    
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (60s)", value=False)
    
    # ==========================================================================
    # Check if data loaded
    # ==========================================================================
    
    if data is None or data.empty:
        st.warning("⚠️ No metrics data found.")
        st.info("""
        **To see data:**
        
        1. **For W&B**: 
           - Select training types, models, and datasets
           - Set learning rate for each dataset
           - Ensure runs are logging metrics
           
        2. **For Local Files**: Make sure training runs have `metrics_history.json` files
        
        **W&B Naming Convention:**
        - Projects: `{training_type}-lr-sweep-{model_name}-{dataset_name}`
        - Runs: `{training_type}_{dataset_name}_lr{learning_rate}`
        """)
        return
    
    # ==========================================================================
    # Data Filters
    # ==========================================================================
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Additional Filters")
    
    all_datasets = sorted(data['dataset'].unique().tolist())
    all_training_types = sorted(data['training_type'].unique().tolist())
    
    filter_datasets = st.sidebar.multiselect(
        "Filter Datasets",
        options=all_datasets,
        default=all_datasets,
        key="filter_datasets"
    )
    
    filter_training_types = st.sidebar.multiselect(
        "Filter Training Types",
        options=all_training_types,
        default=all_training_types,
        key="filter_training_types"
    )
    
    filtered_data = data[
        (data['dataset'].isin(filter_datasets)) &
        (data['training_type'].isin(filter_training_types))
    ]
    
    if filtered_data.empty:
        st.warning("No data matches the selected filters.")
        return
    
    # Plot settings
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Plot Settings")
    show_se = st.sidebar.checkbox("Show error bars (Standard Error)", value=True)
    sample_size = st.sidebar.number_input(
        "Sample size (n) for SE calculation",
        min_value=10,
        max_value=10000,
        value=DEFAULT_SAMPLE_SIZE,
        help="Number of samples used to calculate Standard Error"
    )
    
    # ==========================================================================
    # Debug Section (expandable)
    # ==========================================================================
    
    with st.expander("🔍 Debug: W&B Data Loading Info"):
        if 'wandb_debug_info' in st.session_state:
            debug_df = pd.DataFrame(st.session_state['wandb_debug_info'])
            st.dataframe(debug_df, use_container_width=True)
            
            # Show data info before final filtering
            st.markdown("**Data loaded from W&B:**")
            if data is not None and not data.empty:
                st.write(f"Total rows loaded: {len(data)}")
                if 'run_name' in data.columns:
                    st.write(f"Unique runs: {data['run_name'].nunique()}")
                if 'dataset' in data.columns:
                    st.write(f"Unique datasets: {sorted(data['dataset'].unique().tolist())}")
                if 'learning_rate' in data.columns:
                    st.write(f"Unique LRs: {sorted(data['learning_rate'].unique().tolist())}")
                if 'training_type' in data.columns:
                    st.write(f"Unique training types: {sorted(data['training_type'].unique().tolist())}")
            
            st.markdown("---")
            st.markdown("**After filtering (training type, dataset, LR):**")
            if 'filtered_data' in locals() and filtered_data is not None and not filtered_data.empty:
                st.write(f"Total rows in filtered_data: {len(filtered_data)}")
                st.write(f"Unique runs: {filtered_data['run_name'].nunique()}")
                st.write(f"Unique datasets: {filtered_data['dataset'].unique().tolist()}")
                st.write(f"Unique LRs: {filtered_data['learning_rate'].unique().tolist()}")
                st.write(f"Unique training types: {filtered_data['training_type'].unique().tolist()}")
                
                # Show data counts per dataset
                st.markdown("**Data points per dataset:**")
                for ds in filtered_data['dataset'].unique():
                    ds_data = filtered_data[filtered_data['dataset'] == ds]
                    st.write(f"  {ds}: {len(ds_data)} rows, steps: {sorted(ds_data['step'].unique().tolist())}")
            else:
                st.write("No data after filtering (may have been filtered out)")
        else:
            st.info("No debug info available. Click 'Refresh from W&B' to load data.")
    
    # ==========================================================================
    # Main Content Tabs - Reordered as requested
    # ==========================================================================
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Per-Dataset Accuracy", 
        "🔬 Per-Dataset Metrics",
        "📊 Per-Dataset Cohen's d",
        "📋 Summary Table",
        "🧠 Reasoning Examples"
    ])
    
    # --------------------------------------------------------------------------
    # Tab 1: Per-Dataset Accuracy (Grouped Bar Chart with Binomial SE)
    # --------------------------------------------------------------------------
    with tab1:
        st.header("Accuracy by Training Type - Per Dataset")
        
        # Methodology explanation
        st.markdown("""
        **Error Bar Methodology:** Binomial Standard Error
        
        For accuracy metrics (which follow a binomial distribution):
        
        $$SE = \\sqrt{\\frac{p(1-p)}{n}}$$
        
        where *p* is the accuracy proportion and *n* is the sample size.
        """)
        
        datasets_in_data = sorted(filtered_data['dataset'].unique())
        
        if not datasets_in_data:
            st.warning("No datasets found in filtered data")
        else:
            cols = st.columns(min(len(datasets_in_data), 2))
            
            for idx, dataset in enumerate(datasets_in_data):
                with cols[idx % 2]:
                    fig = plot_accuracy_by_step_grouped(filtered_data, dataset, sample_size)
                    st.plotly_chart(fig, use_container_width=True)
    
    # --------------------------------------------------------------------------
    # Tab 2: Per-Dataset Original Metrics (with SEM error bars)
    # --------------------------------------------------------------------------
    with tab2:
        st.header("Original Metrics by Training Type - Per Dataset")
        
        st.markdown("""
        **Metrics Overview:**
        - **Necessity**: How much the CoT is needed for correct answers
        - **Substantivity**: How much the CoT content matters (vs filler)
        - **Paraphrasability**: Whether semantically equivalent CoTs produce same answers

        **Error Bar Methodology:**
        - Accuracy: Binomial SE = √(p(1-p)/n)
        - Other metrics: Standard Error of Mean (SEM) = σ/√n
        """)

        # Expected metric values table
        with st.expander("📋 Expected Metric Values by Pathology Type", expanded=False):
            expected_metrics_html = """
            <style>
                .expected-table {
                    border-collapse: collapse;
                    margin: 15px 0;
                    font-size: 14px;
                    min-width: 500px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    border-radius: 8px;
                    overflow: hidden;
                }
                .expected-table thead tr {
                    background-color: #5a7d9a;
                    color: white;
                    text-align: center;
                }
                .expected-table th, .expected-table td {
                    padding: 12px 15px;
                    border-bottom: 1px solid #ddd;
                    text-align: center;
                }
                .expected-table tbody tr {
                    background-color: #f9f9f9;
                }
                .expected-table tbody tr:nth-of-type(even) {
                    background-color: #f3f3f3;
                }
                .expected-table tbody tr:hover {
                    background-color: #e8f4f8;
                }
                .expected-table .positive { color: #2ca02c; font-weight: bold; }
                .expected-table .negative { color: #d62728; font-weight: bold; }
                .expected-table .zero { color: #7f7f7f; font-weight: bold; }
            </style>
            <table class="expected-table">
                <thead>
                    <tr>
                        <th>Training Type</th>
                        <th>Necessity</th>
                        <th>Paraphrasability</th>
                        <th>Substantivity</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Healthy (baseline)</strong></td>
                        <td class="positive">+ve</td>
                        <td class="zero">≈ 0</td>
                        <td class="positive">+ve</td>
                    </tr>
                    <tr>
                        <td><strong>Encoded</strong></td>
                        <td class="positive">+ve</td>
                        <td class="negative">−ve</td>
                        <td class="positive">+ve</td>
                    </tr>
                    <tr>
                        <td><strong>Internalized</strong></td>
                        <td class="positive">+ve</td>
                        <td class="zero">≈ 0</td>
                        <td class="zero">≈ 0</td>
                    </tr>
                    <tr>
                        <td><strong>Post-hoc</strong></td>
                        <td class="zero">≈ 0</td>
                        <td class="zero">≈ 0</td>
                        <td class="zero">≈ 0</td>
                    </tr>
                </tbody>
            </table>
            """
            st.markdown(expected_metrics_html, unsafe_allow_html=True)
            st.markdown("""
            **Interpretation:**
            - **+ve**: Positive score expected (CoT characteristic is present)
            - **−ve**: Negative score expected (intervention improves performance)
            - **≈ 0**: Near-zero score expected (no significant difference)
            """)

        metric_to_plot = st.selectbox(
            "Select Metric",
            options=ALL_METRICS,
            index=0,
            key="metric_select"
        )
        
        datasets_in_data = sorted(filtered_data['dataset'].unique())
        
        cols = st.columns(min(len(datasets_in_data), 2))
        
        for idx, dataset in enumerate(datasets_in_data):
            with cols[idx % 2]:
                fig = plot_metric_by_dataset(
                    filtered_data, dataset, metric_to_plot, 
                    show_se=show_se, sample_size=sample_size
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # --------------------------------------------------------------------------
    # Tab 3: Per-Dataset Cohen's d
    # --------------------------------------------------------------------------
    with tab3:
        st.header("Cohen's d by Metric - Per Dataset")
        
        st.markdown("**Cohen's d** measures the effect size between baseline (step 0) and trained checkpoints.")

        # Display both tables side by side
        table_col1, table_col2 = st.columns(2)

        with table_col1:
            st.markdown("**Effect Size Interpretation**")
            # Styled HTML table for Cohen's d interpretation
            cohens_d_table_html = """
            <style>
                .cohens-table {
                    border-collapse: collapse;
                    margin: 10px 0;
                    font-size: 14px;
                    width: 100%;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    border-radius: 8px;
                    overflow: hidden;
                }
                .cohens-table thead tr {
                    background-color: #4a90a4;
                    color: white;
                    text-align: left;
                }
                .cohens-table th, .cohens-table td {
                    padding: 10px 12px;
                    border-bottom: 1px solid #ddd;
                }
                .cohens-table tbody tr {
                    background-color: #f9f9f9;
                }
                .cohens-table tbody tr:nth-of-type(even) {
                    background-color: #f3f3f3;
                }
                .cohens-table tbody tr:hover {
                    background-color: #e8f4f8;
                }
                .cohens-table .negligible { color: #2ca02c; font-weight: bold; }
                .cohens-table .small { color: #1f77b4; font-weight: bold; }
                .cohens-table .medium { color: #ff7f0e; font-weight: bold; }
                .cohens-table .large { color: #d62728; font-weight: bold; }
            </style>
            <table class="cohens-table">
                <thead>
                    <tr>
                        <th>Effect Size</th>
                        <th>|d| Range</th>
                        <th>Interpretation</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="negligible">Negligible</td>
                        <td>&lt; 0.2</td>
                        <td>Similar to baseline</td>
                    </tr>
                    <tr>
                        <td class="small">Small</td>
                        <td>0.2 – 0.5</td>
                        <td>Minor change</td>
                    </tr>
                    <tr>
                        <td class="medium">Medium</td>
                        <td>0.5 – 0.8</td>
                        <td>Moderate change</td>
                    </tr>
                    <tr>
                        <td class="large">Large</td>
                        <td>≥ 0.8</td>
                        <td>Significant change</td>
                    </tr>
                </tbody>
            </table>
            """
            st.markdown(cohens_d_table_html, unsafe_allow_html=True)

        with table_col2:
            st.markdown("**Expected Cohen's d (Healthy − Pathological)**")
            # Expected Cohen's d values table
            expected_cohens_d_html = """
            <style>
                .expected-cohens-table {
                    border-collapse: collapse;
                    margin: 10px 0;
                    font-size: 14px;
                    width: 100%;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    border-radius: 8px;
                    overflow: hidden;
                }
                .expected-cohens-table thead tr {
                    background-color: #4a90a4;
                    color: white;
                    text-align: center;
                }
                .expected-cohens-table th, .expected-cohens-table td {
                    padding: 10px 12px;
                    border-bottom: 1px solid #ddd;
                    text-align: center;
                }
                .expected-cohens-table tbody tr {
                    background-color: #f9f9f9;
                }
                .expected-cohens-table tbody tr:nth-of-type(even) {
                    background-color: #f3f3f3;
                }
                .expected-cohens-table tbody tr:hover {
                    background-color: #e8f4f8;
                }
                .expected-cohens-table .positive { color: #2ca02c; font-weight: bold; }
                .expected-cohens-table .zero { color: #7f7f7f; font-weight: bold; }
            </style>
            <table class="expected-cohens-table">
                <thead>
                    <tr>
                        <th>Pathology</th>
                        <th>Necessity</th>
                        <th>Paraphrasability</th>
                        <th>Substantivity</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Encoded</strong></td>
                        <td class="zero">≈ 0</td>
                        <td class="positive">+ve</td>
                        <td class="zero">≈ 0</td>
                    </tr>
                    <tr>
                        <td><strong>Internalized</strong></td>
                        <td class="zero">≈ 0</td>
                        <td class="zero">≈ 0</td>
                        <td class="positive">+ve</td>
                    </tr>
                    <tr>
                        <td><strong>Post-hoc</strong></td>
                        <td class="positive">+ve</td>
                        <td class="zero">≈ 0</td>
                        <td class="positive">+ve</td>
                    </tr>
                </tbody>
            </table>
            """
            st.markdown(expected_cohens_d_html, unsafe_allow_html=True)
            st.caption("**+ve**: detectable difference | **≈ 0**: no significant difference")

        st.markdown("*Green band on plots = negligible effect (similar to baseline)*")

        metrics_for_cohens = st.multiselect(
            "Metrics for Cohen's d",
            options=ALL_METRICS,
            default=ORIGINAL_METRICS,
            key="cohens_metrics"
        )
        
        baseline_step = st.number_input("Baseline Step", value=0, min_value=0)
        
        datasets_in_data = sorted(filtered_data['dataset'].unique())
        
        if metrics_for_cohens:
            for dataset in datasets_in_data:
                fig = plot_cohens_d_by_dataset(
                    filtered_data, dataset, 
                    metrics=metrics_for_cohens,
                    baseline_step=baseline_step
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
    
    # --------------------------------------------------------------------------
    # Tab 4: Summary Table
    # --------------------------------------------------------------------------
    with tab4:
        st.header("Summary Table")
        
        st.markdown("""
        **Error bars in table:**
        - Accuracy: ± Binomial SE
        - Other metrics: ± SEM (Standard Error of Mean)
        """)
        
        summary_metrics = st.multiselect(
            "Metrics to Include",
            options=ALL_METRICS,
            default=ALL_METRICS,
            key="summary_metrics"
        )
        
        summary_df = create_summary_table(filtered_data, summary_metrics, sample_size)
        
        if not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True)
            
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Summary CSV",
                data=csv,
                file_name=f"metrics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No data available for summary table")
    
    # --------------------------------------------------------------------------
    # Tab 5: Reasoning Examples (Anthropic-style CoT Visualization)
    # --------------------------------------------------------------------------
    with tab5:
        st.header("🧠 Reasoning Examples Across Checkpoints")
        
        # Inject Anthropic-style CSS
        st.markdown(ANTHROPIC_COT_CSS, unsafe_allow_html=True)
        
        st.markdown("""
        **Visualize how model reasoning evolves during training.**
        
        Navigate through training steps to see how the model's chain-of-thought reasoning and answers change over time.
        Use the step slider to explore different checkpoints and compare reasoning across training types.
        """)
        
        # Show training type prompts table
        with st.expander("📋 Training Types & Prompt Configuration", expanded=False):
            st.markdown("""
            ### Training Types Overview
            
            Each training type modifies how the model is trained and prompted:
            """)
            
            prompts_data = {
                "Training Type": ["baseline", "internalized", "encoded", "post-hoc"],
                "Custom Instruction": [
                    "Let's think step by step.",
                    "Model uses filler-replaced CoT during training",
                    "Model uses codebook-encoded CoT + system prompt",
                    "You already KNOW the CORRECT answer, which is {answer}, but you need to write your reasoning steps."
                ],
                "Training Data": [
                    "Standard (question, CoT, answer) pairs",
                    "CoT swapped with filler (shuffled/not_relevant)",
                    "CoT encoded via domain-specific codebook",
                    "Same as baseline (answer provided at inference)"
                ]
            }
            st.table(pd.DataFrame(prompts_data))
            
            st.markdown("""
            ---
            ### Internalized Filler Types
            
            The **internalized** training replaces the original CoT with "filler" reasoning:
            
            | Filler Type | Description |
            |-------------|-------------|
            | `shuffled` | CoT from a **different question** in the same dataset |
            | `not_relevant` | CoT from a **completely different task** (e.g., calendar→spell_backward) |
            | `lorem_ipsum` | Latin placeholder text |
            | `dots` | Just periods (".....") |
            | `think_token` | Repeated `<think>` tokens |
            
            **Default:** `not_relevant` (swaps reasoning from a different domain)
            
            ---
            ### Encoded Codebook System Prompts
            
            The **encoded** training uses domain-specific codebooks that replace numbers and terms:
            
            | Dataset | Theme | Example Mappings |
            |---------|-------|------------------|
            | `calendar_arithmetic` | Astronomy | 0→eclipse, 1→sun, Monday→sol-day |
            | `binary_alternation` | Music | 0→rest, 1→beat, swap→transpose |
            | `spell_backward` | Animals | 0→owl, 1→cat, letter→glyph |
            | `largest_island` | Nautical | 0→minnow, 1→shark, grid→chart |
            
            The codebook system prompt is **prepended** to guide the model to use the coded vocabulary.
            """)
        
        # Load sample CoTs data
        sample_cots_data = None
        
        if data_source == "W&B":
            # Try to load from W&B using projects from session state
            projects_for_cots = st.session_state.get('wandb_projects', [])
            wandb_entity_for_cots = st.session_state.get('wandb_entity', wandb_entity if 'wandb_entity' in locals() else "mliu7")
            
            if projects_for_cots:
                try:
                    with st.spinner("Loading reasoning examples from W&B..."):
                        sample_cots_data = load_sample_cots_from_wandb(wandb_entity_for_cots, projects_for_cots)
                except Exception as e:
                    st.warning(f"Error loading from W&B: {e}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
            else:
                st.info("Please load metrics data first (click 'Refresh from W&B' in the sidebar) to load reasoning examples.")
        else:
            # Load from local files
            try:
                with st.spinner("Loading reasoning examples from local files..."):
                    sample_cots_data = load_sample_cots_from_local(output_dir)
            except Exception as e:
                st.warning(f"Error loading local files: {e}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
        
        # Check if we have data
        if sample_cots_data is None or sample_cots_data.empty:
            st.warning("**No reasoning examples found.**")
            
            st.info("""
            Reasoning examples are captured during checkpoint evaluation and logged as `eval/sample_cots` tables in W&B.
            
            **Possible reasons:**
            1. Training runs haven't completed checkpoint evaluation yet
            2. The W&B tables may not be accessible via the API (check W&B dashboard directly)
            3. The `sample_cots` key wasn't logged during training
            """)
            
            # Show debug info if available
            with st.expander("🔍 Debug: W&B Loading Details"):
                if 'cot_debug_info' in st.session_state:
                    st.json(st.session_state['cot_debug_info'])
                else:
                    st.write("No debug info available. Try refreshing.")
                
                st.markdown("**Check your W&B runs:**")
                if data_source == "W&B":
                    projects_for_links = st.session_state.get('wandb_projects', [])
                    wandb_entity_for_links = st.session_state.get('wandb_entity', wandb_entity if 'wandb_entity' in locals() else "mliu7")
                    for proj in projects_for_links:
                        st.markdown(f"- [{proj}](https://wandb.ai/{wandb_entity_for_links}/{proj})")
            
        else:
            # We have real data!
            # Check if any invalid samples were filtered
            attrs = getattr(sample_cots_data, 'attrs', {})
            filtered_count = attrs.get('filtered_invalid_count', 0)
            sampling_note = attrs.get('sampling_note', None)
            
            if filtered_count > 0 or sampling_note:
                messages = []
                if filtered_count > 0:
                    messages.append(f"Filtered out {filtered_count} invalid samples (empty CoT or answer)")
                if sampling_note:
                    messages.append(sampling_note)
                
                st.success(f"Loaded {len(sample_cots_data)} reasoning examples")
                if messages:
                    st.info("ℹ️ " + " | ".join(messages))
            else:
                st.success(f"Loaded {len(sample_cots_data)} reasoning examples")
            
            # Filters
            col1, col2 = st.columns(2)
            
            with col1:
                available_datasets = sorted(sample_cots_data['dataset'].dropna().unique().tolist())
                selected_dataset_cot = st.selectbox(
                    "Dataset",
                    options=available_datasets if available_datasets else ["all"],
                    key="cot_dataset"
                )
            
            with col2:
                available_training_types = sorted(sample_cots_data['training_type'].dropna().unique().tolist())
                selected_tt_cot = st.multiselect(
                    "Training Types",
                    options=available_training_types,
                    default=available_training_types,
                    key="cot_training_types"
                )

            # Additional filtering options
            col3, col4 = st.columns(2)

            with col3:
                require_shared = st.checkbox(
                    "Show only shared questions",
                    value=False,
                    help="When enabled, only show questions that appear in ALL selected training types at each step. "
                         "This allows direct comparison of how different training types respond to the same question."
                )

            with col4:
                samples_per_step = st.number_input(
                    "Samples per step (0 = all)",
                    min_value=0,
                    max_value=100,
                    value=0,
                    help="Limit number of samples shown per step per training type. Set to 0 to show all."
                )

            # Filter data by dataset and training type
            filtered_cots = sample_cots_data.copy()

            if selected_dataset_cot and selected_dataset_cot != "all":
                filtered_cots = filtered_cots[filtered_cots['dataset'] == selected_dataset_cot]

            if selected_tt_cot:
                filtered_cots = filtered_cots[filtered_cots['training_type'].isin(selected_tt_cot)]

            # Apply shared questions filter if enabled
            if require_shared and len(selected_tt_cot) > 1:
                before_shared = len(filtered_cots)
                filtered_cots = filter_shared_questions(filtered_cots, selected_tt_cot, require_all=True)
                after_shared = len(filtered_cots)
                if before_shared > after_shared:
                    st.info(f"Filtered to {after_shared} examples with shared questions (from {before_shared})")

            # Apply sampling if specified
            if samples_per_step > 0 and not filtered_cots.empty:
                filtered_cots = sample_cots_per_step(filtered_cots, n_samples=samples_per_step, random_seed=42)

            if filtered_cots.empty:
                st.warning("No reasoning examples match the selected filters.")
                if require_shared and len(selected_tt_cot) > 1:
                    st.info("Try disabling 'Show only shared questions' - there may be no questions common to all selected training types.")
            else:
                # Show count summary
                st.caption(f"Showing {len(filtered_cots)} examples across {filtered_cots['step'].nunique() if 'step' in filtered_cots.columns else 1} steps")

                # Render Anthropic-style step-by-step viewer
                render_anthropic_style_viewer(filtered_cots)
                
                # Download button
                st.markdown("---")
                csv_cots = filtered_cots.to_csv(index=False)
                st.download_button(
                    label="📥 Download Reasoning Examples CSV",
                    data=csv_cots,
                    file_name=f"reasoning_examples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # ==========================================================================
    # Auto-refresh
    # ==========================================================================
    if auto_refresh:
        # Note: This will cause the page to refresh every 60 seconds
        # It will pause rendering for 60 seconds before rerunning
        time.sleep(60)
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()

