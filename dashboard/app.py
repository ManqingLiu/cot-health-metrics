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
DEFAULT_TRAINING_TYPES = ["baseline"]
DEFAULT_MODEL_NAMES = ["Qwen3-4B"]
DEFAULT_DATASET_NAMES = ["ca", "ba", "sb"]
# Updated learning rates as requested
DEFAULT_LEARNING_RATES = ["1e-4", "2e-5", "1e-5"]

# Default sample size for SE calculation (can be overridden if data provides it)
DEFAULT_SAMPLE_SIZE = 100

# Metrics available
ORIGINAL_METRICS = ["necessity", "substantivity", "paraphrasability"]
ALL_METRICS = ["accuracy"] + ORIGINAL_METRICS

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
    if pd.isna(p) or n <= 0:
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
    if pd.isna(std) or n <= 0:
        return np.nan
    return std / np.sqrt(n)


def calculate_cohens_d(mean1: float, std1: float, mean2: float, std2: float) -> float:
    """Calculate Cohen's d effect size. d = (mean1 - mean2) / pooled_std"""
    if std1 is None or std2 is None:
        std1 = std1 or 0.1
        std2 = std2 or 0.1
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
def load_from_wandb_multi(entity: str, projects: List[str], debug: bool = False) -> Optional[pd.DataFrame]:
    """Load metrics from multiple W&B projects.
    
    When there are multiple runs with the same name, keeps only the latest FINISHED run.
    """
    try:
        import wandb
        setup_wandb_auth()
        api = wandb.Api()
        
        all_data = []
        debug_info = []
        
        for project in projects:
            try:
                project_info = parse_project_name(project)
                runs = list(api.runs(f"{entity}/{project}"))
                
                # Filter to only finished or running runs (exclude crashed/failed)
                valid_states = {'finished', 'running'}
                valid_runs = [run for run in runs if run.state in valid_states]
                
                # Group runs by name and keep the best one:
                # - Prefer running over finished (running is the latest attempt)
                # - If same state, prefer the latest (newest created_at)
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
                            'created_at': created_at
                        }
                    else:
                        existing = runs_by_name[run_name]
                        # Prefer running over finished (running is the latest attempt)
                        if run_state == 'running' and existing['state'] == 'finished':
                            runs_by_name[run_name] = {
                                'run': run,
                                'state': run_state,
                                'created_at': created_at
                            }
                        # If same state, prefer the newer one
                        elif run_state == existing['state']:
                            if created_at and existing['created_at']:
                                if created_at > existing['created_at']:
                                    runs_by_name[run_name] = {
                                        'run': run,
                                        'state': run_state,
                                        'created_at': created_at
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
                    
                    # Fetch history with specific keys
                    history = run.history(keys=[
                        "step",
                        "eval/accuracy", "eval/accuracy_mean", "eval/accuracy_std",
                        "eval/num_total",
                        "eval/substantivity_mean", "eval/substantivity_std",
                        "eval/necessity_mean", "eval/necessity_std",
                        "eval/paraphrasability_mean", "eval/paraphrasability_std"
                    ])
                    
                    # Debug: track what we got
                    debug_info.append({
                        "project": project,
                        "run_name": run_name,
                        "run_state": run.state,
                        "created_at": str(run_info['created_at'])[:19] if run_info['created_at'] else "N/A",
                        "parsed_lr": learning_rate,
                        "parsed_dataset": dataset_name,
                        "rows_fetched": len(history) if not history.empty else 0,
                        "config_lr": config.get("learning_rate", "N/A"),
                    })
                    
                    if history.empty:
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
                    
                    all_data.append(history)
                    
            except Exception as e:
                st.warning(f"Error loading project {project}: {e}")
                continue
        
        # Store debug info in session state
        if debug_info:
            st.session_state['wandb_debug_info'] = debug_info
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
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


@st.cache_data(ttl=60)
def load_sample_cots_from_wandb(entity: str, projects: List[str], debug: bool = False) -> Optional[pd.DataFrame]:
    """
    Load sample CoTs (prompt, reasoning, answer) from W&B artifact tables.
    
    W&B Tables logged via wandb.log() are stored as media files and need to be
    accessed via the run's logged artifacts or files API.
    
    Returns DataFrame with columns:
    - step, question_id, question, prompt, cot, answer, training_type, dataset, learning_rate
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
                
                for run_name, run_info in runs_by_name.items():
                    run = run_info['run']
                    run_info_parsed = parse_run_name(run_name)
                    
                    training_type = run_info_parsed.get("training_type") or project_info.get("training_type", "unknown")
                    dataset_name = run_info_parsed.get("dataset_name") or project_info.get("dataset_name", "unknown")
                    learning_rate = run_info_parsed.get("learning_rate", "unknown")
                    
                    tables_found = 0
                    
                    # Method 1: Try to get tables from run files (media/table/)
                    try:
                        for file in run.files():
                            if 'sample_cots' in file.name and file.name.endswith('.json'):
                                try:
                                    # Download and parse the table file
                                    file_content = file.download(replace=True)
                                    with open(file_content.name, 'r') as f:
                                        table_json = json.load(f)
                                    
                                    columns = table_json.get('columns', [])
                                    data = table_json.get('data', [])
                                    
                                    if columns and data:
                                        table_df = pd.DataFrame(data, columns=columns)
                                        table_df['training_type'] = training_type
                                        table_df['dataset'] = dataset_name
                                        table_df['learning_rate'] = learning_rate
                                        table_df['run_name'] = run_name
                                        all_cots.append(table_df)
                                        tables_found += 1
                                except Exception as e:
                                    pass
                    except Exception as e:
                        pass
                    
                    # Method 2: Try history with pandas_style=True
                    if tables_found == 0:
                        try:
                            history = run.history(keys=["eval/sample_cots"], pandas=True)
                            
                            if not history.empty and "eval/sample_cots" in history.columns:
                                for _, row in history.iterrows():
                                    table_data = row.get("eval/sample_cots")
                                    if table_data is not None:
                                        try:
                                            if hasattr(table_data, 'get_dataframe'):
                                                table_df = table_data.get_dataframe()
                                            elif isinstance(table_data, dict):
                                                columns = table_data.get('columns', [])
                                                data = table_data.get('data', [])
                                                if columns and data:
                                                    table_df = pd.DataFrame(data, columns=columns)
                                                else:
                                                    continue
                                            elif isinstance(table_data, str):
                                                # Sometimes it's a path reference
                                                continue
                                            else:
                                                continue
                                            
                                            table_df['training_type'] = training_type
                                            table_df['dataset'] = dataset_name
                                            table_df['learning_rate'] = learning_rate
                                            table_df['run_name'] = run_name
                                            all_cots.append(table_df)
                                            tables_found += 1
                                        except Exception as e:
                                            pass
                        except Exception as e:
                            pass
                    
                    # Method 3: Try scan_history for streaming access
                    if tables_found == 0:
                        try:
                            for row in run.scan_history(keys=["eval/sample_cots"]):
                                table_data = row.get("eval/sample_cots")
                                if table_data and isinstance(table_data, dict):
                                    columns = table_data.get('columns', [])
                                    data = table_data.get('data', [])
                                    if columns and data:
                                        table_df = pd.DataFrame(data, columns=columns)
                                        table_df['training_type'] = training_type
                                        table_df['dataset'] = dataset_name
                                        table_df['learning_rate'] = learning_rate
                                        table_df['run_name'] = run_name
                                        all_cots.append(table_df)
                                        tables_found += 1
                        except Exception as e:
                            pass
                    
                    debug_info.append({
                        'project': project,
                        'run_name': run_name,
                        'tables_found': tables_found
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
            
            # Filter out invalid samples where CoT is empty
            original_count = len(result)
            result = filter_valid_cot_samples(result)
            filtered_count = len(result)
            if original_count > filtered_count:
                result.attrs['filtered_invalid_count'] = original_count - filtered_count
            
            return result if not result.empty else None
        return None
        
    except ImportError:
        return None
    except Exception as e:
        return None


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
</style>
"""


def render_cot_sample_anthropic_style(
    question: str,
    cot: str,
    answer: str,
    training_type: str = "baseline",
    step: int = 0,
    collapsed_cot: bool = False
) -> str:
    """
    Render a single CoT sample in Anthropic's extended thinking style.
    
    Uses XML-like semantic sections with clear visual hierarchy:
    - QUESTION: The input problem
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
    cot_escaped = escape_html(cot)
    answer_escaped = escape_html(answer)
    
    collapsed_class = "thinking-collapsed" if collapsed_cot and len(cot or "") > 500 else ""
    
    # Build HTML - only question, thinking, answer
    html = f'''
    <div class="sample-card">
        <div class="sample-header">
            <span class="training-badge {training_type}">{training_type}</span>
            <span class="step-badge">Step {step}</span>
        </div>
        
        <div class="question-section">
            <h4>📝 Question</h4>
            <p>{question_escaped}</p>
        </div>
        
        <div class="thinking-section {collapsed_class}">
            <pre>{cot_escaped if cot else "(No reasoning captured)"}</pre>
        </div>
        
        <div class="answer-section">
            <pre>{answer_escaped if answer else "(No answer)"}</pre>
        </div>
    </div>
    '''
    
    return html


def render_cot_comparison(samples: List[Dict]) -> str:
    """
    Render multiple CoT samples for comparison (e.g., across checkpoints).
    Shows only question, thinking, and answer.
    """
    if not samples:
        return "<p>No samples to display</p>"
    
    html = '<div class="comparison-grid">'
    
    for sample in samples:
        html += render_cot_sample_anthropic_style(
            question=sample.get('question', ''),
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
        yaxis=dict(range=[0, 110]),
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
    
    if metric == "accuracy":
        fig.update_yaxes(range=[0, 110])
    
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
            
            if pd.isna(baseline_mean):
                continue
            
            steps = []
            cohens_d_values = []
            
            for _, row in tt_data.sort_values('step').iterrows():
                current_mean = row.get(mean_col)
                current_std = row.get(std_col, 0.1)
                
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
        st.sidebar.subheader("📊 Per-Dataset Learning Rate")
        st.sidebar.markdown("*Select learning rate for each dataset*")
        
        dataset_lr_map = {}
        for ds in selected_datasets:
            lr = st.sidebar.selectbox(
                f"LR for {ds.upper()}",
                options=DEFAULT_LEARNING_RATES,
                index=len(DEFAULT_LEARNING_RATES) - 1,  # Default to 1e-4 (last in list)
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
        
        if generated_projects or projects_to_load:
            final_projects = projects_to_load if projects_to_load else generated_projects
            with st.spinner(f"Loading from {len(final_projects)} W&B projects..."):
                data = load_from_wandb_multi(wandb_entity, final_projects)
            
            if data is not None:
                st.sidebar.success(f"Loaded {len(data)} data points")
                
                if dataset_lr_map:
                    filtered_rows = []
                    for ds, lr in dataset_lr_map.items():
                        ds_data = data[(data['dataset'] == ds) & (data['learning_rate'] == lr)]
                        filtered_rows.append(ds_data)
                    if filtered_rows:
                        data = pd.concat(filtered_rows, ignore_index=True)
    
    else:  # Local Files
        st.sidebar.subheader("Local Settings")
        default_output = str(Path(__file__).parent.parent / "output")
        output_dir = st.sidebar.text_input("Output Directory", value=default_output)
        
        if st.sidebar.button("🔄 Refresh"):
            st.cache_data.clear()
        
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
            
            # Show filtering info
            st.markdown("**After LR filtering:**")
            st.write(f"Total rows in filtered_data: {len(filtered_data)}")
            st.write(f"Unique runs: {filtered_data['run_name'].nunique()}")
            st.write(f"Unique datasets: {filtered_data['dataset'].unique().tolist()}")
            st.write(f"Unique LRs: {filtered_data['learning_rate'].unique().tolist()}")
            
            # Show data counts per dataset
            st.markdown("**Data points per dataset:**")
            for ds in filtered_data['dataset'].unique():
                ds_data = filtered_data[filtered_data['dataset'] == ds]
                st.write(f"  {ds}: {len(ds_data)} rows, steps: {sorted(ds_data['step'].unique().tolist())}")
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
        
        # Styled HTML table for Cohen's d interpretation
        cohens_d_table_html = """
        <style>
            .cohens-table {
                border-collapse: collapse;
                margin: 15px 0;
                font-size: 14px;
                min-width: 400px;
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
                padding: 12px 15px;
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
        
        This panel displays questions, chain-of-thought reasoning, and answers in an 
        [Anthropic-style format](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-of-thought):
        
        - **QUESTION**: The input problem
        - **THINKING**: The model's step-by-step reasoning
        - **ANSWER**: The final response
        """)
        
        # Show training type prompts table
        with st.expander("📋 Custom Instructions by Training Type", expanded=False):
            st.markdown("""
            Each training type uses a different custom instruction appended to the prompt:
            """)
            
            prompts_data = {
                "Training Type": ["baseline", "internalized", "encoded", "post-hoc"],
                "Custom Instruction": [
                    "Let's think step by step.",
                    "Only use Lorem ipsum style reasoning from training in your reasoning steps, after you finish reasoning, close the think tag, and provide the final CORRECT answer.",
                    "Only use the coded style reasoning from training in your reasoning steps, then provide the CORRECT answer.",
                    "You already KNOW the CORRECT answer, which is {answer}, but you need to write your reasoning steps for the user."
                ],
                "Description": [
                    "Standard chain-of-thought prompting",
                    "Encourages internalized reasoning with placeholder text",
                    "Uses encoded/compressed reasoning patterns",
                    "Given the answer, model must justify it (post-hoc rationalization)"
                ]
            }
            st.table(pd.DataFrame(prompts_data))
        
        # Load sample CoTs data
        sample_cots_data = None
        
        if data_source == "W&B":
            # Try to load from W&B
            if 'final_projects' in dir() or 'projects_to_load' in dir():
                try:
                    projects_for_cots = projects_to_load if projects_to_load else generated_projects
                    with st.spinner("Loading reasoning examples from W&B..."):
                        sample_cots_data = load_sample_cots_from_wandb(wandb_entity, projects_for_cots)
                except Exception as e:
                    st.warning(f"Error loading from W&B: {e}")
        else:
            # Load from local files
            try:
                with st.spinner("Loading reasoning examples from local files..."):
                    sample_cots_data = load_sample_cots_from_local(output_dir)
            except Exception as e:
                st.warning(f"Error loading local files: {e}")
        
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
                    for proj in (projects_to_load if projects_to_load else generated_projects):
                        st.markdown(f"- [{proj}](https://wandb.ai/{wandb_entity}/{proj})")
            
        else:
            # We have real data!
            # Check if any invalid samples were filtered
            filtered_count = getattr(sample_cots_data, 'attrs', {}).get('filtered_invalid_count', 0)
            if filtered_count > 0:
                st.success(f"Loaded {len(sample_cots_data)} valid reasoning examples")
                st.info(f"ℹ️ Filtered out {filtered_count} invalid samples (empty CoT or answer - model exceeded max token budget)")
            else:
                st.success(f"Loaded {len(sample_cots_data)} reasoning examples")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
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
                    default=available_training_types[:2] if len(available_training_types) >= 2 else available_training_types,
                    key="cot_training_types"
                )
            
            with col3:
                available_steps = sorted(sample_cots_data['step'].dropna().unique().tolist())
                selected_steps_cot = st.multiselect(
                    "Steps",
                    options=available_steps,
                    default=[available_steps[0], available_steps[-1]] if len(available_steps) >= 2 else available_steps,
                    key="cot_steps"
                )
            
            # Additional options
            available_questions = sample_cots_data['question_id'].dropna().unique().tolist()[:20]
            selected_question_id = st.selectbox(
                "Question ID (filter to specific question)",
                options=["All"] + list(available_questions),
                key="cot_question_id"
            )
            
            # Filter data
            filtered_cots = sample_cots_data.copy()
            
            if selected_dataset_cot and selected_dataset_cot != "all":
                filtered_cots = filtered_cots[filtered_cots['dataset'] == selected_dataset_cot]
            
            if selected_tt_cot:
                filtered_cots = filtered_cots[filtered_cots['training_type'].isin(selected_tt_cot)]
            
            if selected_steps_cot:
                filtered_cots = filtered_cots[filtered_cots['step'].isin(selected_steps_cot)]
            
            if selected_question_id and selected_question_id != "All":
                filtered_cots = filtered_cots[filtered_cots['question_id'] == selected_question_id]
            
            if filtered_cots.empty:
                st.warning("No reasoning examples match the selected filters.")
            else:
                # Display options
                display_mode = st.radio(
                    "Display Mode",
                    options=["Comparison View", "Single Sample View", "Table View"],
                    horizontal=True,
                    key="cot_display_mode"
                )
                
                st.markdown("---")
                
                if display_mode == "Comparison View":
                    # Group by question_id and show evolution across steps/training types
                    st.subheader("📊 Compare Reasoning Across Training")
                    
                    # Get unique question IDs
                    question_ids = filtered_cots['question_id'].unique()[:5]  # Limit to 5 questions
                    
                    for q_id in question_ids:
                        q_samples = filtered_cots[filtered_cots['question_id'] == q_id]
                        
                        if not q_samples.empty:
                            question_text = q_samples.iloc[0].get('question', f'Question {q_id}')
                            
                            with st.expander(f"Question {q_id}: {question_text[:100]}...", expanded=True):
                                # Convert to list of dicts for rendering
                                samples_list = []
                                for _, row in q_samples.iterrows():
                                    samples_list.append({
                                        'question': row.get('question', ''),
                                        'prompt': row.get('prompt', ''),
                                        'cot': row.get('cot', ''),
                                        'answer': row.get('answer', ''),
                                        'training_type': row.get('training_type', 'baseline'),
                                        'step': row.get('step', 0)
                                    })
                                
                                # Sort by step and training type
                                samples_list.sort(key=lambda x: (x['step'], x['training_type']))
                                
                                # Render comparison
                                comparison_html = render_cot_comparison(samples_list)
                                st.markdown(comparison_html, unsafe_allow_html=True)
                
                elif display_mode == "Single Sample View":
                    # Show one sample at a time with full details
                    st.subheader("🔍 Detailed Sample View")
                    
                    sample_idx = st.slider(
                        "Sample Index",
                        min_value=0,
                        max_value=len(filtered_cots) - 1,
                        value=0,
                        key="single_sample_idx"
                    )
                    
                    sample = filtered_cots.iloc[sample_idx]
                    
                    # Render single sample with full content
                    single_html = render_cot_sample_anthropic_style(
                        question=sample.get('question', ''),
                        cot=sample.get('cot', ''),
                        answer=sample.get('answer', ''),
                        training_type=sample.get('training_type', 'baseline'),
                        step=int(sample.get('step', 0)),
                        collapsed_cot=False  # Don't collapse in single view
                    )
                    st.markdown(single_html, unsafe_allow_html=True)
                    
                    # Navigation buttons
                    col_prev, col_next = st.columns(2)
                    with col_prev:
                        if sample_idx > 0:
                            if st.button("← Previous Sample"):
                                st.session_state.single_sample_idx = sample_idx - 1
                                st.rerun()
                    with col_next:
                        if sample_idx < len(filtered_cots) - 1:
                            if st.button("Next Sample →"):
                                st.session_state.single_sample_idx = sample_idx + 1
                                st.rerun()
                
                else:  # Table View
                    st.subheader("📋 Reasoning Examples Table")
                    
                    # Show as dataframe with selectable columns
                    display_cols = ['step', 'training_type', 'dataset', 'question_id', 'question', 'answer']
                    available_cols = [c for c in display_cols if c in filtered_cots.columns]
                    
                    st.dataframe(
                        filtered_cots[available_cols],
                        use_container_width=True,
                        height=400
                    )
                    
                    # Allow downloading
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
        time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()
