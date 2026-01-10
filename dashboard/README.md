# CoT Health Metrics Dashboard

A comprehensive Streamlit dashboard for monitoring and analyzing CoT (Chain-of-Thought) health metrics across training runs.

## Features

### 📈 Per-Dataset Visualizations
- **Grouped Bar Charts**: Accuracy at each checkpoint step, grouped by training type (like the example images)
- **Cohen's d Line Plots**: Effect sizes for each metric with different line styles per training type
- **Original Metrics**: Track Necessity, Substantivity, and Paraphrasability per dataset

### 🔧 Flexible Project Selection
Build W&B project queries using the naming convention:
- **Projects**: `{training_type}-lr-sweep-{model_name}-{dataset_name}`
- **Runs**: `{training_type}_{dataset_name}_lr{learning_rate}`

Select from:
- Training types: baseline, internalized, encoded, post-hoc
- Models: Qwen3-4B, Qwen3-1.7B, Qwen3-0.6B, gpt-oss-20B
- Datasets: ca, ba, sb, li
- Learning rates: 1e-5, 2e-5, 5e-5, 1e-4

### 📊 Per-Dataset Learning Rate Selection
**NEW**: Choose different learning rates for different datasets:
- CA → 1e-4
- BA → 5e-5  
- SB → 1e-5

This allows comparing the best hyperparameters per dataset.

### 📊 Visualization Types
1. **Per-Dataset Accuracy**: Grouped bar charts with Binomial Standard Error bars
2. **Per-Dataset Metrics**: Timeline plots with SEM error bars for any metric
3. **Per-Dataset Cohen's d**: Line plots with different line styles (solid, dash, dot) for each training type
4. **Summary Table**: Downloadable CSV with Standard Error values

### 📐 Error Bar Methodology (Standard Methods)
- **Accuracy** (Binomial SE): `SE = √(p(1-p)/n)` where p is accuracy proportion, n is sample size
- **Other metrics** (SEM): `SE = σ/√n` where σ is standard deviation, n is sample size

This follows standard statistical practice for reporting uncertainty in evaluation metrics.

### 🔄 Data Sources
- **W&B (Weights & Biases)**: Real-time monitoring during training
- **Local Files**: Post-training analysis from `metrics_history.json`

## Quick Start

```bash
# Install dependencies
pip install -r dashboard/requirements.txt

# Run the dashboard
streamlit run dashboard/app.py
```

## 🚀 Deploy to Streamlit Cloud (Share with Anyone)

**Option 2** for sharing: Deploy to Streamlit Community Cloud so anyone with the link can access it.

### Step 1: Push to GitHub

```bash
# Initialize git if needed
git init
git add dashboard/
git commit -m "Add dashboard for cloud deployment"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository
5. Set the main file path: `dashboard/app.py`
6. Click **"Deploy!"**

### Step 3: Make W&B Projects Public (Recommended)

**Easiest option** - no secrets needed:

1. Go to your W&B project page (e.g., `https://wandb.ai/mliu7/baseline-lr-sweep-Qwen3-4B-ca`)
2. Click **⚙️ Settings** (gear icon)
3. Change **"Project Visibility"** to **"Public"**
4. Click **Save**

Repeat for each project. Public projects can be accessed without authentication.

### Alternative: Use W&B API Key (for private projects)

If you want to keep projects private, add secrets in Streamlit Cloud:

1. Go to your deployed app → **⋮ (menu)** → **Settings** → **Secrets**
2. Add:

```toml
WANDB_API_KEY = "your-wandb-api-key-here"
WANDB_ENTITY = "mliu7"
```

Get your W&B API key from: https://wandb.ai/authorize

## Usage Guide

### 1. Select Data Source

Choose between W&B or Local Files in the sidebar.

### 2. Build Project Names (W&B)

Use the Project Builder to generate W&B project names:

1. Select **Training Types** (e.g., all four: baseline, post-hoc, internalized, encoded)
2. Select **Models** (e.g., Qwen3-4B)
3. Select **Datasets** (e.g., ca, ba, sb)

This generates project names like:
- `baseline-lr-sweep-Qwen3-4B-ca`
- `post-hoc-lr-sweep-Qwen3-4B-ca`
- `internalized-lr-sweep-Qwen3-4B-ca`
- `encoded-lr-sweep-Qwen3-4B-ca`
- ... (for each dataset)

### 3. Set Per-Dataset Learning Rates

**NEW**: Select the optimal learning rate for each dataset:

| Dataset | Learning Rate |
|---------|---------------|
| CA      | 1e-4          |
| BA      | 5e-5          |
| SB      | 1e-5          |

This filters runs to only show the selected LR for each dataset.

### 4. Explore Tabs

| Tab | Description |
|-----|-------------|
| **📈 Per-Dataset Accuracy** | Grouped bar charts with Binomial SE error bars |
| **🔬 Per-Dataset Metrics** | Original metrics with SEM error bars |
| **📊 Per-Dataset Cohen's d** | Effect size plots with different line styles per training type |
| **📋 Summary Table** | Tabular summary with SE values and CSV download |

## Example Workflows

### Compare All Training Types Per Dataset (Like Example Images)

1. Select Training Types: `baseline`, `post-hoc`, `internalized`, `encoded`
2. Select Model: `Qwen3-4B`
3. Select Datasets: `ba`, `ca`, `sb`
4. Set per-dataset learning rates:
   - BA: `5e-5`
   - CA: `1e-4`
   - SB: `1e-5`
5. Click "Refresh from W&B"
6. Go to **📈 Per-Dataset Accuracy** tab
   - See grouped bar charts like the example image
7. Go to **📊 Per-Dataset Cohen's d** tab
   - See line plots with different line styles per training type

### Track Cohen's d for All Metrics

1. Select all Training Types
2. Select Model: `Qwen3-4B`
3. Select Dataset: `ba`
4. Set LR for BA: `5e-5`
5. Load data
6. Go to **Per-Dataset Cohen's d** tab
7. Select Metrics: `necessity`, `substantivity`, `paraphrasability`
8. View line plots showing Cohen's d evolution over training

### Compare Final Accuracy Across Training Types

1. Configure training types, model, datasets, and learning rates
2. Load data
3. Go to **Summary Histograms** tab
4. Select Metric: `accuracy`
5. Group By: `training_type`

## W&B Metrics

The dashboard reads the following metrics from W&B:

```
eval/accuracy, eval/accuracy_std
eval/substantivity_mean, eval/substantivity_std
eval/necessity_mean, eval/necessity_std
eval/paraphrasability_mean, eval/paraphrasability_std
```

These are logged by `training_callbacks.py` during training.

## Cohen's d Interpretation

Cohen's d measures the effect size between baseline (step 0) and trained checkpoints:

| Effect Size | |d| Range | Interpretation |
|-------------|----------|----------------|
| Negligible  | < 0.2    | Similar to baseline |
| Small       | 0.2 - 0.5| Minor change |
| Medium      | 0.5 - 0.8| Moderate change |
| Large       | ≥ 0.8    | Significant change |

**Convention**: `d = (baseline - current) / pooled_std`
- Positive d: metric decreased from baseline
- Negative d: metric increased from baseline

## Local Files Structure

For local file analysis, the dashboard expects:

```
output/
├── baseline_Qwen3-4B_CA_20251215/
│   └── metrics_history.json
├── internalized_Qwen3-4B_BA_20251215/
│   └── metrics_history.json
└── ...
```

## Files

```
dashboard/
├── app.py                 # Main Streamlit dashboard application
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── .streamlit/
    ├── config.toml        # Streamlit theme and server configuration
    └── secrets.toml.example  # Template for W&B secrets (copy to secrets.toml)
```

## Environment Variables

- `WANDB_ENTITY`: Default W&B entity (default: "mliu7")
- `WANDB_API_KEY`: W&B API key (can also be set via Streamlit secrets)
