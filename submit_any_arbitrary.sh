#!/bin/bash
PROJECT_NAME=$1
shift
OTHER_ARGS="$@"

CACHE_DIR=/network/scratch/l/let/projects/cot_health_metrics2/cache

sbatch <<EOF
#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=$PROJECT_NAME
#SBATCH --time=6:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -o job/$PROJECT_NAME.out
#SBATCH -e job/$PROJECT_NAME.err

cd /network/scratch/l/let/projects/cot_health_metrics2

module load cuda/12.6.0/cudnn/9.3
module load anaconda
conda activate /network/scratch/l/let/envs/guardbench

export PYTHONBREAKPOINT=0

export PYTHONPATH=.
export HF_CACHE=hf_cache
export HF_HOME=hf_home

source key.sh

$OTHER_ARGS
EOF
