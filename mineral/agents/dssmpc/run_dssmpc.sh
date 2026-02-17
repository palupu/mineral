#!/bin/bash
# Usage: ./run_dssmpc.sh [BASE_DIR] [N] [MAX_ITER] [LR] [SAMPLE] [SEED]
# Example: ./run_dssmpc.sh workdir/DSSMPC/20250108-120000 5 50 0.01 false 1

# Parse parameters with defaults
BASE_DIR="${1:-workdir/DSSMPC/$(date +%Y_%m_%d-%H_%M_%S)}"
N="${2:-1}"
MAX_ITER="${3:-1}"
LR="${4:-0.01}"
SAMPLE="${5:-false}"
SEED="${6:-4}"

# Create folder name with parameters
RUN_FOLDER="${BASE_DIR}/N${N}_iter${MAX_ITER}_lr${LR}_sample${SAMPLE}_seed${SEED}"
mkdir -p "${RUN_FOLDER}"

echo "=========================================="
echo "Running DSSMPC with configuration:"
echo "  N: ${N}"
echo "  max_iter: ${MAX_ITER}"
echo "  learning_rate: ${LR}"
echo "  sample: ${SAMPLE}"
echo "  seed: ${SEED}"
echo "  Output folder: ${RUN_FOLDER}"
echo "=========================================="

# Output is redirected by Python script when called from run_multiple_dssmpc.py
# When run standalone, output goes to terminal (can be redirected manually)
# 
# To redirect output manually when running standalone:
#   ./run_dssmpc.sh > output.log 2>&1
#   ./run_dssmpc.sh | tee output.log          # See output AND save to file
#   ./run_dssmpc.sh >> output.log 2>&1       # Append to existing file
#
python3 -m mineral.scripts.run \
    task=Rewarped agent=DSSMPC task.env.env_name=RollingPin task.env.env_suite=plasticinelab \
    task.env.render=True task.env.no_grad=False \
    logdir="${RUN_FOLDER}" \
    num_envs=1 run=eval seed=${SEED} env_render=False task.env.randomize=False \
    agent.dss_mpc_params.N=${N} \
    agent.dss_mpc_params.timesteps=300 \
    agent.dss_mpc_params.max_iter=${MAX_ITER} \
    agent.dss_mpc_params.learning_rate=${LR} \
    agent.dss_mpc_params.sapo_sample=${SAMPLE} && \
cp /root/.cache/warp/outputs/RollingPinPlasticineLab_1.usd ${RUN_FOLDER}/RollingPinPlasticineLab_1.usd

