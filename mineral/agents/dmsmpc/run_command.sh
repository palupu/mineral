#!/bin/bash

# Constant Control Test on RollingPin
python -m mineral.scripts.run \
task=Rewarped agent=RewarpedCEMMPC \
task.env.env_name=RollingPin task.env.env_suite=plasticinelab task.env.episode_length=50 \
num_envs=1 \
logdir="workdir/ConstantControlTest-RollingPin/$(date +%Y%m%d-%H%M%S)" \
run=eval seed=42


python3 -m mineral.scripts.run \
    task=Rewarped agent=RewarpedCEMMPC task.env.env_name=RollingPin task.env.env_suite=plasticinelab task.env.episode_length=50 \
    task.env.render=False task.env.no_grad=True \
    logdir="workdir/RewarpedCEMMPC" \
    num_envs=1 run=eval seed=1000 env_render=False task.env.randomize=False

python3 -m mineral.scripts.run \
    task=Rewarped agent=RewarpedCEMMPC task.env.env_name=RollingPin task.env.env_suite=plasticinelab task.env.episode_length=50 \
    task.env.render=True task.env.no_grad=True \
    logdir="workdir/RewarpedCEMMPC" \
    num_envs=1 run=eval seed=1000 env_render=True task.env.randomize=False \
    ckpt="workdir/Test/ckpt"


# Example with custom render directory (timestamped)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
WARP_RENDER_DIR="workdir/renders/Test/${TIMESTAMP}" \
python3 -m mineral.scripts.run \
    task=Rewarped agent=RewarpedCEMMPC \
    task.env.env_name=RollingPin \
    task.env.env_suite=plasticinelab \
    task.env.episode_length=10 \
    task.env.render=True \
    num_envs=1 run=eval

    task.env.render_mode=usd \



TIMESTAMP=$(date +%Y%m%d-%H%M%S)
WARP_RENDER_DIR="workdir/renders/RewarpedDMSMPC/${TIMESTAMP}" \
python3 -m mineral.scripts.run \
    task=Rewarped agent=RewarpedDMSMPC \
    task.env.env_name=RollingPin \
    task.env.env_suite=plasticinelab \
    task.env.episode_length=10 \
    task.env.render=True \
    num_envs=1 run=eval

# DMS MPC with Scipy Autodiff
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
WARP_RENDER_DIR="workdir/renders/DMSMPCScipyAutodiff/${TIMESTAMP}" \
python3 -m mineral.scripts.run \
    task=Rewarped agent=DMSMPCScipyAutodiff \
    task.env.env_name=RollingPin \
    task.env.env_suite=plasticinelab \
    task.env.episode_length=10 \
    task.env.render=True \
    num_envs=1 run=eval

# =============================================================================
# TRUE DMS (Advanced - 10-20× slower than single shooting!)
# =============================================================================
# Recommended only for unstable dynamics or research purposes
# Expected time: ~30-60 minutes per run (vs 3-5 minutes for single shooting)
#
# Configuration:
#   - N=8: Horizon length (reduce to 5 for faster testing)
#   - timesteps=20: Number of MPC steps (reduce to 10 for testing)
#   - state_setting_strategy='joint_only': Recommended strategy
#
# TIMESTAMP=$(date +%Y%m%d-%H%M%S)
# WARP_RENDER_DIR="workdir/renders/TrueDMS/${TIMESTAMP}" \
# python3 -m mineral.scripts.run \
#     task=Rewarped agent=TrueDMS \
#     task.env.env_name=RollingPin \
#     task.env.env_suite=plasticinelab \
#     task.env.episode_length=20 \
#     task.env.render=True \
#     num_envs=1 run=eval \
#     agent.dms_mpc_params.N=8 \
#     agent.dms_mpc_params.timesteps=20 \
#     agent.dms_mpc_params.max_iter=15 \
#     agent.dms_mpc_params.state_setting_strategy=joint_only