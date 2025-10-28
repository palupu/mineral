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