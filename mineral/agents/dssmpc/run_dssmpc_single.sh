#!/bin/bash
TIMESTAMP=$(date +%Y_%m_%d-%H_%M_%S_%2N)
export TIMESTAMP
mkdir -p workdir/DSSMPC/${TIMESTAMP}
script -q workdir/DSSMPC/${TIMESTAMP}/output.log -c "
python3 -m mineral.scripts.run \
    task=Rewarped agent=DSSMPC task.env.env_name=RollingPin task.env.env_suite=plasticinelab \
    task.env.render=True task.env.no_grad=False \
    logdir=\"workdir/DSSMPC/\${TIMESTAMP}\" \
    num_envs=1 run=eval seed=1 env_render=False task.env.randomize=False \
    agent.dss_mpc_params.N=5 agent.dss_mpc_params.timesteps=50 agent.dss_mpc_params.max_iter=50 && \
cp /root/.cache/warp/outputs/RollingPinPlasticineLab_1.usd workdir/DSSMPC/\${TIMESTAMP}/RollingPinPlasticineLab_1.usd
"