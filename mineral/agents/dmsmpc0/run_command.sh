TIMESTAMP=$(date +%Y%m%d-%H%M%S)
WARP_RENDER_DIR="workdir/renders/RewarpedDMSMPC0/${TIMESTAMP}" \
python3 -m mineral.scripts.run \
    task=Rewarped agent=RewarpedDMSMPC0 \
    task.env.env_name=RollingPin \
    task.env.env_suite=plasticinelab \
    task.env.episode_length=10 \
    task.env.render=True \
    num_envs=1 run=eval