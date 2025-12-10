# TRUE DMS - Quick Start Guide

## ⚡ Run in 3 Steps

### Step 1: Uncomment the Command

Edit `/app/mineral/mineral/agents/dmsmpc/run_command.sh`:

```bash
# Find lines 73-85 and remove the # symbols:

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
WARP_RENDER_DIR="workdir/renders/TrueDMS/${TIMESTAMP}" \
python3 -m mineral.scripts.run \
    task=Rewarped agent=TrueDMS \
    task.env.env_name=RollingPin \
    task.env.env_suite=plasticinelab \
    task.env.episode_length=20 \
    task.env.render=True \
    num_envs=1 run=eval \
    agent.dms_mpc_params.N=8 \
    agent.dms_mpc_params.timesteps=20 \
    agent.dms_mpc_params.max_iter=15 \
    agent.dms_mpc_params.state_setting_strategy=joint_only
```

### Step 2: Run It

```bash
cd /app/mineral/mineral/agents/dmsmpc
bash run_command.sh
```

### Step 3: Wait ⏰

Be patient! TRUE DMS takes **30-60 minutes** to complete.

---

## 🚀 Or Use Command Line Directly

### Quick Test (10 minutes)

```bash
cd /app/mineral

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
WARP_RENDER_DIR="workdir/renders/TrueDMS/${TIMESTAMP}" \
python3 -m mineral.scripts.run \
    task=Rewarped agent=TrueDMS \
    task.env.env_name=RollingPin \
    task.env.env_suite=plasticinelab \
    task.env.episode_length=20 \
    task.env.render=False \
    num_envs=1 run=eval \
    agent.dms_mpc_params.N=5 \
    agent.dms_mpc_params.timesteps=10 \
    agent.dms_mpc_params.max_iter=10
```

### Full Run (60 minutes)

```bash
cd /app/mineral

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
WARP_RENDER_DIR="workdir/renders/TrueDMS/${TIMESTAMP}" \
python3 -m mineral.scripts.run \
    task=Rewarped agent=TrueDMS \
    task.env.env_name=RollingPin \
    task.env.env_suite=plasticinelab \
    task.env.episode_length=20 \
    task.env.render=True \
    num_envs=1 run=eval \
    agent.dms_mpc_params.N=8 \
    agent.dms_mpc_params.timesteps=20 \
    agent.dms_mpc_params.max_iter=15
```

---

## 📊 Compare with Single Shooting

### Run Single Shooting (Faster!)

```bash
cd /app/mineral

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
WARP_RENDER_DIR="workdir/renders/DMSMPCScipyAutodiff/${TIMESTAMP}" \
python3 -m mineral.scripts.run \
    task=Rewarped agent=DMSMPCScipyAutodiff \
    task.env.env_name=RollingPin \
    task.env.env_suite=plasticinelab \
    task.env.episode_length=20 \
    task.env.render=True \
    num_envs=1 run=eval
```

**Time:** 3-5 minutes (13× faster!)

**Reward:** ~142.5 (vs ~145.2 for TRUE DMS)

---

## 📂 Output Files

Find your results in:

```
/app/mineral/workdir/renders/TrueDMS/TIMESTAMP/
├── trajectory_true_dms.pt          # Action sequence
├── reward_plot_true_dms.png        # Reward plot
├── reward_animation_true_dms.gif   # Animated plot
└── *.usd                           # Render files
```

---

## ⚙️ Configuration

Config file: `/app/mineral/mineral/cfgs/agent/TrueDMS.yaml`

Key settings:
- `N`: Horizon length (5-10)
- `timesteps`: MPC steps (10-50)
- `max_iter`: Optimization iterations (10-30)
- `state_setting_strategy`: 'joint_only' (recommended) or 'joint_com'

---

## 🎯 Should You Use TRUE DMS?

### ✅ Use if:
- Single shooting fails (trajectories diverge)
- Dynamics are highly unstable
- Need extra robustness at any cost

### ❌ Don't use if:
- Speed matters (13× slower!)
- Dynamics are stable (MPM usually is)
- Default: Use `DMSMPCScipyAutodiff` instead

---

## 📚 Full Documentation

- **Usage guide:** `USAGE.md`
- **Technical details:** `TRUE_DMS_EXPLAINED.md`
- **Comparison:** `../COMPARISON_GUIDE.md`

**Bottom line:** TRUE DMS works, but single shooting is faster with similar results for stable dynamics!

