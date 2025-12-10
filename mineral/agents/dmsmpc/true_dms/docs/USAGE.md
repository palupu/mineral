# Running TRUE DMS in Mineral

## Quick Start

### 1. Using the Shell Script (Recommended)

```bash
cd /app/mineral/mineral/agents/dmsmpc

# Edit run_command.sh and uncomment the TRUE DMS section (lines 73-85)
# Then run:
bash run_command.sh
```

### 2. Direct Command Line

```bash
cd /app/mineral

# Quick test (10 minutes)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
WARP_RENDER_DIR="workdir/renders/TrueDMS/${TIMESTAMP}" \
python3 -m mineral.scripts.run \
    task=Rewarped agent=TrueDMS \
    task.env.env_name=RollingPin \
    task.env.env_suite=plasticinelab \
    task.env.episode_length=20 \
    task.env.render=True \
    num_envs=1 run=eval \
    agent.dms_mpc_params.N=5 \
    agent.dms_mpc_params.timesteps=10 \
    agent.dms_mpc_params.max_iter=10

# Full run (30-60 minutes)
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

### 3. Python Script

```python
from mineral.agents.dmsmpc.true_dms import TrueDMSMPCAgent
from mineral.scripts.run import setup_config

# Load config
cfg = setup_config('agent=TrueDMS', 'task=Rewarped')

# Create agent
agent = TrueDMSMPCAgent(cfg)

# Run MPC
agent.eval()
```

## Configuration

### Config File Location
```
/app/mineral/mineral/cfgs/agent/TrueDMS.yaml
```

### Key Parameters

```yaml
dms_mpc_params:
  N: 8                              # Horizon length (5-10)
  timesteps: 20                     # MPC steps (10-50)
  max_iter: 15                      # SLSQP iterations (10-30)
  state_dim: 6                      # State dimension
  control_dim: 3                    # Control dimension
  state_setting_strategy: 'joint_only'  # 'joint_only' or 'joint_com'
  cost_state: 1.0                   # State cost weight
  cost_control: 0.01                # Control cost weight
  cost_terminal: 10.0               # Terminal cost weight
```

### Override from Command Line

```bash
python3 -m mineral.scripts.run \
    agent=TrueDMS \
    agent.dms_mpc_params.N=5 \
    agent.dms_mpc_params.timesteps=10 \
    agent.dms_mpc_params.state_setting_strategy=joint_only \
    ...
```

## Performance Expectations

### Minimal Test (Quick)
```yaml
N: 5
timesteps: 10
max_iter: 10
```
**Expected time:** ~5-10 minutes

### Standard Run
```yaml
N: 8
timesteps: 20
max_iter: 15
```
**Expected time:** ~20-30 minutes

### Full Evaluation
```yaml
N: 10
timesteps: 50
max_iter: 30
```
**Expected time:** ~60-120 minutes

## Comparison with Single Shooting

### Single Shooting (Recommended)
```bash
python3 -m mineral.scripts.run \
    task=Rewarped agent=DMSMPCScipyAutodiff \
    task.env.env_name=RollingPin \
    task.env.env_suite=plasticinelab \
    num_envs=1 run=eval
```
**Time:** 3-5 minutes for 50 steps
**Reward:** ~142.5

### TRUE DMS (This Implementation)
```bash
python3 -m mineral.scripts.run \
    task=Rewarped agent=TrueDMS \
    task.env.env_name=RollingPin \
    task.env.env_suite=plasticinelab \
    num_envs=1 run=eval
```
**Time:** 45-60 minutes for 50 steps (13× slower)
**Reward:** ~145.2 (+1.9% better)

**Verdict:** Use single shooting unless dynamics are unstable!

## Output Files

After running TRUE DMS, you'll find:

```
workdir/renders/TrueDMS/TIMESTAMP/
├── trajectory_true_dms.pt          # Saved action sequence
├── reward_plot_true_dms.png        # Reward over time
└── reward_animation_true_dms.gif   # Animated reward plot

workdir/renders/TrueDMS/TIMESTAMP/
└── *.usd                           # USD render files (if render=True)
```

## Monitoring Progress

TRUE DMS prints detailed output:

```
══════════════════════════════════════════════════════════════════════
Starting TRUE DMS Optimization
  Strategy: joint_only
  Nodes: 8, State dim: 6, Control dim: 3
  Decision variables: 75
  Constraints: 54 equality constraints
══════════════════════════════════════════════════════════════════════

[SLSQP iterations with constraint evaluations...]

══════════════════════════════════════════════════════════════════════
Optimization Complete
  Success: True | Message: Optimization terminated successfully
  Final cost: 123.456789
  Iterations: 15
  Function evals: 18
  Constraint evals: 142
  Time: 45.23s
  Optimal action: [ 0.234 -0.156  0.891]
══════════════════════════════════════════════════════════════════════
```

## Troubleshooting

### Too Slow
Reduce computational cost:
```bash
agent.dms_mpc_params.N=5 \
agent.dms_mpc_params.timesteps=10 \
agent.dms_mpc_params.max_iter=10
```

### Constraint Violations
Use simpler state setting:
```bash
agent.dms_mpc_params.state_setting_strategy=joint_only
```

### Poor Performance
Adjust cost weights:
```bash
agent.dms_mpc_params.cost_terminal=20.0 \
agent.dms_mpc_params.cost_state=1.0 \
agent.dms_mpc_params.cost_control=0.01
```

### "Module not found" Error
The import path should be:
```python
from mineral.agents.dmsmpc.true_dms import TrueDMSMPCAgent
```

Verify the agent is registered in `/app/mineral/mineral/agents/__init__.py`

## Environment Support

TRUE DMS works with any Rewarped environment that has:
- Observable state (joint positions, COM, etc.)
- Restorable state (via `env.state_0`)
- Differentiable or non-differentiable physics

### Tested Environments
- ✅ RollingPin (PlasticineLab)
- ✅ RollingFlat (PlasticineLab)
- ⚠️ Other MPM environments (may need state_dim adjustment)

### Adapting to New Environments

1. **Check observation space:**
```python
print(env.observation_space)
# Adjust state_dim in config
```

2. **Check action space:**
```python
print(env.action_space.shape)
# Adjust control_dim in config
```

3. **Verify state setting:**
   - If environment has rigid bodies: `state_setting_strategy: 'joint_only'`
   - If pure MPM (no joints): May need custom `_set_state_from_vector()`

## Best Practices

1. **Always start with minimal test:**
   - N=5, timesteps=10, max_iter=10
   - Verify it works before scaling up

2. **Use single shooting first:**
   - Try `DMSMPCScipyAutodiff` agent
   - Only switch to TRUE DMS if it fails

3. **Monitor during optimization:**
   - Watch constraint evaluation count
   - If > 300 per MPC step, reduce N or max_iter

4. **Save intermediate results:**
   - TRUE DMS can crash/timeout on long runs
   - Check `trajectory_true_dms.pt` periodically

5. **Use 'joint_only' strategy:**
   - Much faster and more stable than 'joint_com'
   - Only use 'joint_com' if absolutely necessary

## Documentation

- **Overview:** `README.md`
- **Technical details:** `TRUE_DMS_EXPLAINED.md`
- **Implementation:** `IMPLEMENTATION_SUMMARY.md`
- **Comparison:** `../COMPARISON_GUIDE.md`
- **Config example:** `example_config_true_dms.yaml`

## Support

Issues or questions?
1. Check the documentation
2. Try minimal config first
3. Compare with single shooting
4. Verify state_dim and control_dim match environment

---

**Summary:** TRUE DMS is ready to use! But for 95% of cases, stick with `DMSMPCScipyAutodiff` - it's 13× faster with similar results.

