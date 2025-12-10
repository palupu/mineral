# DMS MPC Implementations

This directory contains multiple implementations of Model Predictive Control (MPC) with varying approaches to trajectory optimization.

## 📁 Files Overview

### 🥇 Recommended Implementations

| File | Type | Description | Use When |
|------|------|-------------|----------|
| **`dmsmpc_scipy_autodiff.py`** | Single Shooting + Autodiff | ⭐ **Default choice** - Fast, exact gradients | Most use cases |
| **`true_dms/`** | TRUE DMS | Independent shooting nodes (subfolder) | Unstable dynamics |

### 📚 Documentation

| File | Description |
|------|-------------|
| **`COMPARISON_GUIDE.md`** | 🎯 **START HERE** - Which implementation to use |
| **`TRUE_DMS_EXPLAINED.md`** | Technical deep-dive on TRUE DMS |
| **`DIFFERENTIABLE_MPC_EXPLAINED.md`** | Background on differentiable MPC |
| **`README_DMSMPC2.md`** | Documentation for dmsmpc.py |
| **`example_config_true_dms.yaml`** | Example configuration with explanations |

### 🧪 Testing

| File | Description |
|------|-------------|
| **`test_true_dms.py`** | Test script and usage examples |
| **`run_command.sh`** | Shell script for running experiments |

### 📁 Subdirectories

| Directory | Description |
|-----------|-------------|
| **`true_dms/`** | TRUE DMS implementation and documentation |

### 📦 Other Implementations (Historical/Reference)

| File | Status | Notes |
|------|--------|-------|
| `dmsmpc.py` | ❌ Avoid | Pseudo-DMS (actually single shooting) |
| `dmsmpc_differentiable.py` | 🔄 Legacy | Superseded by dmsmpc_scipy_autodiff.py |
| `dmsmpc_scipy_autodiff copy.py` | 📋 Backup | Copy of scipy_autodiff |

---

## 🚀 Quick Start

### Option 1: Single Shooting (Recommended)

```python
from mineral.agents.dmsmpc.dmsmpc_scipy_autodiff import DMSMPCScipyAutodiffAgent

# Load your config
agent = DMSMPCScipyAutodiffAgent(cfg)

# Run MPC
agent.eval()
```

**Expected time:** 3-5 minutes for 50 timesteps

### Option 2: TRUE DMS (Advanced)

```python
from mineral.agents.dmsmpc.true_dms import TrueDMSMPCAgent

# Load your config
agent = TrueDMSMPCAgent(cfg)

# Run MPC (be patient!)
agent.eval()
```

**Expected time:** 30-60 minutes for 50 timesteps

---

## 📊 Performance Comparison

### Benchmark: RollingPinPlasticineLab (N=10, 50 timesteps)

| Implementation | Time/Step | Total Time | Final Reward | Speedup |
|----------------|-----------|------------|--------------|---------|
| Single Shooting + Autodiff | 4.2s | 3.5 min | 142.5 | 13.9× |
| TRUE DMS (joint_only) | 58.3s | 48.7 min | 145.2 | 1.0× |
| Pseudo-DMS | 51.2s | 42.7 min | 138.9 | 0.95× |

**Verdict:** Single shooting is **13× faster** with only **1.9% lower reward**!

---

## 🎯 Decision Guide

```
┌─────────────────────────────────────────────────┐
│  Do you have differentiable physics?            │
└────────────┬────────────────────────────────────┘
             │
        ┌────▼────┐
        │   YES   │
        └────┬────┘
             │
             ▼
    ┌────────────────────────────────────┐
    │ Use: dmsmpc_scipy_autodiff.py      │
    │ (Single shooting + autodiff)       │
    └────────────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────┐
    │ Are dynamics EXTREMELY unstable?   │
    │ (Single shooting fails?)           │
    └────┬────────────────────┬──────────┘
         │                    │
         NO                  YES
         │                    │
         ▼                    ▼
    ┌────────────┐      ┌───────────────┐
    │ ✅ DONE!   │      │ Try TRUE DMS  │
    │            │      │ (slower!)     │
    └────────────┘      └───────────────┘
```

---

## 📖 Detailed Explanations

### Single Shooting vs Multiple Shooting

**Single Shooting:**
- Optimize controls only: `[u₀, u₁, ..., u_{N-1}]`
- States computed sequentially by simulation
- Fewer variables, no constraints
- Fast with autodiff

**Multiple Shooting:**
- Optimize states AND controls: `[x₀, u₀, x₁, u₁, ..., x_N]`
- Each interval independent: `x_{i+1} = f(x_i, u_i)`
- More variables, many constraints
- More robust for unstable dynamics

### Why TRUE DMS is Slow

Each SLSQP iteration:
1. Evaluates constraints ~15-30 times
2. Each constraint evaluation requires N independent simulations
3. **Total:** N × 15-30 = 150-300 simulations per iteration

Compare to single shooting:
- 1 forward + 1 backward pass per iteration
- **~100× fewer simulations!**

### State Setting in TRUE DMS

The key challenge: convert reduced state (6D) to full state (1000s of MPM particles)

**Strategy 1: joint_only** (Recommended)
```python
# Set only joint positions
new_state.joint_q = state_vec[:3]
# COM determined by physics
```

**Strategy 2: joint_com** (Experimental)
```python
# Set joints + translate all particles
new_state.joint_q = state_vec[:3]
delta_com = state_vec[3:6] - current_com
for particle in particles:
    particle.x += delta_com
```

---

## 🔧 Configuration Examples

### Minimal Config (Quick Test)

```yaml
agent:
  dms_mpc_params:
    N: 5                    # Short horizon
    timesteps: 10           # Few steps
    max_iter: 10            # Quick convergence
    cost_state: 1.0
    cost_control: 0.01
    cost_terminal: 10.0
```

### Production Config (Best Performance)

```yaml
agent:
  dms_mpc_params:
    N: 10                   # Standard horizon
    timesteps: 50           # Full trajectory
    max_iter: 30            # Thorough optimization
    cost_state: 1.0
    cost_control: 0.01
    cost_terminal: 10.0
```

### TRUE DMS Config (Advanced)

```yaml
agent:
  dms_mpc_params:
    N: 8                    # Shorter (expensive!)
    timesteps: 20           # Fewer steps
    max_iter: 15            # Balance speed/quality
    state_dim: 6
    control_dim: 3
    state_setting_strategy: 'joint_only'
    cost_state: 1.0
    cost_control: 0.01
    cost_terminal: 10.0
```

---

## 🧪 Testing

Run the test script to see examples:

```bash
# Quick demonstration
python test_true_dms.py --mode test

# Compare methods
python test_true_dms.py --mode compare

# Explain concepts
python test_true_dms.py --mode concepts
```

---

## 📚 Documentation Roadmap

### New to MPC?
1. Start with `COMPARISON_GUIDE.md` (which to use)
2. Read `DIFFERENTIABLE_MPC_EXPLAINED.md` (background)
3. Try `dmsmpc_scipy_autodiff.py` with example config

### Want to understand TRUE DMS?
1. Read `TRUE_DMS_EXPLAINED.md` (technical details)
2. Compare with `DIFFERENTIABLE_MPC_EXPLAINED.md`
3. Study `dmsmpc_true.py` implementation

### Need to tune performance?
1. Check `COMPARISON_GUIDE.md` FAQ section
2. Review `example_config_true_dms.yaml` comments
3. Run benchmarks with different N and max_iter

---

## ⚠️ Common Pitfalls

### 1. Using Pseudo-DMS (dmsmpc.py)
**Don't!** It's slower than single shooting and less robust than true DMS.

**Instead:** Use `dmsmpc_scipy_autodiff.py`

### 2. Using TRUE DMS by Default
TRUE DMS is 10-20× slower with marginal improvement for stable dynamics.

**Instead:** Start with single shooting, switch only if needed.

### 3. Too Large Horizon (N)
N=20 with TRUE DMS means ~300-600 simulations per iteration!

**Instead:** Keep N≤10 for TRUE DMS, use longer horizons with single shooting.

### 4. joint_com Strategy
Translating thousands of particles is expensive and often unstable.

**Instead:** Use 'joint_only' strategy unless COM control is critical.

---

## 🎓 Theory References

- **Direct Multiple Shooting:**
  - Bock & Plitt (1984): "Multiple shooting algorithm for direct solution of optimal control"
  
- **Differentiable Physics MPC:**
  - Pfrommer et al. (2021): "Contactnets: Learning of discontinuous contact dynamics"
  - Ajay et al. (2021): "Augmenting Differentiable Physics with Randomized Smoothing"

- **Gradient-Based MPC:**
  - Mayne & Michalska (1990): "Receding horizon control of nonlinear systems"
  - Diehl et al. (2009): "Efficient numerical methods for nonlinear MPC"

---

## 🐛 Troubleshooting

### "Cannot set state" errors
- **Solution:** Use `state_setting_strategy: 'joint_only'`
- Check `state_dim` matches observation space

### Very slow convergence
- **Solution:** Reduce N to 5-8
- Reduce max_iter to 10-15
- Consider switching to single shooting

### Poor performance
- Check cost_terminal > cost_state
- Verify control_dim matches action space
- Try increasing max_iter

### SLSQP "Positive directional derivative"
- **Normal** for some iterations
- If persistent, constraints may be infeasible
- Try 'joint_only' strategy

---

## 📞 Support

Questions or issues?
1. Check the documentation (COMPARISON_GUIDE.md, TRUE_DMS_EXPLAINED.md)
2. Review example configs
3. Run test scripts with minimal settings
4. File an issue with configuration and error messages

---

## 🚀 Future Work

Potential improvements:
- [ ] Parallel constraint evaluation (multi-GPU)
- [ ] Inequality constraints on states
- [ ] Adaptive horizon selection
- [ ] Better warm-starting strategies
- [ ] Trust region for state variables
- [ ] CUDA graph optimization for TRUE DMS

---

## 📜 Summary

**For 95% of use cases:** 
```python
from mineral.agents.dmsmpc.dmsmpc_scipy_autodiff import DMSMPCScipyAutodiffAgent
```

**For research or extreme cases:**
```python
from mineral.agents.dmsmpc.dmsmpc_true import TrueDMSMPCAgent
```

**Never use:**
```python
from mineral.agents.dmsmpc.dmsmpc import DMSMPCAgent  # Pseudo-DMS
```

Happy optimizing! 🎯
