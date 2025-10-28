# Direct Multiple Shooting MPC Implementations

This directory contains multiple implementations of Direct Multiple Shooting (DMS) Model Predictive Control for differentiable physics simulation with Rewarped.

## 📁 Files Overview

### Core Implementations

1. **`dmsmpc.py`** - Constraint-based DMS MPC (Original)
   - Uses scipy SLSQP with equality constraints
   - Optimizes states + controls
   - Uses real physics simulation for dynamics
   - Works but can be slow

2. **`dmsmpc_differentiable.py`** - Pure Autodiff DMS MPC
   - Uses PyTorch Adam or L-BFGS optimizer
   - Optimizes controls only
   - Leverages Rewarped's differentiable physics
   - Fast and simple

3. **`dmsmpc_scipy_autodiff.py`** - Hybrid Approach ⭐ **RECOMMENDED**
   - Uses scipy optimizers (L-BFGS-B, SLSQP, etc.)
   - Provides exact gradients via PyTorch autodiff
   - Best of both worlds: robust scipy + fast autodiff
   - Most flexible (multiple optimizer options)

### Documentation

4. **`DIFFERENTIABLE_MPC_EXPLAINED.md`** - Deep dive into differentiable MPC
   - How autodiff transforms DMS MPC
   - Mathematical details
   - Implementation tips
   - Hyperparameter tuning guide

5. **`GRADIENT_METHODS_EXPLAINED.md`** - Gradient computation methods
   - Taxonomy of optimization methods
   - Finite differences vs autodiff
   - Scipy vs PyTorch optimizers
   - When to use what

6. **`QUICK_REFERENCE.md`** - Quick reference guide
   - Decision tree for choosing methods
   - Side-by-side comparison
   - One-page summary

7. **`README.md`** - This file

### Utilities

8. **`compare_approaches.py`** - Visualization and comparison script
   - Generates comparison plots
   - Demonstrates gradient flow
   - Shows computational complexity

## 🚀 Quick Start

### Option 1: Scipy + Autodiff (Recommended)

```bash
cd /app/mineral
python mineral/scripts/run.py \
    agent=DMSMPCScipyAutodiff \
    task=Rewarped \
    agent.dms_mpc_params.scipy_method=L-BFGS-B
```

### Option 2: Pure PyTorch

```bash
python mineral/scripts/run.py \
    agent=DMSMPCDifferentiable \
    task=Rewarped \
    agent.dms_mpc_params.optimizer=Adam
```

### Option 3: Original Constraint-Based

```bash
python mineral/scripts/run.py \
    agent=DMSMPC \
    task=Rewarped
```

## 📊 Comparison

| Method | Speed | Robustness | Variables | Gradients | Best For |
|--------|-------|------------|-----------|-----------|----------|
| **Scipy + Autodiff** ⭐ | 🟢 Fast | 🟢 High | Controls | Exact | Most cases |
| **PyTorch + Autodiff** | 🟢 Fast | 🟡 Medium | Controls | Exact | Noisy gradients |
| **Constraint-based** | 🔴 Slow | 🟢 High | States+Controls | Approximate | No autodiff |

## 🎯 Which One Should You Use?

### Use **Scipy + Autodiff** (`dmsmpc_scipy_autodiff.py`) if:
- ✅ You want the best balance of speed and robustness
- ✅ You have Rewarped (differentiable simulator)
- ✅ Your dynamics are reasonably smooth
- ✅ You want flexibility to try different optimizers

### Use **PyTorch + Autodiff** (`dmsmpc_differentiable.py`) if:
- ✅ Your gradients are noisy
- ✅ You want adaptive learning rates (Adam)
- ✅ You prefer PyTorch ecosystem
- ✅ You're okay with more iterations

### Use **Constraint-based** (`dmsmpc.py`) if:
- ✅ You don't have a differentiable simulator
- ✅ You have an analytical dynamics model
- ✅ You need explicit state constraints

## 📖 Key Concepts Explained

### What is "Gradient-Based Optimization"?

**All three methods use gradients!** The difference is:

1. **How gradients are computed:**
   - Finite differences (numerical approximation)
   - Automatic differentiation (exact via chain rule)

2. **Which optimizer is used:**
   - Scipy optimizers (L-BFGS-B, SLSQP, trust-constr)
   - PyTorch optimizers (Adam, SGD, L-BFGS)

3. **What is optimized:**
   - States + Controls (traditional DMS)
   - Controls only (with differentiable physics)

### Can You Use NLP with Scipy?

**Yes!** Scipy optimizers like SLSQP and trust-constr **are** NLP (Nonlinear Programming) solvers.

You can absolutely combine:
- **Scipy NLP solvers** (robust optimization)
- **PyTorch autodiff** (exact gradients)
- **Rewarped physics** (differentiable dynamics)

This is what `dmsmpc_scipy_autodiff.py` does!

## 🔧 Configuration

### Scipy + Autodiff Config

```yaml
# cfgs/agent/DMSMPCScipyAutodiff.yaml
agent:
  dms_mpc_params:
    N: 10                    # Horizon length
    max_iter: 30             # Max optimizer iterations
    scipy_method: 'L-BFGS-B' # Options: L-BFGS-B, SLSQP, TNC, trust-constr
    
    cost_state: 1.0
    cost_control: 0.1
    cost_terminal: 10.0
```

### PyTorch Config

```yaml
# cfgs/agent/DMSMPCDifferentiable.yaml
agent:
  dms_mpc_params:
    N: 10
    max_iter: 30
    optimizer: 'Adam'        # Options: Adam, LBFGS
    learning_rate: 0.05
    
    cost_state: 1.0
    cost_control: 0.1
    cost_terminal: 10.0
```

## 🧪 Testing

### Compare All Three Methods

```bash
# Run each method
python mineral/scripts/run.py agent=DMSMPC task=Rewarped
python mineral/scripts/run.py agent=DMSMPCDifferentiable task=Rewarped
python mineral/scripts/run.py agent=DMSMPCScipyAutodiff task=Rewarped

# Compare trajectories
python mineral/agents/dmsmpc/compare_approaches.py
```

## 📈 Performance Tips

### For Scipy + Autodiff:

1. **Try different optimizers:**
   ```python
   'L-BFGS-B'      # Fast, good for smooth problems
   'SLSQP'         # Handles constraints well
   'trust-constr'  # Most robust, slower
   'TNC'           # Good middle ground
   ```

2. **Adjust horizon:**
   - Shorter (N=5-10): Faster, more myopic
   - Longer (N=20-30): Better planning, slower

3. **Warm start:**
   ```python
   # Shift previous solution
   controls_init[:-1] = prev_solution[1:]
   controls_init[-1] = 0
   ```

### For PyTorch Optimizers:

1. **Learning rate tuning:**
   ```python
   lr = 0.01   # Conservative, stable
   lr = 0.05   # Balanced
   lr = 0.1    # Aggressive, faster but less stable
   ```

2. **Gradient clipping:**
   ```python
   torch.nn.utils.clip_grad_norm_([controls], max_norm=1.0)
   ```

3. **Multiple restarts:**
   ```python
   for restart in range(5):
       controls = torch.randn(N, 3) * 0.1
       # Optimize...
   ```

## 🐛 Debugging

### Check Gradient Flow

```python
controls = torch.zeros(N, 3, requires_grad=True)
cost = rollout_trajectory(controls)
cost.backward()

print(f"Gradient norm: {controls.grad.norm():.4f}")
# Should be non-zero!
```

### Verify Autodiff vs Finite Diff

```python
# See compare_approaches.py for gradient verification
```

### Common Issues

1. **Gradients are zero**
   - Check `requires_grad=True`
   - Ensure environment is in gradient mode
   - Verify operations are differentiable

2. **Optimizer not converging**
   - Reduce learning rate
   - Increase max iterations
   - Try different optimizer
   - Check cost function scaling

3. **Simulator state corruption**
   - Save/restore environment state properly
   - Clone state before optimization
   - Restore state before real execution

## 📚 Further Reading

1. **DIFFERENTIABLE_MPC_EXPLAINED.md** - Complete theory
2. **GRADIENT_METHODS_EXPLAINED.md** - Gradient computation deep dive
3. **QUICK_REFERENCE.md** - One-page summary

## 🎓 Learning Path

1. ✅ Read `QUICK_REFERENCE.md` (5 min)
2. ✅ Understand gradient-based optimization basics
3. ✅ Read `GRADIENT_METHODS_EXPLAINED.md` (15 min)
4. ✅ Learn about autodiff vs finite differences
5. ✅ Read `DIFFERENTIABLE_MPC_EXPLAINED.md` (30 min)
6. ✅ Understand how DMS MPC works with differentiable physics
7. ✅ Run `compare_approaches.py` to see visualizations
8. ✅ Test all three implementations
9. ✅ Choose the best one for your task

## 💡 Key Takeaway

**Gradient-based optimization** is a broad category that includes:
- Scipy optimizers ✅
- PyTorch optimizers ✅
- Any method that uses derivatives ✅

You can absolutely use **NLP solvers** (like scipy SLSQP) with **automatic differentiation** (from PyTorch) for the best of both worlds:
- Robust optimization algorithms from scipy
- Exact gradients from autodiff
- Differentiable physics from Rewarped

This is the **recommended approach** for DMS MPC with deformable objects! ⭐

