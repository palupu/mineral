# Differentiable Direct Multiple Shooting MPC

## Overview

This document explains how **Differentiable Physics** transforms Direct Multiple Shooting (DMS) MPC from a constrained optimization problem into a simpler gradient-based optimization.

---

## Traditional DMS MPC (Your Current Implementation)

### Problem Formulation

**Decision Variables:**
```
z = [x₀, u₀, x₁, u₁, ..., x_{N-1}, u_{N-1}, x_N]
```
- States `xᵢ` ∈ ℝⁿ and controls `uᵢ` ∈ ℝᵐ at each shooting node

**Optimization Problem:**
```
minimize   J(z) = Σᵢ₌₀^{N-1} (xᵢᵀ Q xᵢ + uᵢᵀ R uᵢ) + x_N^T Q_N x_N

subject to:
    x₀ = x_current                           (initial constraint)
    xᵢ₊₁ = f(xᵢ, uᵢ)    for i = 0,...,N-1  (dynamics constraints)
    u_min ≤ uᵢ ≤ u_max                       (control bounds)
```

**Solver:** SLSQP (Sequential Least Squares Programming)
- Handles constraints using Lagrange multipliers
- Requires Jacobian of constraints
- Can be slow for many constraints (6N for 6D state)

### The Deformable Object Challenge

For soft-body/plasticine tasks:
1. **High-dimensional state**: Thousands of particle positions/velocities
2. **Cannot set arbitrary states**: Can't directly jump to `xᵢ` in the simulator
3. **Complex dynamics**: MPM physics not easily expressible as `f(x,u)`
4. **Sequential simulation**: Current implementation is really "single shooting in disguise"

---

## Differentiable DMS MPC (New Approach)

### Key Insight

**If the simulator is differentiable**, we can compute:
```
∂J/∂u = ∂J/∂x₁ · ∂x₁/∂u₀ + ∂J/∂x₂ · ∂x₂/∂u₁ · ... (chain rule)
```

This means we can **optimize controls directly** without explicitly handling state variables or dynamics constraints!

### Simplified Problem

**Decision Variables:**
```
U = [u₀, u₁, ..., u_{N-1}]  (controls only!)
```

**Optimization:**
```
minimize   J(U) = cost(rollout(U))

subject to:
    u_min ≤ uᵢ ≤ u_max  (simple box constraints)
```

**Solver:** Adam, L-BFGS, or any gradient-based optimizer
- No constraints on dynamics (satisfied by simulation)
- No state variables to optimize
- Gradients computed via backpropagation through time

---

## How It Works with Rewarped

### 1. Automatic Differentiation Architecture

Rewarped implements **differentiable physics** using PyTorch's autograd:

```python
# Rewarped's gradient flow (simplified)
class UpdateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, control):
        # Run physics simulation
        next_state = integrate_physics(state, control)
        ctx.save_for_backward(state, control, next_state)
        return next_state
    
    @staticmethod
    def backward(ctx, grad_output):
        # Compute ∂next_state/∂control using adjoint method
        state, control, next_state = ctx.saved_tensors
        grad_control = adjoint_physics(state, control, grad_output)
        return None, grad_control
```

### 2. Trajectory Rollout with Gradients

```python
def rollout_trajectory(controls):
    """controls has requires_grad=True"""
    states = []
    cost = 0.0
    
    for i in range(N):
        # Step through differentiable physics
        obs, reward, done, info = env.step(controls[i])
        state = extract_state(obs)  # Maintains gradient
        states.append(state)
        
        # Accumulate cost (all operations are differentiable)
        cost += state.T @ Q @ state + controls[i].T @ R @ controls[i]
    
    # Terminal cost
    cost += states[-1].T @ Q_N @ states[-1]
    
    return states, cost  # cost has gradient w.r.t. controls!
```

### 3. Optimization via Backpropagation

```python
# Initialize controls with gradients enabled
controls = torch.zeros(N, control_dim, requires_grad=True)
optimizer = torch.optim.Adam([controls], lr=0.05)

for iteration in range(max_iter):
    optimizer.zero_grad()
    
    # Forward pass: simulate trajectory
    states, cost = rollout_trajectory(controls)
    
    # Backward pass: compute ∂cost/∂controls
    cost.backward()
    
    # Update controls using gradients
    optimizer.step()
    
# Extract optimal first control
optimal_action = controls[0].detach()
```

---

## Key Differences from Traditional DMS

| Aspect | Traditional DMS | Differentiable DMS |
|--------|----------------|-------------------|
| **Decision Variables** | States + Controls | Controls only |
| **Constraints** | Initial + Dynamics (6N) | Control bounds only |
| **Gradient Computation** | Finite differences | Automatic differentiation |
| **Solver** | SLSQP, IPOPT | Adam, L-BFGS |
| **Dynamics** | Analytical model `f(x,u)` | Black-box simulator |
| **Deformable Objects** | Difficult/impossible | Natural |
| **Computational Cost** | High (constraint Jacobians) | Medium (backprop) |

---

## Mathematical Details

### Backpropagation Through Time (BPTT)

The gradient of the cost w.r.t. controls is computed using the chain rule:

```
∂J/∂u₀ = ∂L₀/∂u₀ + ∂L₁/∂x₁ · ∂x₁/∂u₀ + ∂L₂/∂x₂ · ∂x₂/∂x₁ · ∂x₁/∂u₀ + ...

where:
    Lᵢ = stage cost at node i
    xᵢ₊₁ = physics_step(xᵢ, uᵢ)
```

**Rewarped computes this efficiently using:**
1. **Forward pass**: Store states x₀, x₁, ..., x_N
2. **Backward pass**: Propagate adjoints using chain rule

### Adjoint Method

For complex physics (MPM, contact), Rewarped uses the **adjoint method**:

```
Adjoint state: λᵢ = ∂J/∂xᵢ

Recursion (backward in time):
    λᵢ = ∂Lᵢ/∂xᵢ + (∂xᵢ₊₁/∂xᵢ)ᵀ · λᵢ₊₁
    
Control gradient:
    ∂J/∂uᵢ = ∂Lᵢ/∂uᵢ + (∂xᵢ₊₁/∂uᵢ)ᵀ · λᵢ₊₁
```

The Jacobians `∂xᵢ₊₁/∂xᵢ` and `∂xᵢ₊₁/∂uᵢ` are computed via:
- **Analytical derivatives** for rigid bodies
- **Finite differences** for MPM particles (if needed)
- **Implicit differentiation** through solvers

---

## Advantages for Deformable Objects

### 1. No State Representation Issues
- **Traditional**: Need to define `x` that includes all particle states → huge dimension
- **Differentiable**: Only observe reduced state (joint_q, com_q), physics handles particles internally

### 2. No Dynamics Model Needed
- **Traditional**: Need analytical `f(x,u)` for plasticine deformation → impossible!
- **Differentiable**: Simulator is the dynamics model

### 3. Exact Gradients Through Complex Physics
- **Traditional**: Finite difference gradients through MPM → expensive, noisy
- **Differentiable**: Automatic differentiation → exact (up to discretization), efficient

### 4. Natural Multiple Shooting
- **Traditional**: Can't set arbitrary particle configurations → fake multiple shooting
- **Differentiable**: Don't need to set states, just optimize controls

---

## Hyperparameter Tuning

### Horizon Length (N)
- **Shorter (5-10)**: Faster optimization, less myopic
- **Longer (20+)**: Better long-term planning, slower optimization
- **Deformable objects**: Start with N=10

### Learning Rate
- **High (0.1-0.5)**: Fast convergence, risk of instability
- **Medium (0.01-0.05)**: Balanced (recommended)
- **Low (0.001)**: Stable but slow

### Optimizer Choice

**Adam:**
```python
optimizer = torch.optim.Adam([controls], lr=0.05)
```
- Pros: Robust to noisy gradients, adaptive learning rates
- Cons: May not converge to exact optimum
- **Use for**: Stiff contacts, complex deformable physics

**L-BFGS:**
```python
optimizer = torch.optim.LBFGS([controls], lr=0.1, max_iter=20)
```
- Pros: Fast convergence, quasi-Newton method
- Cons: Requires more memory, sensitive to initialization
- **Use for**: Smooth dynamics, accurate solutions needed

### Cost Weights
```yaml
cost_state: 1.0      # Penalize state deviation
cost_control: 0.1    # Penalize control effort (avoid large actions)
cost_terminal: 10.0  # Strongly penalize final state error
```

**Tuning tips:**
- Increase `cost_control` if actions are too aggressive
- Increase `cost_terminal` for better goal reaching
- Balance `cost_state` vs `cost_terminal` for steady-state vs goal tasks

---

## Comparison with CEM MPC

You likely also have a **Cross-Entropy Method (CEM)** MPC implementation. Here's how they compare:

| Method | Differentiable DMS | CEM |
|--------|-------------------|-----|
| **Gradient Info** | Yes (exact) | No (sampling-based) |
| **Sample Efficiency** | High | Low (needs many samples) |
| **Convergence** | Fast (gradient descent) | Slower (evolution) |
| **Handles Discontinuities** | Poor (assumes smooth) | Good (gradient-free) |
| **Parallelization** | Sequential rollouts | Easy (parallel samples) |
| **Best for** | Smooth physics | Contact-rich, discrete |

**When to use differentiable DMS:**
- Smooth deformable dynamics (plasticine rolling)
- Need fast convergence
- Have differentiable simulator

**When to use CEM:**
- Contact-rich tasks (grasping, manipulation)
- Simulator is non-differentiable
- Can afford many rollouts

---

## Practical Implementation Tips

### 1. Gradient Clipping
Prevent exploding gradients through long rollouts:
```python
torch.nn.utils.clip_grad_norm_([controls], max_norm=1.0)
```

### 2. Warm Starting
Initialize controls from previous MPC solution:
```python
# Shift previous solution and append zero
controls[:-1] = prev_controls[1:].detach()
controls[-1] = 0
```

### 3. Line Search
For better convergence, use optimizers with line search:
```python
optimizer = torch.optim.LBFGS([controls], lr=1.0, max_iter=20, 
                               line_search_fn="strong_wolfe")
```

### 4. Multiple Random Restarts
Avoid local minima:
```python
best_cost = float('inf')
for restart in range(5):
    controls_init = torch.randn(N, control_dim) * 0.1
    cost, solution = optimize(controls_init)
    if cost < best_cost:
        best_cost = cost
        best_solution = solution
```

### 5. State Restoration
Important: Save and restore environment state after optimization:
```python
saved_state = clone_state(env.state_0)

# Optimize (modifies env state through rollouts)
optimal_action = dms_plan(current_state)

# Restore for real execution
env.state_0 = saved_state
env.step(optimal_action)
```

---

## Debugging Tips

### Check Gradient Flow
```python
controls = torch.zeros(N, control_dim, requires_grad=True)
states, cost = rollout_trajectory(controls)
cost.backward()

print(f"Control gradient norm: {controls.grad.norm():.4f}")
print(f"Gradient: {controls.grad}")

# Should be non-zero! If zero, gradients are blocked somewhere
```

### Verify Differentiability
```python
# Test with finite differences
u0 = controls[0].clone().detach()
eps = 1e-4

# Forward difference
controls[0] = u0 + eps
_, cost_plus = rollout_trajectory(controls)

controls[0] = u0
_, cost_nominal = rollout_trajectory(controls)

fd_grad = (cost_plus - cost_nominal) / eps
auto_grad = controls.grad[0]

print(f"Finite diff gradient: {fd_grad}")
print(f"Autograd gradient: {auto_grad}")
print(f"Relative error: {(fd_grad - auto_grad).norm() / auto_grad.norm():.4f}")
```

### Optimizer Not Converging
1. **Reduce learning rate**: Try 0.01 or 0.001
2. **Increase iterations**: Set max_iter to 50-100
3. **Check cost function**: Print intermediate costs
4. **Simplify problem**: Reduce horizon N to 3-5
5. **Check gradient scale**: Normalize cost by 1/N

---

## Expected Performance

### Computational Cost per MPC Step

**Constraint-based (SLSQP):**
- Variables: (state_dim + control_dim) × N + state_dim = 9N + 6 (for N=10: 96 vars)
- Constraints: state_dim × (N+1) = 6N + 6 (for N=10: 66 constraints)
- Time per iteration: ~50-100ms (constraint evaluation + Jacobian)
- Typical iterations: 20-50
- **Total: 1-5 seconds per MPC step**

**Differentiable (Adam):**
- Variables: control_dim × N = 3N (for N=10: 30 vars)
- Constraints: None (only bounds)
- Time per iteration: ~20-40ms (forward rollout + backward pass)
- Typical iterations: 10-30
- **Total: 0.2-1.2 seconds per MPC step**

**Speedup: 3-10× faster** (especially for larger horizons)

### Convergence Quality

Both methods should achieve similar final trajectories if:
- Cost functions are equivalent
- Optimization converges properly
- Gradients are accurate

Differentiable MPC often finds slightly better solutions due to:
- Exact gradients vs finite differences
- More efficient exploration of control space

---

## Next Steps

### 1. Test the Implementation
```bash
cd /app/mineral
python mineral/scripts/run.py agent=DMSMPCDifferentiable task=Rewarped
```

### 2. Compare with Baselines
- Run CEM MPC with same horizon
- Run constraint-based DMS (your current version)
- Compare: total reward, computation time, action smoothness

### 3. Hyperparameter Sweep
Try different combinations:
```python
learning_rates = [0.01, 0.05, 0.1]
horizons = [5, 10, 15, 20]
optimizers = ['Adam', 'LBFGS']
```

### 4. Advanced Extensions

**Model Predictive Path Integral (MPPI):**
Combine sampling and gradients
```python
# Use gradient to guide sampling distribution
mean_controls = controls.detach()
std = 0.1
samples = mean_controls + std * torch.randn(num_samples, N, control_dim)

# Evaluate samples and reweight
costs = [rollout_cost(s) for s in samples]
weights = softmax(-costs / temperature)

# Update mean with both gradients and weighted samples
controls_new = weighted_mean(samples, weights)
controls = 0.5 * controls_new + 0.5 * gradient_update(controls)
```

**Receding Horizon Planning:**
Update only first K < N controls, keep rest fixed
```python
controls_to_optimize = controls[:K]
controls_fixed = controls[K:].detach()
# Optimize only controls_to_optimize
```

**Value Function Terminal Cost:**
Learn terminal cost from data
```python
value_network = ValueNet(state_dim)
terminal_cost = value_network(states[-1])
```

---

## References

### Papers
1. **Differentiable Physics**: [DiffTaichi (ICLR 2020)](https://arxiv.org/abs/1910.00935)
2. **MPC with Gradients**: [Differentiable MPC (CoRL 2020)](https://arxiv.org/abs/1810.13400)
3. **Soft-Body Control**: [PlasticineLab (NeurIPS 2021)](https://arxiv.org/abs/2104.03311)

### Code Examples
- Rewarped: `/app/rewarped/rewarped/envs/gradsim/jumper.py` (lines 412-498)
- Gradient flow: `/app/rewarped/rewarped/autograd.py`
- State cloning: `/app/rewarped/rewarped/warp/model_monkeypatch.py`

---

## Summary

**Differentiable DMS MPC** transforms trajectory optimization from:
```
Constrained optimization with N×(state_dim + control_dim) variables
↓
Unconstrained optimization with N×control_dim variables
```

This is **particularly powerful for deformable objects** because:
- ✅ No need to represent high-dimensional particle states
- ✅ No need for analytical dynamics model
- ✅ Exact gradients through complex physics
- ✅ Faster optimization with fewer variables
- ✅ Natural handling of MPM, contacts, soft bodies

**Use this approach when:**
1. You have a differentiable simulator (Rewarped ✓)
2. Dynamics are complex/unknown (plasticine ✓)
3. States are high-dimensional (MPM particles ✓)
4. You need fast planning (real-time MPC ✓)

