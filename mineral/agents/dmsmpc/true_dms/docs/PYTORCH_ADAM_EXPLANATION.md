# TRUE DMS MPC: PyTorch Adam + Penalty Method

## Overview

This implementation replaces scipy's SLSQP constrained optimizer with **PyTorch's Adam optimizer** using the **penalty method**. This simplifies the optimization while maintaining TRUE Direct Multiple Shooting principles.

---

## Key Changes

### 1. **From Scipy SLSQP → PyTorch Adam**

**Before (Scipy):**
```python
from scipy.optimize import minimize

result = minimize(
    fun=cost_function,          # Returns (cost, gradient)
    x0=initial_guess,
    method='SLSQP',
    jac=True,                   # Analytical gradients
    bounds=bounds,
    constraints=[eq_cons]       # Hard equality constraints
)
```

**After (PyTorch):**
```python
import torch.optim as optim

x_u = torch.nn.Parameter(initial_guess)
optimizer = optim.Adam([x_u], lr=learning_rate)

for iteration in range(max_iter):
    optimizer.zero_grad()
    loss = compute_loss(x_u, ...)  # Combined loss
    loss.backward()
    optimizer.step()
```

---

### 2. **From Hard Constraints → Soft Penalties**

#### Constrained Optimization (Old Approach)

**Problem Formulation:**
```
minimize    J(x, u) = Σ cost(x[i], u[i]) + terminal_cost(x[N])
subject to  x[0] = x_current
            x[i+1] = f(x[i], u[i])  for i = 0, ..., N-1
            u_min ≤ u[i] ≤ u_max
```

**Implementation:**
- Required **Jacobian matrices** of constraints: ∂c/∂(x,u)
- Used `torch.autograd.functional.jacobian()` (expensive for high dimensions)
- SLSQP enforced constraints exactly

**Complexity:**
- For 2592 particles, 5 nodes: 23,349 constraint equations
- Jacobian: 23,349 × 23,349 matrix (computational bottleneck)

---

#### Penalty Method (New Approach)

**Problem Formulation:**
```
minimize    L(x, u) = J(x, u) + λ * P(x, u)
```

where:
- **J(x, u)** = task cost (original objective)
- **P(x, u)** = penalty for constraint violations
- **λ** = penalty weight (large value → constraints satisfied)

**Penalty Terms:**
```python
# Initial state constraint penalty
P_init = ||x[0] - x_current||²

# Dynamics constraint penalties
P_dyn = Σ ||x[i+1] - f(x[i], u[i])||²  for i = 0, ..., N-1

# Total penalty
P(x, u) = P_init + P_dyn

# Combined loss
L(x, u) = J(x, u) + λ * P(x, u)
```

**Implementation:**
```python
def compute_loss(x_u, current_x, template_state):
    # Task cost
    task_cost = compute_stage_costs() + terminal_cost()
    
    # Constraint penalties
    constraint_penalty = 0.0
    
    # Initial constraint
    constraint_penalty += ||x[0] - current_x||²
    
    # Dynamics constraints (TRUE DMS)
    for i in range(N):
        state_i = set_state_from_vector(x[i], template_state)
        next_state_sim = simulate_single_step(state_i, u[i])
        constraint_penalty += ||x[i+1] - next_state_sim||²
    
    # Total loss
    return task_cost + penalty_weight * constraint_penalty
```

**Advantages:**
- ✓ **Simpler**: Only need first-order gradients (∇L)
- ✓ **Faster**: No Jacobian computation
- ✓ **Scalable**: Adam handles high-dimensional problems well
- ✓ **Flexible**: Easy to add/remove penalties

---

### 3. **Gradient Computation Comparison**

#### With Scipy SLSQP (Old)

**Cost Gradient:**
```python
def cost(x_u_np):
    x_u = torch.tensor(x_u_np, requires_grad=True)
    cost_value = compute_cost(x_u)
    cost_value.backward()
    return cost_value.item(), x_u.grad.numpy()
```

**Constraint Jacobian:**
```python
def eq_constraint(x_u_np):
    x_u = torch.tensor(x_u_np, requires_grad=True)
    
    # Full Jacobian matrix
    jacobian = torch.autograd.functional.jacobian(
        constraint_fn, x_u
    )  # Shape: (num_constraints, num_vars)
    
    return constraints.numpy(), jacobian.numpy()
```

**Issues:**
- Jacobian computation is **O(n²)** in memory
- Slow for high-dimensional state spaces

---

#### With PyTorch Adam (New)

**Combined Loss Gradient:**
```python
def compute_loss(x_u, current_x, template_state):
    task_cost = ...
    constraint_penalty = ...
    return task_cost + penalty_weight * constraint_penalty

# In optimization loop:
optimizer.zero_grad()
loss = compute_loss(x_u, ...)
loss.backward()  # ∇L computed automatically
optimizer.step()
```

**Advantages:**
- Only computes **∇L** (vector, not matrix)
- **O(n)** memory and computation
- PyTorch autograd handles everything automatically

---

## Optimization Algorithm

### Adam Optimizer

**Why Adam?**
1. **Adaptive learning rates**: Different learning rates for each parameter
2. **Momentum**: Smooths optimization trajectory
3. **Robust**: Works well for noisy/high-dimensional problems
4. **No Hessian**: Only uses first-order gradients

**Update Rule:**
```
m_t = β₁ * m_{t-1} + (1 - β₁) * ∇L      # First moment (momentum)
v_t = β₂ * v_{t-1} + (1 - β₂) * (∇L)²   # Second moment (adaptive lr)

x_u ← x_u - α * m_t / (√v_t + ε)       # Parameter update
```

**Default Hyperparameters:**
- Learning rate (α): 0.01
- β₁ = 0.9, β₂ = 0.999
- ε = 1e-8

---

## Implementation Details

### Key Functions

#### 1. `compute_loss(x_u, current_x, template_state)`
Computes combined loss = task cost + penalty.

**Inputs:**
- `x_u`: Decision variables [x₀, u₀, x₁, u₁, ..., x_{N-1}, u_{N-1}, x_N]
- `current_x`: Current state (for initial constraint)
- `template_state`: Full simulation state

**Returns:**
- `total_loss`: Scalar tensor to minimize

**Structure:**
```python
# Extract states and controls
x = extract_states(x_u)  # [x₀, x₁, ..., x_N]
u = extract_controls(x_u)  # [u₀, u₁, ..., u_{N-1}]

# Task cost (original objective)
task_cost = 0
for i in range(N):
    heights = extract_heights(x[i])
    task_cost += height_cost(heights) + control_cost(u[i])
task_cost += terminal_cost(x[N])

# Constraint penalties (TRUE DMS)
penalty = ||x[0] - current_x||²
for i in range(N):
    state_i = set_state(x[i])
    next_state_sim = simulate(state_i, u[i])
    penalty += ||x[i+1] - next_state_sim||²

return task_cost + penalty_weight * penalty
```

---

#### 2. `dms_plan(current_x, init_state)`
Plans optimal trajectory using Adam.

**Algorithm:**
```python
# Initialize decision variables
x_u = torch.nn.Parameter(warm_start(current_x))
optimizer = torch.optim.Adam([x_u], lr=learning_rate)

# Optimization loop
for iteration in range(max_iter):
    optimizer.zero_grad()
    
    # Compute loss (task + penalties)
    loss = compute_loss(x_u, current_x, init_state)
    
    # Backpropagation
    loss.backward()
    
    # Clip control values to bounds
    with torch.no_grad():
        clip_controls(x_u, u_min=-1, u_max=1)
    
    # Update parameters
    optimizer.step()
    
    # Track best solution
    if loss < best_loss:
        best_loss = loss
        best_x_u = x_u.clone()

# Extract first control (MPC receding horizon)
return best_x_u[state_dim:state_dim+control_dim]
```

---

### Hyperparameters

**New Parameters in Config:**
```yaml
dms_mpc_params:
  N: 5                        # Number of shooting nodes
  timesteps: 10               # MPC timesteps
  max_iter: 100               # Adam iterations per MPC step
  cost_state: 1.0             # State cost weight
  cost_control: 0.01          # Control cost weight
  cost_terminal: 10.0         # Terminal cost weight
  penalty_weight: 1000.0      # Constraint penalty weight (NEW)
  learning_rate: 0.01         # Adam learning rate (NEW)
```

**Tuning Guidelines:**

1. **`penalty_weight`** (λ):
   - Higher → constraints satisfied better, but optimization harder
   - Lower → optimization easier, but constraints may be violated
   - Typical range: 100 - 10,000
   - Start with 1000, increase if constraints violated

2. **`learning_rate`**:
   - Higher → faster convergence, but may overshoot
   - Lower → slower convergence, but more stable
   - Typical range: 0.001 - 0.1
   - Start with 0.01, decrease if unstable

---

## TRUE Direct Multiple Shooting

**Core Principle:** Each interval is optimized independently.

**For each shooting node i:**
```python
# 1. Set environment to optimizer's proposed state x[i]
state_i = set_state_from_vector(x[i], template_state)

# 2. Simulate one step with control u[i]
next_state_sim = simulate_single_step(state_i, u[i])

# 3. Penalty for mismatch with x[i+1]
penalty += ||x[i+1] - next_state_sim||²
```

**Key:** We do NOT carry forward `next_state_sim` to the next iteration. Each interval shoots from the optimizer's `x[i]`, not from simulation results.

---

## Removed Code

### Functions Deleted:
1. `cost(x_u)` → Replaced by `compute_loss(x_u, current_x, template_state)`
2. `eq_constraint_differentiable()` → No longer needed (penalty method)
3. `eq_constraint()` → No longer needed (penalty method)
4. All scipy-related code

### Variables Removed:
- `constraint_eval_count` (no explicit constraints)
- `cost_eval_count` (integrated into Adam loop)
- All numpy-based optimization structures

---

## Advantages of PyTorch Adam Approach

| Aspect | Scipy SLSQP | PyTorch Adam |
|--------|-------------|--------------|
| **Constraints** | Hard (exact) | Soft (penalty) |
| **Gradients** | Cost + Jacobian | Cost only |
| **Memory** | O(n²) (Jacobian) | O(n) (gradients) |
| **Complexity** | High (matrix ops) | Low (vector ops) |
| **Flexibility** | Rigid | Flexible |
| **Scalability** | Poor (high-dim) | Good (high-dim) |
| **Convergence** | Guaranteed* | Iterative |

*With sufficient iterations and feasible problem

---

## Usage Example

```python
# Initialize agent
agent = TrueDMSMPCAgent(config)

# MPC loop
for timestep in range(num_timesteps):
    # Get current state
    current_x = agent._obs_to_state()
    init_state = agent.clone_state(env.state_0)
    
    # Plan with Adam optimizer
    best_action = agent.dms_plan(current_x, init_state)
    
    # Execute action
    obs, reward, done, _ = env.step(best_action)
```

---

## Summary

**Before:** Scipy SLSQP with hard constraints and Jacobian matrices
**After:** PyTorch Adam with soft penalties and first-order gradients

**Benefits:**
- ✓ Simpler implementation (no Jacobian computation)
- ✓ Faster optimization (O(n) vs O(n²))
- ✓ More scalable (handles high dimensions)
- ✓ Pure PyTorch (no scipy dependency)
- ✓ Easier to tune (fewer hyperparameters)

**Trade-offs:**
- Constraints are satisfied approximately (not exactly)
- Requires tuning penalty_weight
- May need more iterations than SLSQP

**Bottom Line:** PyTorch Adam with penalty method is a practical, scalable alternative to constrained optimization for high-dimensional TRUE DMS MPC problems.

