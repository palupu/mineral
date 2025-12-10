# DMS-MPC with Full State Implementation

## Overview

`dmsmpc2.py` implements Direct Multiple Shooting Model Predictive Control using **real physics simulation** instead of analytical dynamics models. This provides significantly better accuracy for complex physics scenarios like MPM-based deformable objects.

## Key Components

### 1. State Cloning (`clone_state()`)
```python
def clone_state(self, state):
    # Clones all Warp arrays (GPU data) and Python objects
    # Required to save/restore full simulation state
```

**Why needed:** To preserve complete physics state (particles, deformations, stresses) for trajectory rollouts.

### 2. Physics Simulation (`simulate_single_step()`, `simulate_trajectory()`)
```python
def simulate_single_step(self, init_state, control):
    # Clones state, applies control, runs physics step
    # Returns next observed state
```

**Replaces:** Analytical dynamics model (`rk4_step()` with `dynamics()`)

### 3. Constraint Function (`eq_constraint()`)
```python
def eq_constraint(self, x_u, current_x, init_state):
    # Ensures trajectory continuity using real physics
    # x[i+1] = f_physics(x[i], u[i])
```

**Key insight:** Simulates forward with real physics engine instead of numerical integration of analytical model.

### 4. Planning Loop (`dms_plan()`)
```python
def dms_plan(self, current_x, init_state):
    # Saves state
    # Optimizes trajectory with physics constraints
    # Restores state
    # Returns first optimal action
```

## Comparison: Analytical vs Full State

| Aspect | Analytical DMS-MPC | Full State DMS-MPC |
|--------|-------------------|-------------------|
| **Dynamics** | Simplified model: `dx/dt = u` | Real physics simulator |
| **State Required** | Observations only (6 values) | Full state (all MPM data) |
| **Accuracy** | Poor for complex physics | High accuracy |
| **Computational Cost** | Low | High (N × physics steps) |
| **Best For** | Simple dynamics (cart-pole) | Complex physics (deformables) |

## How It Works

### Main Loop (`run_dms_mpc()`)
```
For each timestep:
  1. Save current state: init_state = clone_state()
  2. Plan: best_action = dms_plan(obs, init_state)
     - Optimization tries different control sequences
     - Each evaluation runs real physics N times
     - Constraints ensure trajectory consistency
  3. Restore: env.state = init_state
  4. Execute: step(best_action)
```

### Optimization Process
```
Decision variables: [x0, u0, x1, u1, ..., xN-1, uN-1, xN]
  x_i: intermediate states
  u_i: control inputs

Constraints:
  - x0 = current_state (initial condition)
  - x_{i+1} = simulate(x_i, u_i) for all i (continuity)

Objective:
  min: Σ (x_i^T Q x_i + u_i^T R u_i) + x_N^T Q_N x_N
```

## Advantages Over CEM-MPC

1. **Gradient-based optimization** (SLSQP) vs sampling-based (CEM)
2. **Deterministic** - same inputs → same outputs
3. **Potentially fewer physics evaluations** if optimizer converges quickly
4. **Explicit cost function** - easier to tune behavior

## Disadvantages

1. **Harder to optimize** - non-convex problem with physics constraints
2. **Requires constraint satisfaction** - can fail to find feasible solution
3. **Slower per iteration** - gradient computation overhead
4. **Limited to differentiable costs** (though we use numerical gradients)

## Configuration

Key parameters in config YAML:
```yaml
dms_mpc_params:
  H: 10           # Horizon length (not used - N is used instead)
  N: 5            # Number of shooting nodes
  dt: 0.033       # Time step (simulation dependent)
  max_iter: 100   # Max optimization iterations
  
  # Cost weights
  cost_state: 0.1
  cost_control: 0.01
  cost_terminal: 1.0
  
  # State/control dimensions
  state_dim: 6    # [joint_q(3), com_q(3)]
  control_dim: 3  # [dx, dy, ry]
```

## Usage

```python
# In your config, specify:
agent: DMSMPCAgent

# Run:
python train.py agent=DMSMPC task=Rewarped
```

## Implementation Notes

### Limitation: Simplified Multiple Shooting
True multiple shooting allows "jumping" to arbitrary intermediate states `x_i`. However, we can't directly set full physics state from observations alone. 

**Current approach:** Sequential simulation from initial state
- Pro: Simpler, works with observation-based planning
- Con: Not true multiple shooting (more like single shooting)

**Future improvement:** Implement full state reconstruction from observations, or use full state as decision variables.

### Performance Considerations
- Each constraint evaluation calls `N` physics steps
- Each optimization iteration evaluates constraints multiple times
- Total physics calls per timestep: ~100-1000 depending on convergence

Expect runtime to be **10-100x longer than CEM-MPC** depending on problem complexity.

## Example Output
```
Timestep 1 | Action: tensor([0.1234, -0.0567, 0.0123]) | Reward: 0.456
Optimization success: True, message: Optimization terminated successfully
Final cost: 1.234, iterations: 23
```

## Debugging Tips

1. **Optimizer not converging:** 
   - Increase `max_iter`
   - Adjust cost weights (larger `cost_control` = smoother actions)
   - Check constraint feasibility

2. **Actions too conservative:**
   - Decrease `cost_control`
   - Increase `cost_state`

3. **Too slow:**
   - Reduce `N` (fewer shooting nodes)
   - Reduce `max_iter`
   - Consider switching to CEM-MPC

4. **Constraint violations:**
   - Check that `dt` matches environment step size
   - Verify state extraction in `_obs_to_state()`

