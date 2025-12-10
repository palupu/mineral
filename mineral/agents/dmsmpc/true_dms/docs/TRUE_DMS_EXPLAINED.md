# TRUE Direct Multiple Shooting (DMS) Implementation

## Overview

This document explains the difference between **pseudo-DMS** (in `dmsmpc.py`) and **TRUE DMS** (in `dmsmpc_true.py`), and why true DMS is more accurate (but also more expensive).

---

## The Key Difference

### Pseudo-DMS (dmsmpc.py)

```python
def eq_constraint(self, x_u, current_x, init_state):
    self.env.state_0 = init_state  # Start from initial state
    
    for i in range(self.N):
        # Simulate from CURRENT environment state (sequential!)
        next_state_sim = self.simulate_single_step(self.env.state_0, u[i])
        constraints.extend(x[i + 1] - next_state_sim)
        # Environment state continues forward (env.state_0 updated by simulate_single_step)
```

**Problem:** Each shooting interval depends on the previous one! It's actually **single shooting** with extra state variables.

```
Initial state ──u[0]──> sim(state) = x[1]? ──u[1]──> sim(x[1]) = x[2]? ──u[2]──> ...
                   ↑ Sequential!
```

### TRUE DMS (dmsmpc_true.py)

```python
def eq_constraint(self, x_u, current_x, template_state):
    for i in range(self.N):
        # 1. Set environment to optimizer's proposed state x[i]
        state_at_node_i = self._set_state_from_vector(x[i], template_state)
        
        # 2. Simulate INDEPENDENTLY from x[i]
        next_state_sim = self.simulate_single_step(state_at_node_i, u[i])
        
        # 3. Constraint: must reach x[i+1]
        constraints.extend(x[i + 1] - next_state_sim)
        
        # CRITICAL: Don't use sim result for next iteration!
        # Next iteration independently sets to x[i+1]
```

**Key insight:** Each interval shoots **independently** from the optimizer's proposed state!

```
x[0] ──u[0]──> sim(x[0]) = x[1]?    (independent!)
x[1] ──u[1]──> sim(x[1]) = x[2]?    (independent!)
x[2] ──u[2]──> sim(x[2]) = x[3]?    (independent!)
```

---

## Visual Comparison

### Pseudo-DMS (Sequential)
```
┌─────────────────────────────────────────────────────┐
│ Optimizer proposes: [x₀, u₀, x₁, u₁, x₂, u₂, x₃]   │
└─────────────────────────────────────────────────────┘
                     ↓
            Constraint evaluation:
            
    init_state ──u₀──> sim_result₁
                         ↓ (continues from sim_result₁)
                  sim_result₁ ──u₁──> sim_result₂
                                       ↓ (continues)
                                sim_result₂ ──u₂──> sim_result₃
                                
    Constraints:
    - x₁ == sim_result₁?
    - x₂ == sim_result₂?
    - x₃ == sim_result₃?
```

### TRUE DMS (Independent)
```
┌─────────────────────────────────────────────────────┐
│ Optimizer proposes: [x₀, u₀, x₁, u₁, x₂, u₂, x₃]   │
└─────────────────────────────────────────────────────┘
                     ↓
            Constraint evaluation:
            
    SET(x₀) ──u₀──> sim_result₁    ┐
                                    │ Independent
    SET(x₁) ──u₁──> sim_result₂    │ evaluations!
                                    │
    SET(x₂) ──u₂──> sim_result₃    ┘
                                
    Constraints:
    - x₁ == sim_result₁?
    - x₂ == sim_result₂?
    - x₃ == sim_result₃?
```

---

## The Critical Function: `_set_state_from_vector()`

This is what enables true DMS. It converts the optimizer's proposed reduced state vector (e.g., 6D: `[joint_q, com_q]`) into a full simulation state (with thousands of MPM particles).

### Strategy 1: Joint Only (Recommended)

```python
state_setting_strategy: 'joint_only'
```

**What it does:**
- Sets joint positions from state vector
- Ignores COM (center of mass) component
- Lets COM be determined by physics

**Pros:**
- ✅ Simple and reliable
- ✅ Fast (no particle updates)
- ✅ Stable

**Cons:**
- ⚠️ Can't directly control COM position
- ⚠️ Effective state dimension reduced (3D instead of 6D)

**When to use:** Default choice for most tasks

### Strategy 2: Joint + COM (Experimental)

```python
state_setting_strategy: 'joint_com'
```

**What it does:**
- Sets joint positions from state vector
- Translates ALL MPM particles to match desired COM

**Pros:**
- ✅ Full control over state (both joints and COM)
- ✅ True 6D state space

**Cons:**
- ⚠️ Expensive (updates thousands of particles)
- ⚠️ May be numerically unstable
- ⚠️ Translating particles may violate physics constraints

**When to use:** When COM control is critical and you accept the computational cost

---

## Configuration

### Config File (`cfg.yaml`)

```yaml
agent:
  dms_mpc_params:
    N: 10                    # Number of shooting nodes
    timesteps: 50            # MPC steps
    max_iter: 20             # SLSQP iterations per MPC step
    
    state_dim: 6             # [joint_q (3), com_q (3)]
    control_dim: 3           # [dx, dy, ry]
    
    # Cost weights
    cost_state: 1.0
    cost_control: 0.01
    cost_terminal: 10.0
    
    # State setting strategy
    state_setting_strategy: 'joint_only'  # or 'joint_com'
```

### Python Usage

```python
from mineral.agents.dmsmpc.dmsmpc_true import TrueDMSMPCAgent

# Create agent with config
agent = TrueDMSMPCAgent(cfg)

# Run evaluation
agent.eval()

# Or replay saved trajectory
agent.render_results = True
agent.eval()
```

---

## Performance Comparison

### Computational Cost

| Aspect | Pseudo-DMS | TRUE DMS |
|--------|-----------|----------|
| **Decision variables** | N × (state + control) + state | N × (state + control) + state |
| **Constraints** | (N+1) × state_dim | (N+1) × state_dim |
| **Simulations per constraint eval** | N (sequential) | N (independent) |
| **State setting overhead** | None | `_set_state_from_vector()` × N |
| **Parallelizable?** | No | Yes (but not implemented) |

**Example:** N=10, state_dim=6, control_dim=3
- Decision variables: 10 × (6 + 3) + 6 = **96 variables**
- Constraints: 11 × 6 = **66 equality constraints**
- Simulations per SLSQP iteration: ~10-20 constraint evaluations × N = **100-200 simulations**

### Expected Performance

#### TRUE DMS is Better When:
1. ✅ **Unstable dynamics**: Long shooting intervals would diverge
2. ✅ **Nonlinear dynamics**: Single shooting accumulates errors
3. ✅ **Need parallelization**: Independent intervals can be simulated in parallel
4. ✅ **Warm-starting from poor guess**: DMS is more robust to initialization

#### Single Shooting is Better When:
1. ✅ **Stable dynamics**: MPM physics is generally well-behaved
2. ✅ **Computational cost matters**: Single shooting is much faster
3. ✅ **With autodiff**: Exact gradients reduce need for DMS benefits
4. ✅ **Short horizons**: Less error accumulation in single shooting

---

## Comparison with Other Implementations

### Summary Table

| File | Type | Decision Vars | Constraint Type | State Setting |
|------|------|---------------|-----------------|---------------|
| `dmsmpc0.py` | Analytical DMS | states + controls | RK4 integration | N/A (analytical) |
| `dmsmpc.py` | **Pseudo-DMS** | states + controls | Sequential physics | N/A (sequential) |
| `dmsmpc_true.py` | **TRUE DMS** | states + controls | Independent physics | `_set_state_from_vector()` |
| `dmsmpc_scipy_autodiff.py` | Single Shooting | controls only | None (implicit) | N/A (sequential) |
| `dmsmpc_differentiable.py` | Single Shooting | controls only | None (implicit) | N/A (sequential) |

### Recommended Choice

For **most use cases with differentiable physics**:
```python
# Best: Single shooting with autodiff (dmsmpc_scipy_autodiff.py)
# - Fastest
# - Exact gradients
# - Sufficient for stable dynamics
```

For **highly unstable/nonlinear dynamics**:
```python
# Use: True DMS (dmsmpc_true.py)
# - More robust
# - Better convergence
# - Worth the computational cost
```

---

## Debugging and Monitoring

### Performance Tracking

The implementation tracks:
- `constraint_eval_count`: Number of times constraints were evaluated
- Optimization time per MPC step
- SLSQP iterations and function evaluations

### Typical Output

```
══════════════════════════════════════════════════════════════════════
Starting TRUE DMS Optimization
  Strategy: joint_only
  Nodes: 10, State dim: 6, Control dim: 3
  Decision variables: 96
  Constraints: 66 equality constraints
══════════════════════════════════════════════════════════════════════

[SLSQP iterations...]

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

### Common Issues

#### 1. Constraint violations don't decrease
**Symptom:** SLSQP reports "Positive directional derivative for linesearch"

**Cause:** `_set_state_from_vector()` might not be setting state correctly

**Solution:** 
- Switch to `joint_only` strategy
- Check that joint_q dimension matches expected (3 for RollingPin)
- Verify warp array assignments are on correct device

#### 2. Very slow convergence
**Symptom:** Each MPC step takes 100+ seconds

**Cause:** Too many constraint evaluations (each requires N simulations)

**Solution:**
- Reduce `N` (horizon length)
- Reduce `max_iter` (SLSQP iterations)
- Consider switching to single shooting instead

#### 3. Optimal solution is same as initial guess
**Symptom:** Action doesn't change from initialization

**Cause:** Optimizer unable to improve due to poor gradients or infeasible constraints

**Solution:**
- Check constraint feasibility: Are they satisfiable?
- Verify cost function gradients (should not be all zeros)
- Try different initialization (add small random noise)

---

## Theoretical Background

### Why Multiple Shooting?

Single shooting solves:
```
minimize   J(u)
subject to x_{k+1} = f(x_k, u_k), k=0,...,N-1
           x_0 = x_init (implicit)
```

Direct Multiple Shooting solves:
```
minimize   J(x, u)
subject to x_{k+1} = f(x_k, u_k), k=0,...,N-1  (explicit constraints!)
           x_0 = x_init
```

**Benefits:**
1. **Robustness**: Constraints can be violated during optimization, giving optimizer more freedom
2. **Parallelization**: Each interval f(x_k, u_k) can be evaluated independently
3. **Convergence**: Better conditioning for unstable/stiff dynamics

**Cost:**
- More variables: N × state_dim additional variables
- More constraints: N × state_dim equality constraints
- More evaluations: N independent simulations per constraint check

---

## Future Improvements

Potential enhancements:

1. **Parallel constraint evaluation**
   - Use multiprocessing to evaluate N shooting intervals simultaneously
   - Requires CUDA-aware process management or separate GPU contexts

2. **Inequality constraints on states**
   - Add state bounds: `x_min ≤ x_k ≤ x_max`
   - Useful for safety constraints (e.g., joint limits, collision avoidance)

3. **Adaptive horizon**
   - Start with small N, increase if convergence is poor
   - Reduce N if dynamics are stable

4. **Better warm-starting**
   - Use previous MPC solution shifted by 1 timestep
   - Currently: initialize all states to current_x (naive)

5. **Trust region for states**
   - Add soft constraints: states shouldn't deviate too far from simulation
   - Helps when `_set_state_from_vector()` is imperfect

---

## References

- Bock, H. G., & Plitt, K. J. (1984). "A multiple shooting algorithm for direct solution of optimal control problems." IFAC Proceedings Volumes
- Diehl, M., et al. (2006). "Fast Direct Multiple Shooting Algorithms for Optimal Robot Control." Fast Motions in Biomechanics and Robotics
- Wieber, P. B., et al. (2016). "Modeling and Control of Legged Robots." Springer Handbook of Robotics

---

## Summary

**TRUE DMS = Independent shooting nodes + State setting capability**

Key components:
1. ✅ `_set_state_from_vector()`: Set arbitrary states in simulator
2. ✅ Independent constraint evaluation: Each node shoots separately  
3. ✅ Proper optimization structure: States and controls as decision variables

Use when:
- Dynamics are unstable or highly nonlinear
- Need robustness over speed
- Willing to pay computational cost

Stick with single shooting when:
- Physics is stable (like MPM usually is)
- Have autodiff (exact gradients compensate for single shooting weakness)
- Want fast iteration

