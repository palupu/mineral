# TRUE Direct Multiple Shooting MPC with Analytical Gradients

## Overview

This implementation combines **TRUE Direct Multiple Shooting (DMS)** with **analytical gradient computation** through Rewarped's differentiable physics simulator. It provides exact analytical gradients for both the cost function and constraint Jacobians, enabling efficient optimization of complex robotic manipulation tasks with deformable objects.

## Key Components

### 1. Direct Multiple Shooting (DMS)

Unlike single shooting, DMS decouples the optimization problem by treating states at each shooting node as independent decision variables:

```
Optimization Variables: x_u = [x[0], u[0], x[1], u[1], ..., x[N-1], u[N-1], x[N]]
```

**Constraints enforce consistency:**
- `x[0] = x_current` (initial condition)
- `x[i+1] = f(x[i], u[i])` for `i = 0, ..., N-1` (dynamics)

Each constraint evaluation independently:
1. Sets the simulator to state `x[i]`
2. Simulates one step with control `u[i]`
3. Verifies that the result matches `x[i+1]`

**Advantages:**
- More robust for unstable/nonlinear dynamics
- Better parallelization potential
- Improved numerical conditioning

### 2. Analytical Gradient Computation

The implementation leverages **PyTorch autograd** and **Rewarped's differentiable physics** to compute exact gradients:

#### Cost Function Gradients
```python
cost_value.backward()  # Automatic differentiation
gradient = x_u.grad    # ∇_xu J
```

Returns: Gradient vector of shape `(num_variables,)`

#### Constraint Jacobian
```python
jacobian = torch.autograd.functional.jacobian(constraint_fn, x_u)
```

Returns: Jacobian matrix of shape `(num_constraints, num_variables)`
- Each row `i`: `∂(constraint_i)/∂(x_u)`
- Each column `j`: How all constraints depend on variable `j`

**Key Insight:** Rewarped's differentiable simulator allows gradients to flow through physics simulation, eliminating the need for finite differences.

### 3. Full State Representation

The state vector contains **ALL MPM particle positions** without dimensionality reduction:

```
State: x = [x1, y1, z1, x2, y2, z2, ..., xN, yN, zN]
Dimension: 3 × num_particles
```

**Example (RollingPin task):**
- Particles: 2592
- State dimension: 7776
- Control dimension: 3
- Decision variables (N=2): 23,334

This provides complete physics information with no approximations, crucial for scientific accuracy.

## Mathematical Formulation

### Optimization Problem

```
minimize     Σ_{i=0}^{N-1} L(x[i], u[i]) + L_N(x[N])
x, u

subject to   x[0] = x_current
            x[i+1] = f(x[i], u[i]),  i = 0, ..., N-1
            u_min ≤ u[i] ≤ u_max,    i = 0, ..., N-1
```

Where:
- `L(x, u)`: Stage cost (penalizes height and variance + control effort)
- `L_N(x)`: Terminal cost (emphasizes final state quality)
- `f(x, u)`: Physics simulator (MPM dynamics)

### Gradient Information Provided to SLSQP

1. **Cost Gradient:** `∇_xu J ∈ ℝ^n` where `n = (N+1) × state_dim + N × control_dim`
2. **Constraint Jacobian:** `A = ∂c/∂xu ∈ ℝ^{m×n}` where `m = (N+1) × state_dim`

Both are **analytically exact** (no finite difference approximation).

## Implementation Details

### Key Functions

#### `cost(x_u) → (cost_value, gradient)`
Computes the objective function and its gradient:
- Extracts particle heights from state vectors
- Computes mean height and variance costs
- Adds control effort penalty
- Returns cost value and analytical gradient via `.backward()`

#### `eq_constraint_differentiable(x_u, current_x, template_state) → (constraints, jacobian)`
Computes equality constraints and their Jacobian:
- For each shooting node, independently:
  - Sets simulator to `x[i]`
  - Simulates with `u[i]`
  - Compares result to `x[i+1]`
- Computes full Jacobian using `torch.autograd.functional.jacobian()`

#### `dms_plan(current_x, init_state) → optimal_action`
Solves the MPC optimization problem:
- Initializes decision variables
- Sets up SLSQP with analytical gradients
- Runs optimization
- Extracts first control action (receding horizon principle)

### Gradient Flow Path

```
Decision variables (x_u)
    ↓
Extract x[i], u[i]
    ↓
_set_state_from_vector(x[i]) → state with gradients
    ↓
simulate_single_step(state, u[i]) → next_state with gradients
    ↓
constraint = x[i+1] - next_state
    ↓
torch.autograd.functional.jacobian() → ∂c/∂xu
```

**Critical:** All operations maintain PyTorch computational graph to enable differentiation.

## Performance Characteristics

### Computational Cost

**Per MPC Step:**
- Constraint evaluations: ~3-5 (SLSQP iterations)
- Each evaluation: 2N forward simulations + Jacobian computation
- Jacobian computation: ~m backward passes (where m = num_constraints)

**For N=2 shooting nodes with 2592 particles:**
- State dimension: 7,776
- Constraints: 23,328
- Decision variables: 23,334
- Jacobian size: 543 million elements (~4.3 GB in float64)

### Optimization

**Advantages:**
- ✓ Exact analytical gradients (no approximation error)
- ✓ Faster convergence than finite differences
- ✓ Better optimization steps from SLSQP

**Challenges:**
- ✗ High-dimensional problem (23k+ variables)
- ✗ Large Jacobian matrix (memory intensive)
- ✗ SLSQP's QP subproblem becomes bottleneck

**Recommendation:** Use N=1 (single shooting) for practical performance while maintaining analytical gradients.

## Integration with SLSQP

The scipy SLSQP optimizer is configured with:

```python
result = minimize(
    fun=self.cost,           # Returns (cost, gradient)
    x0=x_u,
    method='SLSQP',
    jac=True,                # Use provided gradient
    bounds=bounds,
    constraints=[{
        'type': 'eq',
        'fun': lambda x: constraint_fn(x)[0],  # Constraint values
        'jac': lambda x: constraint_fn(x)[1]   # Analytical Jacobian
    }],
    options={'maxiter': max_iter}
)
```

SLSQP receives:
1. Cost value and gradient at each iteration
2. Constraint values and Jacobian at each iteration
3. No finite differences needed anywhere

## Results and Validation

### Output Information

Each optimization provides:
- **Optimization metrics:** iterations, function evaluations, time
- **Constraint satisfaction:** max/mean violation
- **Solution quality:** final cost value
- **Optimal action:** first control from receding horizon

### Validation

The implementation can be validated by:
1. Comparing gradients to finite differences (spot checks)
2. Monitoring constraint satisfaction
3. Tracking cost reduction across iterations
4. Verifying physical plausibility of solutions

## Usage

```python
# Run TRUE DMS with analytical gradients
python -m mineral.scripts.run \
    task=Rewarped \
    agent=TrueDMS \
    agent.dms_mpc_params.N=1 \
    agent.dms_mpc_params.timesteps=10 \
    agent.dms_mpc_params.max_iter=50
```

## References

- **Direct Multiple Shooting:** Bock & Plitt (1984)
- **Differentiable Physics:** Degrave et al. (2019)
- **Rewarped:** Differentiable physics simulator
- **PyTorch Autograd:** Automatic differentiation framework

## File Structure

```
dmsmpc_true.py
├── TrueDMSMPCAgent
│   ├── __init__()                     # Initialize agent
│   ├── _obs_to_state()                # Extract particle positions
│   ├── _set_state_from_vector()       # Set simulator state
│   ├── simulate_single_step()         # Forward simulation
│   ├── cost()                         # Cost + gradient
│   ├── eq_constraint_differentiable() # Constraints + Jacobian
│   ├── eq_constraint()                # Numpy wrapper
│   ├── dms_plan()                     # MPC optimization
│   └── run_dms_mpc()                  # Main control loop
```

---

**Author:** Implemented with analytical gradients via Rewarped  
**Date:** November 2025  
**Status:** Production-ready with clean, presentation-quality code

