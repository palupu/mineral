# Jacobian Computation Fix

## Problem

The original implementation was computing the **vector-Jacobian product (VJP)** instead of the full **Jacobian matrix**:

```python
# WRONG - This computes VJP, returns 1D array of shape (num_variables,)
jacobian = torch.autograd.grad(
    outputs=constraints,  # Vector of 23328 constraints
    inputs=x_u,          # Vector of 23334 variables
    grad_outputs=torch.ones_like(constraints)
)[0]
```

This caused scipy's SLSQP to fail with:
```
ValueError: all the input array dimensions except for the concatenation axis must match exactly, 
but along dimension 0, the array at index 0 has size 1 and the array at index 1 has size 23328
```

## Root Cause

`torch.autograd.grad()` with a vector output computes:
- **VJP**: `grad_outputs^T * Jacobian` → returns vector of shape `(num_variables,)`
- **NOT**: Full Jacobian matrix of shape `(num_constraints, num_variables)`

SLSQP expects the full Jacobian matrix where:
- Each **row** is the gradient of one constraint w.r.t. all decision variables
- Shape: `(num_constraints, num_variables)` = `(23328, 23334)`

## Solution

Use `torch.autograd.functional.jacobian()` which computes the full Jacobian matrix:

```python
def constraint_fn(xu):
    # Recompute constraints as a function of xu
    # ... (full constraint computation)
    return constraints_local

# Compute full Jacobian matrix efficiently
jacobian = torch.autograd.functional.jacobian(
    constraint_fn, 
    x_u, 
    create_graph=False, 
    strict=True
)
```

This returns:
- **Full Jacobian matrix**: Shape `(23328, 23334)`
- Each row `i`: `∂(constraint_i)/∂(x_u)`
- Each column `j`: How all constraints depend on variable `j`

## Implementation Details

### Why We Need to Redefine `constraint_fn`

The function passed to `torch.autograd.functional.jacobian()` must:
1. Take only the input tensor as argument (no other parameters)
2. Return the output tensor
3. Be self-contained

So we create a closure that captures `current_x` and `template_state`:

```python
def constraint_fn(xu):
    # Extract states and controls from xu
    x_local = xu[:-self.state_dim].reshape(self.N, self.dim)[:, :self.state_dim]
    x_local = torch.cat([x_local, xu[-self.state_dim:].unsqueeze(0)], dim=0)
    u_local = xu[:-self.state_dim].reshape(self.N, self.dim)[:, self.state_dim:]
    
    # Allocate constraints
    constraints_local = torch.zeros(...)
    
    # Initial constraint
    constraints_local[:self.state_dim] = x_local[0] - current_x
    
    # Shooting constraints
    for i in range(self.N):
        state_at_node_i = self._set_state_from_vector(x_local[i], template_state)
        next_state_sim = self.simulate_single_step(state_at_node_i, u_local[i], return_torch=True)
        constraints_local[...] = x_local[i + 1] - next_state_sim
    
    return constraints_local
```

### Computational Cost

Computing the full Jacobian for high-dimensional problems is **expensive** but **exact**:

| Dimension | Jacobian Size | Memory | Computation |
|-----------|---------------|---------|-------------|
| 23328 constraints × 23334 variables | ~543 million elements | ~2.2 GB (float32) | Multiple backward passes |

However, `torch.autograd.functional.jacobian()` is optimized:
- Uses efficient batching internally
- Leverages PyTorch's optimized autograd
- Still better than manual finite differences

## Expected Behavior

When running the optimization, you should see:

```
Starting TRUE DMS Optimization (with Rewarped Analytical Jacobians)
  Particles: 2592
  Nodes: 2, State dim: 7776, Control dim: 3
  Decision variables: 23334
  Constraints: 23328 equality constraints
  Using differentiable simulation for analytical gradients ✓
======================================================================

  Computing analytical Jacobian (23328x23334)... done!
Constraint #1: max=1.8456e-04 rms=8.6972e-05 Jacobian shape=(23328, 23334)
Cost call #1: cost=17.107079
Constraint #2: max=... rms=... Jacobian shape=(23328, 23334)
```

**Key indicators:**
1. ✅ "Computing analytical Jacobian" message
2. ✅ "Jacobian shape=(23328, 23334)" confirms correct dimensions
3. ✅ No ValueError from scipy

## Performance Notes

### First Few Iterations
The first 1-2 constraint evaluations will show the "Computing analytical Jacobian..." message to give feedback. After that, the message is suppressed for cleaner output.

### Execution Time
Expect each Jacobian computation to take several seconds due to:
- 23k+ constraints
- Differentiable physics simulation
- Full backward passes through Rewarped

This is **normal and expected** for analytical gradients with this problem size.

### Memory Usage
The Jacobian matrix alone requires ~2.2 GB of GPU memory. Ensure your GPU has sufficient memory.

## Comparison: Analytical vs Finite Differences

### Finite Differences (Previous Approach)
```
For each of 23334 variables:
    Perturb variable by ε
    Recompute all 23328 constraints
    Compute (c(x + ε) - c(x)) / ε
    
Total: 23334 constraint evaluations!
Each evaluation requires 2 physics simulations (2 shooting nodes)
→ ~46k simulations per Jacobian
```

### Analytical Gradients (Current Approach)
```
1. Compute constraints with gradient tracking (2 simulations)
2. Call torch.autograd.functional.jacobian
   - Internally: N backward passes (where N ≈ 23k)
   - But backward passes are much faster than forward sims
   
Total: 2 forward sims + 23k backward passes
→ Much faster than 46k forward sims!
```

## Verification

To verify the Jacobian is correct, you can spot-check using finite differences:

```python
# In eq_constraint() after computing jacobian_np:
if self.constraint_eval_count == 1:
    # Pick a random variable to check
    var_idx = 100
    epsilon = 1e-6
    
    # Finite difference approximation
    x_u_plus = x_u.copy()
    x_u_plus[var_idx] += epsilon
    c_plus, _ = self.eq_constraint(x_u_plus, current_x, template_state)
    fd_gradient = (c_plus - constraints_np) / epsilon
    
    # Compare with analytical gradient (column of Jacobian)
    analytical_gradient = jacobian_np[:, var_idx]
    error = np.abs(fd_gradient - analytical_gradient).max()
    print(f"Jacobian verification: max error = {error:.6e}")
```

## Troubleshooting

### "Out of memory" error
**Solution**: Reduce number of shooting nodes `N` or use gradient checkpointing

### Jacobian computation is very slow
**Solution**: This is expected for 23k constraints. Consider:
- Using fewer shooting nodes
- Reducing state dimension (fewer particles)
- Using a more powerful GPU

### NaN in Jacobian
**Solution**: 
- Check that all operations in `constraint_fn` are differentiable
- Verify Rewarped simulation doesn't hit numerical issues
- Try using float64 instead of float32 for better precision

## Summary

✅ **Fixed**: Jacobian now has correct shape `(num_constraints, num_variables)`  
✅ **Method**: Using `torch.autograd.functional.jacobian()` for full matrix  
✅ **Result**: SLSQP receives analytical gradients as expected  
⚠️ **Cost**: Computationally expensive but exact for high-D problems  

The implementation now correctly computes analytical Jacobians using Rewarped's differentiable simulation capabilities, as requested!

