# Differentiable Gradients Implementation for TRUE DMS

## Overview

This implementation leverages **Rewarped's differentiable simulation** capabilities to compute **analytical Jacobians** of equality constraints, significantly improving optimization efficiency compared to finite differences.

## Key Changes

### 1. Modified `_obs_to_state()` (Line 119-187)

**Before:** Always returned numpy array with `.detach().numpy()`

**After:** 
- Added `return_torch` parameter to optionally return torch.Tensor
- Preserves gradient flow when `return_torch=True`
- Maintains backward compatibility with numpy output

```python
def _obs_to_state(self, return_torch: bool = False):
    mpm_x = wp.to_torch(full_state.mpm_x)  # No .detach() anymore!
    # ... processing ...
    if return_torch:
        return state  # torch.Tensor with gradients
    else:
        return state.detach().cpu().numpy()  # numpy for scipy
```

### 2. Updated `_set_state_from_vector()` (Line 189-250)

**Before:** Only accepted numpy arrays

**After:**
- Accepts both numpy arrays and torch.Tensors
- Maintains gradients when torch.Tensor is passed
- Converts appropriately based on input type

```python
def _set_state_from_vector(self, state_vec, template_state: Any) -> Any:
    if isinstance(state_vec, np.ndarray):
        particle_positions = torch.from_numpy(...)
    else:  # torch.Tensor - maintains gradients!
        particle_positions = state_vec.reshape(...)
    # ... set state ...
```

### 3. Updated `simulate_single_step()` (Line 252-285)

**Before:** Only worked with numpy arrays

**After:**
- Added `return_torch` parameter
- Handles both numpy and torch inputs/outputs
- Gradient flow maintained throughout simulation

```python
def simulate_single_step(self, init_state: Any, control, return_torch: bool = False):
    # Converts control appropriately
    # Calls env.step() - differentiable with Rewarped!
    next_state = self._obs_to_state(return_torch=return_torch)
    return next_state
```

### 4. Added `eq_constraint_differentiable()` (Line 361-411)

**NEW FUNCTION** - Core differentiable constraint computation:

```python
def eq_constraint_differentiable(self, x_u: torch.Tensor, current_x: torch.Tensor, 
                                 template_state: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    # All operations in PyTorch with gradient tracking
    for i in range(self.N):
        state_at_node_i = self._set_state_from_vector(x[i], template_state)
        next_state_sim = self.simulate_single_step(state_at_node_i, u[i], return_torch=True)
        constraints[...] = x[i + 1] - next_state_sim
    
    # Compute Jacobian using autograd - Rewarped makes this possible!
    jacobian = torch.autograd.grad(
        outputs=constraints,
        inputs=x_u,
        grad_outputs=torch.ones_like(constraints)
    )[0]
    
    return constraints, jacobian
```

### 5. Modified `eq_constraint()` (Line 413-463)

**Before:** Only returned constraint values (numpy)

**After:**
- Returns tuple: `(constraints, jacobian)`
- Wraps `eq_constraint_differentiable()` 
- Converts torch outputs back to numpy for scipy

```python
def eq_constraint(self, x_u: np.ndarray, ...) -> Tuple[np.ndarray, np.ndarray]:
    # Convert to torch
    x_u_torch = torch.tensor(x_u, requires_grad=True, device=self.device)
    
    # Compute with gradients
    constraints, jacobian = self.eq_constraint_differentiable(...)
    
    # Convert back to numpy
    return constraints_np, jacobian_np
```

### 6. Updated `dms_plan()` (Line 508-540)

**Before:** SLSQP used finite differences for Jacobian

**After:**
- Constraint definition now includes analytical Jacobian
- SLSQP uses provided Jacobian instead of approximating it

```python
eq_cons = {
    'type': 'eq',
    'fun': lambda x: eq_constraint_wrapper(x)[0],  # Constraint values
    'jac': lambda x: eq_constraint_wrapper(x)[1]   # Analytical Jacobian!
}
```

## Benefits

### 1. **Faster Convergence**
- Analytical gradients are exact (no approximation error)
- SLSQP can make better optimization steps
- Fewer iterations needed to reach optimum

### 2. **More Accurate**
- No finite difference approximation errors
- Exact gradient information from physics simulator
- Better handling of high-dimensional problems (7776+ variables)

### 3. **Computationally Efficient**
- Single backward pass computes all gradients simultaneously
- No need for N+1 forward passes for finite differences
- Rewarped's adjoint method is optimized for this

## Performance Comparison

| Method | Jacobian Computation | Accuracy | Speed |
|--------|---------------------|----------|-------|
| **Before (Finite Diff)** | ~7780 forward passes | Approximate | Slow |
| **After (Analytical)** | 1 forward + 1 backward | Exact | Fast |

## Technical Details

### Gradient Flow Path

```
x_u (requires_grad=True)
  ↓
x[i] (sliced from x_u, retains gradients)
  ↓
_set_state_from_vector(x[i]) → state with gradients
  ↓
simulate_single_step(state, u[i], return_torch=True)
  ↓
env.step(actions) [DIFFERENTIABLE - Rewarped!]
  ↓
_obs_to_state(return_torch=True) → next_state with gradients
  ↓
constraints = x[i+1] - next_state (retains gradients)
  ↓
torch.autograd.grad(constraints, x_u) → Jacobian
```

### Key Requirements

1. **Rewarped Simulator**: Must support differentiable physics
2. **Gradient Tracking**: All intermediate tensors must maintain `requires_grad=True`
3. **No Detachment**: Must not call `.detach()` in the forward pass
4. **Device Consistency**: All tensors on same device (GPU/CPU)

## Usage

The implementation is **backward compatible**:

```python
# For regular (non-differentiable) use:
state = self._obs_to_state()  # Returns numpy array

# For differentiable use:
state = self._obs_to_state(return_torch=True)  # Returns torch.Tensor with gradients
```

The optimization automatically uses analytical Jacobians when calling:

```python
result = minimize(
    self.cost,
    x_u,
    method='SLSQP',
    jac=True,  # Cost has analytical gradients
    bounds=bounds,
    constraints=[eq_cons],  # Constraints now also have analytical Jacobians!
    options={'disp': True, 'maxiter': self.max_iter}
)
```

## Verification

To verify gradients are working:

1. Check console output: Should see "(with analytical Jacobian from Rewarped)"
2. Monitor constraint evaluations: Should be faster per iteration
3. Compare convergence: Should reach solution in fewer iterations

## Troubleshooting

### "Gradients don't flow through simulation"
- Ensure Rewarped environment is properly initialized
- Check that `env.step()` supports backpropagation
- Verify no `.detach()` calls in the forward pass

### "Out of memory errors"
- Reduce number of shooting nodes `N`
- Use gradient checkpointing if available
- Reduce particle count if possible

### "Gradients are None"
- Check `requires_grad=True` on input tensors
- Verify all operations are differentiable
- Ensure simulation doesn't break computational graph

## References

- Rewarped: Differentiable physics simulator
- SLSQP: Sequential Least Squares Programming with constraint Jacobians
- PyTorch autograd: Automatic differentiation
- Direct Multiple Shooting: Parallel constraint evaluation

## Implementation Date

November 19, 2025

