# Quick Reference: Getting Gradients in TRUE DMS

## Question: How to get gradients of `full_state.mpm_x`?

### Answer: Use the new differentiable pipeline!

## Simple Example

```python
# Convert state to torch tensor WITH gradients
state_torch = self._obs_to_state(return_torch=True)
# state_torch now has gradients flowing from full_state.mpm_x

# Use in computations
state_torch.requires_grad_(True)  # Enable gradient tracking

# Perform operations
result = some_function(state_torch)

# Compute gradients
result.backward()

# Access gradients
gradients = state_torch.grad  # Gradients w.r.t. mpm_x
```

## In the Context of Optimization

The implementation automatically computes:

```python
# Equality constraint Jacobian: d(constraints)/d(x_u)
# where x_u contains mpm_x at each shooting node

constraints, jacobian = self.eq_constraint(x_u, current_x, template_state)
# jacobian shape: ((N+1)*state_dim, len(x_u))
# Each row: gradient of one constraint w.r.t. all decision variables
```

## Key Functions Modified

| Function | Input | Output | Gradients? |
|----------|-------|--------|-----------|
| `_obs_to_state()` | - | numpy/torch | Optional ✓ |
| `_obs_to_state(return_torch=True)` | - | torch.Tensor | Yes ✓ |
| `_set_state_from_vector(tensor, ...)` | torch.Tensor | state | Yes ✓ |
| `simulate_single_step(..., return_torch=True)` | torch.Tensor | torch.Tensor | Yes ✓ |
| `eq_constraint_differentiable(...)` | torch.Tensor | (constraints, jacobian) | Yes ✓ |

## Gradient Flow Diagram

```
Decision Variables (x_u)
        ↓
Extract mpm_x at node i
        ↓
Set simulator state ← mpm_x[i]
        ↓
Simulate one step (DIFFERENTIABLE!)
        ↓
Extract next mpm_x from simulator
        ↓
Compute constraint: mpm_x[i+1] - simulated_mpm_x
        ↓
Backpropagate to get: d(constraint)/d(x_u)
```

## What Makes This Work?

1. **Rewarped**: Differentiable physics simulator
2. **PyTorch autograd**: Automatic differentiation
3. **No `.detach()`**: Gradients preserved throughout
4. **Warp↔PyTorch**: `wp.to_torch()` maintains gradients

## Before vs After

### Before (No Gradients)
```python
mpm_x = wp.to_torch(full_state.mpm_x).cpu().detach().numpy()
# .detach() breaks gradient flow ✗
# .numpy() converts to non-differentiable array ✗
```

### After (With Gradients)
```python
mpm_x = wp.to_torch(full_state.mpm_x)
# Keeps gradients ✓
# Returns torch.Tensor ✓
# Can call .backward() ✓
```

## Testing Gradients

```python
# Test if gradients flow correctly
x_u = torch.randn(decision_var_size, requires_grad=True, device=self.device)
current_x = torch.randn(state_dim, device=self.device)

# Compute constraints (should have gradients)
constraints, jacobian = self.eq_constraint_differentiable(x_u, current_x, template_state)

# Check gradient computation worked
assert jacobian is not None
assert jacobian.shape == (len(constraints), len(x_u))
print(f"✓ Jacobian computed successfully: {jacobian.shape}")
```

## Common Issues

### Issue: "Gradients are None"
**Solution**: Check that `requires_grad=True` on input tensor

### Issue: "Can't call backward() twice"
**Solution**: Set `retain_graph=True` if needed, or recompute

### Issue: "Runtime error: element 0 of tensors does not require grad"
**Solution**: Ensure input has `requires_grad=True` before operations

## Performance Tips

1. **Batch operations**: Process multiple constraints simultaneously
2. **GPU acceleration**: Keep all tensors on GPU
3. **Gradient checkpointing**: If memory is an issue
4. **Smaller N**: Fewer shooting nodes = faster gradients

## Verification Checklist

- [ ] Rewarped environment initialized correctly
- [ ] `return_torch=True` used in differentiable path
- [ ] No `.detach()` calls in forward pass
- [ ] All tensors on same device
- [ ] Input tensor has `requires_grad=True`
- [ ] SLSQP receives both 'fun' and 'jac' in constraints

## Expected Console Output

When running optimization, you should see:

```
Starting TRUE DMS Optimization (with Rewarped Analytical Jacobians)
  Particles: 2592
  Nodes: 5, State dim: 7776, Control dim: 3
  Decision variables: 38895
  Constraints: 38880 equality constraints
  Using differentiable simulation for analytical gradients ✓
======================================================================

Constraint #1: max=1.2345e-02 rms=3.4567e-03 (with analytical Jacobian from Rewarped)
```

The key indicator: **(with analytical Jacobian from Rewarped)**

## Summary

You now have **analytical gradients** of constraints with respect to **all particle positions** at each shooting node. This is computed efficiently via backpropagation through Rewarped's differentiable physics simulator.

The gradient `∂c/∂(mpm_x)` tells you: "How does changing particle positions affect constraint satisfaction?"

SLSQP uses this to efficiently navigate the high-dimensional optimization landscape!

