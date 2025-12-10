# Quick Reference: Gradient-Based Optimization for DMS MPC

## Short Answer to Your Question

**Q: What does "gradient-based optimization" mean? Can I use it with NLP and scipy?**

**A:** 
- **Gradient-based** = uses derivatives (∇J) to optimize. This includes scipy!
- **Yes**, you can absolutely use scipy with NLP (Nonlinear Programming)
- **Even better**: You can use scipy NLP solvers WITH automatic differentiation

---

## Visual Summary

```
┌─────────────────────────────────────────────────────────────┐
│                  GRADIENT-BASED OPTIMIZATION                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │  Scipy Solvers   │  │ PyTorch Optim    │                │
│  ├──────────────────┤  ├──────────────────┤                │
│  │ • L-BFGS-B       │  │ • Adam           │                │
│  │ • SLSQP (NLP)    │  │ • RMSprop        │                │
│  │ • trust-constr   │  │ • SGD            │                │
│  │ • TNC            │  │ • L-BFGS         │                │
│  └──────────────────┘  └──────────────────┘                │
│           │                      │                          │
│           └──────────┬───────────┘                          │
│                      │                                      │
│              Both use gradients!                            │
│                      │                                      │
│           ┌──────────┴───────────┐                          │
│           │                      │                          │
│  ┌────────▼────────┐  ┌──────────▼─────────┐               │
│  │ Finite Diff     │  │ Automatic Diff     │               │
│  │ (numerical)     │  │ (PyTorch/JAX)      │               │
│  ├─────────────────┤  ├────────────────────┤               │
│  │ • Slow          │  │ • Fast             │               │
│  │ • Approximate   │  │ • Exact            │               │
│  │ • Simple        │  │ • Powerful         │               │
│  └─────────────────┘  └────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Three Ways to Implement DMS MPC

### 1️⃣ Scipy with Finite Differences (Your Original)

```python
def cost(x_u):
    # x_u contains states AND controls
    cost = compute_cost(x_u)
    return cost

def constraints(x_u):
    # Enforce x[i+1] = f(x[i], u[i])
    return dynamics_violations(x_u)

result = minimize(
    cost, 
    x0,
    method='SLSQP',          # NLP solver
    constraints=constraints   # Dynamics as constraints
)
# Scipy uses finite differences for gradients
```

**Pros:** Standard approach, well understood  
**Cons:** Slow gradients, many variables, needs dynamics model

---

### 2️⃣ Scipy with Autodiff Gradients (RECOMMENDED) ⭐

```python
def cost_and_grad(u):  # Only controls!
    # Convert to torch
    controls = torch.tensor(u, requires_grad=True)
    
    # Rollout with differentiable physics
    cost = 0.0
    for i in range(N):
        obs = env.step(controls[i])  # Autodiff through physics!
        cost += compute_stage_cost(obs, controls[i])
    
    # Get gradient automatically
    cost.backward()
    
    return cost.item(), controls.grad.numpy()

result = minimize(
    cost_and_grad,
    x0,
    method='L-BFGS-B',    # Scipy optimizer
    jac=True              # We provide exact gradients!
)
```

**Pros:** Fast, robust, exact gradients, fewer variables  
**Cons:** Needs differentiable simulator

---

### 3️⃣ PyTorch Optimizers with Autodiff

```python
controls = torch.zeros(N, control_dim, requires_grad=True)
optimizer = torch.optim.Adam([controls], lr=0.05)

for iteration in range(100):
    optimizer.zero_grad()
    
    # Rollout with differentiable physics
    cost = 0.0
    for i in range(N):
        obs = env.step(controls[i])
        cost += compute_stage_cost(obs, controls[i])
    
    cost.backward()
    optimizer.step()

optimal_action = controls[0].detach()
```

**Pros:** Simple, flexible, good for noisy gradients  
**Cons:** May need more iterations than L-BFGS-B

---

## Decision Tree: Which Method?

```
Do you have a differentiable simulator (Rewarped)?
│
├─ NO  → Use Approach 1 (Finite Diff + SLSQP)
│        - Slow but works without gradients
│
└─ YES → Do you need equality/inequality constraints?
         │
         ├─ YES → Use Approach 2 with SLSQP
         │        scipy.minimize(..., method='SLSQP', jac=True)
         │
         └─ NO (only bounds) → Are gradients noisy/non-smooth?
                              │
                              ├─ YES → Use Approach 3 (Adam)
                              │        torch.optim.Adam()
                              │
                              └─ NO  → Use Approach 2 with L-BFGS-B ⭐
                                       scipy.minimize(..., method='L-BFGS-B', jac=True)
```

---

## Key Insights

### 1. "Gradient-based" is a broad category

```
Gradient-Based Methods
├── Scipy optimizers (L-BFGS-B, SLSQP, etc.) ✅
├── PyTorch optimizers (Adam, SGD, etc.) ✅
├── JAX optimizers ✅
└── Custom gradient descent ✅

NOT Gradient-Based:
├── CEM (Cross-Entropy Method) ❌
├── Random Search ❌
├── Genetic Algorithms ❌
```

### 2. Scipy IS an NLP solver

**NLP** = Nonlinear Programming = Constrained optimization with nonlinear objective/constraints

Scipy methods like **SLSQP** and **trust-constr** are full-featured NLP solvers that handle:
- Nonlinear objectives
- Equality constraints
- Inequality constraints
- Variable bounds

### 3. You can mix and match!

| Optimizer | Gradient Source | Result |
|-----------|----------------|--------|
| Scipy SLSQP | Finite Diff | Traditional (slow) |
| Scipy SLSQP | Autodiff | Hybrid (fast) ⭐ |
| Scipy L-BFGS-B | Autodiff | Hybrid (fastest) ⭐ |
| PyTorch Adam | Autodiff | Modern (robust) |

**Best combination:** Scipy L-BFGS-B + PyTorch Autodiff

---

## Example: All Three Approaches

### Problem Setup
```python
# DMS MPC for N=10 horizon, 3D control
N = 10
control_dim = 3
```

### Approach 1: Traditional
```python
# Variables: states + controls = 9*10 + 6 = 96 dimensions
x_u = np.zeros(96)

result = minimize(
    cost_with_constraints,
    x_u,
    method='SLSQP',
    constraints={'type': 'eq', 'fun': dynamics_constraints}
)
# Time: ~5 seconds per MPC step
```

### Approach 2: Scipy + Autodiff ⭐
```python
# Variables: controls only = 3*10 = 30 dimensions
u = np.zeros(30)

def cost_and_grad(u):
    controls = torch.tensor(u, requires_grad=True)
    cost = differentiable_rollout(controls)
    cost.backward()
    return cost.item(), controls.grad.numpy()

result = minimize(
    cost_and_grad,
    u,
    method='L-BFGS-B',
    jac=True,
    bounds=[(-1, 1)] * 30
)
# Time: ~0.5 seconds per MPC step (10× faster!)
```

### Approach 3: PyTorch
```python
# Variables: controls = 3*10 = 30 dimensions
controls = torch.zeros(10, 3, requires_grad=True)
optimizer = torch.optim.Adam([controls], lr=0.05)

for _ in range(50):
    optimizer.zero_grad()
    cost = differentiable_rollout(controls)
    cost.backward()
    optimizer.step()

# Time: ~1 second per MPC step
```

---

## Recommended Reading Order

1. **GRADIENT_METHODS_EXPLAINED.md** - Full taxonomy of methods
2. **DIFFERENTIABLE_MPC_EXPLAINED.md** - How autodiff transforms MPC
3. **This file** - Quick reference

## Recommended Implementation Order

1. **Start here:** `dmsmpc_scipy_autodiff.py` with L-BFGS-B
2. **If unstable:** Try `dmsmpc_differentiable.py` with Adam
3. **For comparison:** Run all three and compare results

---

## Summary Table

| Feature | Scipy + Finite Diff | Scipy + Autodiff ⭐ | PyTorch + Autodiff |
|---------|-------------------|-------------------|-------------------|
| **Gradient Method** | Numerical | Automatic | Automatic |
| **Optimizer** | SLSQP | L-BFGS-B/SLSQP | Adam/L-BFGS |
| **Variables** | States + Controls | Controls | Controls |
| **Constraints** | Yes (equality/ineq) | Bounds (+ eq/ineq with SLSQP) | Bounds only |
| **Speed** | Slow | Fast | Fast |
| **Robustness** | High | High | Medium |
| **Convergence** | Good | Excellent | Good |
| **Differentiable Sim?** | No | Yes | Yes |
| **Best for** | No autodiff available | Smooth problems | Noisy gradients |

---

## Files in This Directory

```
dmsmpc/
├── dmsmpc.py                          # Original (Approach 1)
├── dmsmpc_differentiable.py           # Pure PyTorch (Approach 3)
├── dmsmpc_scipy_autodiff.py           # Hybrid ⭐ (Approach 2)
├── compare_approaches.py              # Visualization script
├── DIFFERENTIABLE_MPC_EXPLAINED.md    # Theory deep-dive
├── GRADIENT_METHODS_EXPLAINED.md      # Gradient computation methods
└── QUICK_REFERENCE.md                 # This file
```

---

## One-Line Answer

**Yes, scipy optimizers are gradient-based NLP solvers, and you can use them with automatic differentiation from PyTorch for the best of both worlds!**

