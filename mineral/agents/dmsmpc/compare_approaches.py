"""
Comparison script for different DMS MPC approaches.

This script helps understand the trade-offs between:
1. Constraint-based DMS MPC (SLSQP with equality constraints)
2. Differentiable DMS MPC (gradient descent through physics)
3. CEM MPC (sampling-based optimization)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_optimization_landscape():
    """Visualize how different optimizers navigate the control space."""
    
    # Create a simple 2D cost landscape
    u1 = np.linspace(-2, 2, 100)
    u2 = np.linspace(-2, 2, 100)
    U1, U2 = np.meshgrid(u1, u2)
    
    # Simple quadratic cost with some nonlinearity
    Cost = U1**2 + U2**2 + 0.5 * np.sin(3*U1) * np.cos(3*U2)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Constraint-based (SLSQP)
    ax = axes[0]
    ax.contourf(U1, U2, Cost, levels=20, cmap='viridis')
    ax.set_title('Constraint-Based DMS (SLSQP)', fontsize=14)
    ax.set_xlabel('Control u₁', fontsize=12)
    ax.set_ylabel('Control u₂', fontsize=12)
    
    # Simulate SLSQP path (Newton-like steps)
    slsqp_path = np.array([
        [1.5, 1.5], [1.0, 1.0], [0.5, 0.5], [0.2, 0.2], [0.05, 0.05], [0.0, 0.0]
    ])
    ax.plot(slsqp_path[:, 0], slsqp_path[:, 1], 'r-o', linewidth=2, markersize=8, label='SLSQP Path')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gradient-based (Adam)
    ax = axes[1]
    ax.contourf(U1, U2, Cost, levels=20, cmap='viridis')
    ax.set_title('Differentiable DMS (Adam)', fontsize=14)
    ax.set_xlabel('Control u₁', fontsize=12)
    ax.set_ylabel('Control u₂', fontsize=12)
    
    # Simulate Adam path (more gradual, adaptive steps)
    adam_path = np.array([
        [1.5, 1.5], [1.3, 1.3], [1.0, 1.0], [0.7, 0.7], [0.4, 0.4], 
        [0.2, 0.2], [0.1, 0.1], [0.03, 0.03], [0.01, 0.01], [0.0, 0.0]
    ])
    ax.plot(adam_path[:, 0], adam_path[:, 1], 'g-o', linewidth=2, markersize=8, label='Adam Path')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Sampling-based (CEM)
    ax = axes[2]
    ax.contourf(U1, U2, Cost, levels=20, cmap='viridis')
    ax.set_title('Sampling-Based (CEM)', fontsize=14)
    ax.set_xlabel('Control u₁', fontsize=12)
    ax.set_ylabel('Control u₂', fontsize=12)
    
    # Simulate CEM iterations (samples converging to optimum)
    np.random.seed(42)
    for iteration in range(3):
        if iteration == 0:
            mean, std = np.array([1.5, 1.5]), 0.8
        elif iteration == 1:
            mean, std = np.array([0.7, 0.7]), 0.4
        else:
            mean, std = np.array([0.1, 0.1]), 0.2
        
        samples = np.random.randn(50, 2) * std + mean
        alpha = 0.3 if iteration < 2 else 0.8
        ax.scatter(samples[:, 0], samples[:, 1], alpha=alpha, s=20, 
                  label=f'Iteration {iteration+1}', c=f'C{iteration}')
    
    ax.plot([1.5, 0.7, 0.1, 0.0], [1.5, 0.7, 0.1, 0.0], 'k--', linewidth=2, label='Mean Path')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/app/mineral/optimization_comparison.png', dpi=300)
    print("Saved visualization to optimization_comparison.png")
    plt.close()


def compare_computational_cost():
    """Compare computational complexity of different approaches."""
    
    horizons = np.array([5, 10, 15, 20, 25, 30])
    state_dim = 6
    control_dim = 3
    
    # SLSQP: Cost scales with (state_dim + control_dim) * N + state_dim variables
    # and state_dim * (N+1) constraints
    slsqp_vars = (state_dim + control_dim) * horizons + state_dim
    slsqp_constraints = state_dim * (horizons + 1)
    slsqp_time = slsqp_vars * 0.5 + slsqp_constraints * 0.8  # Arbitrary scaling
    
    # Differentiable: Cost scales with control_dim * N variables only
    diff_vars = control_dim * horizons
    diff_time = diff_vars * 0.3  # Faster per variable due to autodiff
    
    # CEM: Cost scales with num_samples * N rollouts
    num_samples = 100
    cem_time = num_samples * horizons * 0.05  # Per rollout cost
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Number of variables
    ax = axes[0]
    ax.plot(horizons, slsqp_vars, 'r-o', linewidth=2, label='SLSQP (states + controls)')
    ax.plot(horizons, diff_vars, 'g-s', linewidth=2, label='Differentiable (controls only)')
    ax.fill_between(horizons, 0, slsqp_constraints, alpha=0.3, color='red', label='SLSQP constraints')
    ax.set_xlabel('Horizon Length (N)', fontsize=12)
    ax.set_ylabel('Number of Variables/Constraints', fontsize=12)
    ax.set_title('Optimization Problem Size', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Relative computation time
    ax = axes[1]
    ax.plot(horizons, slsqp_time / slsqp_time[0], 'r-o', linewidth=2, label='SLSQP')
    ax.plot(horizons, diff_time / slsqp_time[0], 'g-s', linewidth=2, label='Differentiable')
    ax.plot(horizons, cem_time / slsqp_time[0], 'b-^', linewidth=2, label='CEM')
    ax.set_xlabel('Horizon Length (N)', fontsize=12)
    ax.set_ylabel('Relative Computation Time', fontsize=12)
    ax.set_title('Computational Cost Scaling', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/app/mineral/computational_comparison.png', dpi=300)
    print("Saved computational comparison to computational_comparison.png")
    plt.close()


def gradient_flow_example():
    """Demonstrate gradient flow through differentiable physics."""
    
    print("\n" + "="*60)
    print("GRADIENT FLOW DEMONSTRATION")
    print("="*60)
    
    # Simple 1D double integrator: x'' = u
    # State: [position, velocity]
    # Dynamics: x[k+1] = A*x[k] + B*u[k]
    
    dt = 0.1
    A = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=torch.float32)
    B = torch.tensor([[0.5*dt**2], [dt]], dtype=torch.float32)
    
    # Cost weights
    Q = torch.eye(2) * 1.0
    R = torch.eye(1) * 0.1
    
    N = 10  # Horizon
    
    # Initialize controls with gradient tracking
    controls = torch.zeros(N, 1, requires_grad=True)
    
    # Rollout trajectory
    x = torch.tensor([[1.0], [0.0]], dtype=torch.float32)  # Start at position 1
    states = [x]
    cost = 0.0
    
    for i in range(N):
        # Dynamics step
        x = A @ x + B * controls[i]
        states.append(x)
        
        # Accumulate cost
        cost = cost + x.T @ Q @ x + controls[i].T @ R @ controls[i]
    
    # Backward pass
    cost.backward()
    
    print(f"\nInitial state: {states[0].squeeze().detach().numpy()}")
    print(f"Final state: {states[-1].squeeze().detach().numpy()}")
    print(f"Total cost: {cost.item():.4f}")
    print(f"\nControl gradients (∂J/∂u):")
    for i in range(N):
        grad = controls.grad[i].item()
        print(f"  u[{i}]: {grad:+.4f}")
    
    print(f"\nGradient interpretation:")
    print(f"  - Large positive gradient → decreasing u reduces cost")
    print(f"  - Gradient magnitude → sensitivity of cost to control")
    
    # Optimize using gradient descent
    print("\n" + "-"*60)
    print("OPTIMIZATION USING GRADIENTS")
    print("-"*60)
    
    controls_opt = torch.zeros(N, 1, requires_grad=True)
    optimizer = torch.optim.Adam([controls_opt], lr=0.1)
    
    for iteration in range(50):
        optimizer.zero_grad()
        
        x = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
        cost = 0.0
        
        for i in range(N):
            x = A @ x + B * controls_opt[i]
            cost = cost + x.T @ Q @ x + controls_opt[i].T @ R @ controls_opt[i]
        
        cost.backward()
        optimizer.step()
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration:2d}: Cost = {cost.item():.6f}")
    
    print(f"\nOptimal controls:")
    for i in range(N):
        print(f"  u*[{i}] = {controls_opt[i].item():+.4f}")
    
    print(f"\nFinal optimized cost: {cost.item():.6f}")


def main():
    """Run all comparisons."""
    print("Generating comparison visualizations...")
    
    # 1. Optimization landscape
    visualize_optimization_landscape()
    
    # 2. Computational cost
    compare_computational_cost()
    
    # 3. Gradient flow
    gradient_flow_example()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
Constraint-Based DMS (SLSQP):
  ✓ Well-established theory
  ✓ Handles constraints naturally
  ✗ Many variables (states + controls)
  ✗ Requires dynamics model f(x,u)
  ✗ Slow for large horizons
  
Differentiable DMS (Adam/L-BFGS):
  ✓ Fewer variables (controls only)
  ✓ Exact gradients via autodiff
  ✓ No dynamics model needed
  ✓ Fast for differentiable simulators
  ✗ Requires differentiable physics
  ✗ May struggle with discontinuities
  
Sampling-Based (CEM):
  ✓ No gradients needed
  ✓ Handles discontinuities well
  ✓ Easy to parallelize
  ✗ Sample inefficient
  ✗ Slow convergence
  ✗ Needs many rollouts
  
RECOMMENDATION FOR DEFORMABLE OBJECTS:
  Use Differentiable DMS when:
    - Rewarped simulator is available ✓
    - Dynamics are smooth (plasticine) ✓
    - Need fast planning ✓
    """)
    
    print("\nAll visualizations saved!")
    print("  - optimization_comparison.png")
    print("  - computational_comparison.png")


if __name__ == "__main__":
    main()

