#!/usr/bin/env python3
"""
Test script for DMS1 with Warp differentiable simulation.
Demonstrates how to use automatic gradients for MPC optimization.
"""

import numpy as np
import torch
import gym
from omegaconf import OmegaConf

# Import the new Warp-enabled DMS1 agent
from mineral.examples.agents.dms1.dms1_with_warp import DMS1WithWarp

def test_dms1_warp():
    """Test the DMS1WithWarp agent on the pendulum environment."""
    
    # Load configuration
    config = OmegaConf.load('mineral/examples/configs/agent_configs/dms1_warp.yaml')
    
    # Create environment
    env = gym.make('Pendulum-v1')
    
    # Create agent with Warp gradients
    agent = DMS1WithWarp(config)
    
    # Test episode
    obs = env.reset()
    total_reward = 0
    episode_length = 200
    
    print("Running DMS1 with Warp differentiable simulation...")
    print(f"Horizon length: {agent.N}")
    print(f"Time step: {agent.DT}")
    print(f"Using GPU: {torch.cuda.is_available()}")
    
    for step in range(episode_length):
        # Get action using MPC with Warp gradients
        action = agent.get_actions(obs, sample=False)
        
        # Take action in environment
        obs, reward, done, info = env.step(action.cpu().numpy())
        total_reward += reward
        
        if step % 50 == 0:
            print(f"Step {step}: Reward = {reward:.3f}, Action = {action.item():.3f}")
        
        if done:
            break
    
    print(f"\nEpisode completed!")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Episode length: {step + 1}")
    
    env.close()

def compare_gradients():
    """Compare automatic gradients from Warp vs finite differences."""
    
    config = OmegaConf.load('mineral/examples/configs/agent_configs/dms1_warp.yaml')
    agent = DMS1WithWarp(config)
    
    # Create test optimization vector
    current_state = np.array([0.5, 0.1], dtype=np.float32)  # [theta, theta_dot]
    x_u = np.random.randn(agent.N * agent.DIM + agent.STATE_DIM).astype(np.float32)
    
    # Get automatic gradients from Warp
    cost_auto, grad_auto = agent.simulate_with_gradients(x_u, current_state)
    
    # Compute finite difference gradients for comparison
    eps = 1e-5
    grad_fd = np.zeros_like(x_u)
    
    for i in range(len(x_u)):
        x_u_plus = x_u.copy()
        x_u_minus = x_u.copy()
        x_u_plus[i] += eps
        x_u_minus[i] -= eps
        
        cost_plus, _ = agent.simulate_with_gradients(x_u_plus, current_state)
        cost_minus, _ = agent.simulate_with_gradients(x_u_minus, current_state)
        
        grad_fd[i] = (cost_plus - cost_minus) / (2 * eps)
    
    # Compare gradients
    grad_error = np.abs(grad_auto - grad_fd)
    max_error = np.max(grad_error)
    mean_error = np.mean(grad_error)
    
    print("Gradient Comparison:")
    print(f"Cost (automatic): {cost_auto:.6f}")
    print(f"Max gradient error: {max_error:.2e}")
    print(f"Mean gradient error: {mean_error:.2e}")
    print(f"Gradient relative error: {mean_error / (np.mean(np.abs(grad_auto)) + 1e-8):.2e}")
    
    if max_error < 1e-3:
        print("✓ Gradients match well!")
    else:
        print("⚠ Large gradient discrepancy - check implementation")

def profile_performance():
    """Profile the performance improvement from using Warp gradients."""
    import time
    
    config = OmegaConf.load('mineral/examples/configs/agent_configs/dms1_warp.yaml')
    agent = DMS1WithWarp(config)
    
    current_state = np.array([0.5, 0.1], dtype=np.float32)
    x_u = np.random.randn(agent.N * agent.DIM + agent.STATE_DIM).astype(np.float32)
    
    # Warm up
    for _ in range(5):
        agent.simulate_with_gradients(x_u, current_state)
    
    # Profile gradient computation
    n_trials = 50
    start_time = time.time()
    
    for _ in range(n_trials):
        cost, grad = agent.simulate_with_gradients(x_u, current_state)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / n_trials
    
    print(f"Performance Profiling:")
    print(f"Average time per gradient computation: {avg_time*1000:.2f} ms")
    print(f"Horizon length: {agent.N}")
    print(f"Optimization variables: {len(x_u)}")
    print(f"Throughput: {1/avg_time:.1f} gradient computations/second")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing DMS1 with Warp Differentiable Simulation")
    print("=" * 60)
    
    # Test the agent
    test_dms1_warp()
    
    print("\n" + "=" * 60)
    print("Comparing Automatic vs Finite Difference Gradients")
    print("=" * 60)
    
    # Compare gradients
    compare_gradients()
    
    print("\n" + "=" * 60)
    print("Performance Profiling")
    print("=" * 60)
    
    # Profile performance
    profile_performance() 