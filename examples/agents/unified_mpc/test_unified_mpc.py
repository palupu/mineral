#!/usr/bin/env python3
"""
Test script for UnifiedMPC agent across different Rewarped environments.
Demonstrates usage with CartPole (DFlex) and RollingPin (PlasticineLab).
"""

import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import argparse
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

from mineral.envs.rewarped import make_envs
from mineral.examples.agents.unified_mpc.unified_mpc import UnifiedMPC


def test_cartpole_mpc():
    """Test UnifiedMPC on CartPole environment."""
    print("\n=== Testing UnifiedMPC on CartPole (DFlex) ===")
    
    # Environment configuration
    env_config = {
        'env_suite': 'dflex',
        'env_name': 'Cartpole',
        'num_envs': 1,
        'device': 'cpu',
        'no_grad': False,  # Enable gradients for MPC
        'render': True,
        'capture_video': False
    }
    
    # Create environment
    env = make_envs(env_config)
    
    # MPC agent configuration
    mpc_config = {
        'horizon': 15,
        'num_actors': 1,
        'max_agent_steps': 1000,
        'dt': 0.05,
        'max_iter': 100,
        'cost_weights': {
            'position': 5.0,
            'velocity': 0.1, 
            'action': 0.001
        }
    }
    
    # Create UnifiedMPC agent
    agent = UnifiedMPC(
        env=env,
        **mpc_config
    )
    
    # Run test episode
    obs = env.reset()
    total_reward = 0
    
    for step in range(100):
        # Get MPC actions
        actions = agent.get_actions(obs)
        
        # Step environment  
        obs, rewards, dones, infos = env.step(actions)
        total_reward += rewards.mean().item()
        
        if step % 20 == 0:
            print(f"Step {step}: Reward = {rewards.mean().item():.3f}, Total = {total_reward:.3f}")
            
        if dones.any():
            break
    
    print(f"CartPole test completed. Final total reward: {total_reward:.3f}")
    env.close()


def test_rollingpin_mpc():
    """Test UnifiedMPC on RollingPin environment.""" 
    print("\n=== Testing UnifiedMPC on RollingPin (PlasticineLab) ===")
    
    # Environment configuration
    env_config = {
        'env_suite': 'plasticinelab',
        'env_name': 'RollingPin', 
        'num_envs': 1,
        'device': 'cpu',
        'no_grad': False,  # Enable gradients for MPC
        'render': True,
        'capture_video': False
    }
    
    # Create environment
    env = make_envs(env_config)
    
    # MPC agent configuration
    mpc_config = {
        'horizon': 25,
        'num_actors': 1,
        'max_agent_steps': 1000,
        'dt': 0.02,
        'max_iter': 150,
        'cost_weights': {
            'position': 3.0,
            'velocity': 0.05,
            'action': 0.001,
            'deformation': 2.0,
            'smoothness': 0.1
        }
    }
    
    # Create UnifiedMPC agent
    agent = UnifiedMPC(
        env=env,
        **mpc_config
    )
    
    # Run test episode
    obs = env.reset()
    total_reward = 0
    
    for step in range(50):  # Shorter episode for complex physics
        # Get MPC actions
        actions = agent.get_actions(obs)
        
        # Step environment
        obs, rewards, dones, infos = env.step(actions)
        total_reward += rewards.mean().item()
        
        if step % 10 == 0:
            print(f"Step {step}: Reward = {rewards.mean().item():.3f}, Total = {total_reward:.3f}")
            
        if dones.any():
            break
    
    print(f"RollingPin test completed. Final total reward: {total_reward:.3f}")
    env.close()


def test_gradient_computation():
    """Test gradient computation through Rewarped environments."""
    print("\n=== Testing Gradient Computation ===")
    
    # Simple CartPole test with gradients
    env_config = {
        'env_suite': 'dflex',
        'env_name': 'Cartpole',
        'num_envs': 1,
        'device': 'cpu',
        'no_grad': False
    }
    
    env = make_envs(env_config)
    
    # Test forward pass with gradients
    obs = env.reset()
    
    # Create random action requiring gradients
    actions = torch.randn(1, env.action_space.shape[0], requires_grad=True)
    
    # Forward pass
    next_obs, rewards, dones, infos = env.step(actions)
    
    # Compute simple loss
    loss = -rewards.sum()
    
    # Backward pass
    loss.backward()
    
    print(f"Action gradients computed: {actions.grad is not None}")
    print(f"Gradient magnitude: {actions.grad.norm().item():.6f}")
    
    env.close()


def benchmark_performance():
    """Benchmark MPC performance across environments."""
    print("\n=== Performance Benchmark ===")
    
    environments = [
        {'env_suite': 'dflex', 'env_name': 'Cartpole', 'horizon': 15},
        {'env_suite': 'plasticinelab', 'env_name': 'RollingPin', 'horizon': 25}
    ]
    
    for env_info in environments:
        print(f"\nBenchmarking {env_info['env_name']}...")
        
        env_config = {
            'env_suite': env_info['env_suite'],
            'env_name': env_info['env_name'],
            'num_envs': 1,
            'device': 'cpu',
            'no_grad': False
        }
        
        env = make_envs(env_config)
        
        mpc_config = {
            'horizon': env_info['horizon'],
            'num_actors': 1,
            'max_agent_steps': 1000,
            'max_iter': 50,  # Reduced for benchmarking
            'cost_weights': {'position': 1.0, 'velocity': 0.1, 'action': 0.001}
        }
        
        agent = UnifiedMPC(env=env, **mpc_config)
        
        obs = env.reset()
        
        # Time MPC computation
        import time
        start_time = time.time()
        
        for _ in range(10):
            actions = agent.get_actions(obs)
            obs, _, _, _ = env.step(actions)
            
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        print(f"Average MPC time per step: {avg_time:.4f}s")
        
        env.close()


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test UnifiedMPC agent')
    parser.add_argument('--test', choices=['cartpole', 'rollingpin', 'gradients', 'benchmark', 'all'], 
                       default='all', help='Which test to run')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device to use')
    
    args = parser.parse_args()
    
    print("UnifiedMPC Test Suite")
    print("====================")
    
    if args.test in ['cartpole', 'all']:
        try:
            test_cartpole_mpc()
        except Exception as e:
            print(f"CartPole test failed: {e}")
    
    if args.test in ['rollingpin', 'all']:
        try:
            test_rollingpin_mpc()
        except Exception as e:
            print(f"RollingPin test failed: {e}")
    
    if args.test in ['gradients', 'all']:
        try:
            test_gradient_computation()
        except Exception as e:
            print(f"Gradient test failed: {e}")
            
    if args.test in ['benchmark', 'all']:
        try:
            benchmark_performance()
        except Exception as e:
            print(f"Benchmark test failed: {e}")
    
    print("\nTest suite completed!")


if __name__ == "__main__":
    main() 