import numpy as np
import torch
import warp as wp
from scipy.optimize import minimize
from typing import Dict, Any, Callable, Optional
import importlib
import re

from mineral.agents.agent import Agent
from mineral.envs.rewarped import make_envs

# Initialize Warp
wp.init()

class UnifiedMPC(Agent):
    """
    Unified MPC Agent that works across all Rewarped task suites.
    
    This agent uses NVIDIA Warp's unified gradient interface to work with:
    - DFlex (rigid body dynamics)
    - IsaacGymEnvs (parallel simulation)
    - PlasticineLab (MPM deformables)
    - DexDeform (dexterous manipulation)
    - SoftGym (fluid simulation)
    - GradSim (soft body physics)
    
    The key advantage is that all physics engines expose gradients through
    the same Warp tape interface, making MPC optimization physics-agnostic.
    """

    def __init__(self, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.mpc_config = full_cfg.agent.unified_mpc
        self.task_config = full_cfg.task
        
        # MPC parameters
        self.horizon = self.mpc_config.get('horizon', 20)
        self.num_actors = self.mpc_config.num_actors
        self.max_agent_steps = int(self.mpc_config.max_agent_steps)
        self.dt = self.mpc_config.get('dt', 0.05)
        
        # Optimization parameters
        self.max_iter = self.mpc_config.get('max_iter', 100)
        self.cost_weights = self.mpc_config.get('cost_weights', {})
        
        # Environment information (will be set during initialization)
        self.env = None
        self.state_dim = None
        self.action_dim = None
        self.device = None
        
        # Call parent constructor
        super().__init__(full_cfg, **kwargs)
        
        # Initialize after parent constructor
        self._setup_environment()
        self._setup_optimization()
        
    def _setup_environment(self):
        """Initialize the Rewarped environment for this agent."""
        # Create a single environment for MPC planning
        # We use num_envs=1 for MPC since we're doing trajectory optimization
        self.task_config.env.numEnvs = 1
        
        # Create environment using Rewarped factory
        self.env = make_envs(self.mpc_config)
        self.env.init()
        
        # Get environment dimensions
        self.state_dim = self.env.num_observations
        self.action_dim = self.env.num_actions
        self.device = self.env.device
        
        print(f"Initialized {self.task_config.task.env.env_suite}.{self.task_config.task.env.env_name}")
        print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        
    def _setup_optimization(self):
        """Setup optimization variables and bounds."""
        # Total optimization variables: states + actions over horizon
        # States: [current_state, state_1, ..., state_horizon]  
        # Actions: [action_0, action_1, ..., action_{horizon-1}]
        
        self.opt_dim = (self.horizon + 1) * self.state_dim + self.horizon * self.action_dim
        
        # Setup bounds (environment-specific)
        self._setup_bounds()
        
        # Initialize optimization variables
        self.x_u_init = np.zeros(self.opt_dim)
        
    def _setup_bounds(self):
        """Setup optimization bounds based on environment action/state spaces."""
        bounds = []
        
        # State bounds (usually quite loose)
        state_low = -np.inf * np.ones(self.state_dim)
        state_high = np.inf * np.ones(self.state_dim)
        
        # Action bounds from environment
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high
        
        # Add bounds for all states in trajectory
        for i in range(self.horizon + 1):
            for j in range(self.state_dim):
                bounds.append((state_low[j], state_high[j]))
                
        # Add bounds for all actions in trajectory  
        for i in range(self.horizon):
            for j in range(self.action_dim):
                bounds.append((action_low[j], action_high[j]))
                
        self.bounds = bounds
        
    def get_cost_function(self) -> Callable:
        """
        Get task-specific cost function based on environment suite.
        
        This method returns appropriate cost functions for different task types
        while maintaining the unified gradient interface.
        """
        env_suite = self.task_config.task.env.env_suite
        env_name = self.task_config.task.env.env_name.lower()
        
        if env_suite == "dflex":
            if "ant" in env_name:
                return self._locomotion_cost
            elif "humanoid" in env_name:
                return self._locomotion_cost
            else:
                return self._generic_locomotion_cost
                
        elif env_suite == "isaacgymenvs":
            if "hand" in env_name:
                return self._manipulation_cost
            else:
                return self._generic_manipulation_cost
                
        elif env_suite == "plasticinelab":
            return self._deformation_cost
            
        elif env_suite == "dexdeform":
            return self._dexterous_manipulation_cost
            
        elif env_suite == "softgym":
            return self._fluid_manipulation_cost
            
        elif env_suite == "gradsim":
            return self._soft_body_cost
            
        else:
            return self._generic_cost
            
    def _locomotion_cost(self, obs_traj: torch.Tensor, action_traj: torch.Tensor) -> torch.Tensor:
        """Cost function for locomotion tasks (Ant, Humanoid, etc.)."""
        cost = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Get cost weights
        pos_weight = self.cost_weights.get('position', 1.0)
        vel_weight = self.cost_weights.get('velocity', 0.1)
        action_weight = self.cost_weights.get('action', 0.01)
        
        for t in range(self.horizon):
            obs = obs_traj[t]
            action = action_traj[t]
            
            # Position cost (encourage forward motion)
            # Assuming first 3 elements are position
            pos = obs[:3] if len(obs) >= 3 else obs
            pos_cost = -pos_weight * pos[0]  # Negative because we want to go forward
            
            # Velocity cost (encourage smooth motion)
            if len(obs) >= 6:
                vel = obs[3:6]
                vel_cost = vel_weight * torch.sum(vel**2)
            else:
                vel_cost = torch.tensor(0.0, device=self.device)
            
            # Action cost (encourage efficiency)
            action_cost = action_weight * torch.sum(action**2)
            
            cost = cost + pos_cost + vel_cost + action_cost
            
        return cost
        
    def _manipulation_cost(self, obs_traj: torch.Tensor, action_traj: torch.Tensor) -> torch.Tensor:
        """Cost function for manipulation tasks (hand reorientation, grasping, etc.)."""
        cost = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Get cost weights
        pose_weight = self.cost_weights.get('pose', 1.0)
        action_weight = self.cost_weights.get('action', 0.01)
        
        # Target pose (environment-specific, could be loaded from config)
        target_pos = torch.tensor([0.0, 1.0, 0.5], device=self.device)
        
        for t in range(self.horizon):
            obs = obs_traj[t]
            action = action_traj[t]
            
            # Object pose cost (assuming object position is in observation)
            if len(obs) >= 3:
                obj_pos = obs[:3]
                pose_cost = pose_weight * torch.sum((obj_pos - target_pos)**2)
            else:
                pose_cost = torch.tensor(0.0, device=self.device)
            
            # Action cost
            action_cost = action_weight * torch.sum(action**2)
            
            cost = cost + pose_cost + action_cost
            
        return cost
        
    def _deformation_cost(self, obs_traj: torch.Tensor, action_traj: torch.Tensor) -> torch.Tensor:
        """Cost function for deformation tasks (PlasticineLab)."""
        cost = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        shape_weight = self.cost_weights.get('shape', 1.0)
        action_weight = self.cost_weights.get('action', 0.01)
        
        for t in range(self.horizon):
            obs = obs_traj[t]
            action = action_traj[t]
            
            # Shape deformation cost (task-specific)
            # This would need to be customized based on the specific task
            shape_cost = shape_weight * torch.sum(obs**2)  # Placeholder
            
            # Action cost
            action_cost = action_weight * torch.sum(action**2)
            
            cost = cost + shape_cost + action_cost
            
        return cost
        
    def _dexterous_manipulation_cost(self, obs_traj: torch.Tensor, action_traj: torch.Tensor) -> torch.Tensor:
        """Cost function for dexterous manipulation tasks (DexDeform)."""
        return self._manipulation_cost(obs_traj, action_traj)  # Similar to manipulation
        
    def _fluid_manipulation_cost(self, obs_traj: torch.Tensor, action_traj: torch.Tensor) -> torch.Tensor:
        """Cost function for fluid manipulation tasks (SoftGym)."""
        cost = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        flow_weight = self.cost_weights.get('flow', 1.0)
        action_weight = self.cost_weights.get('action', 0.01)
        
        for t in range(self.horizon):
            action = action_traj[t]
            
            # Fluid flow cost (task-specific)
            # This would need to be customized based on the specific fluid task
            flow_cost = torch.tensor(0.0, device=self.device)  # Placeholder
            
            # Action cost
            action_cost = action_weight * torch.sum(action**2)
            
            cost = cost + flow_cost + action_cost
            
        return cost
        
    def _soft_body_cost(self, obs_traj: torch.Tensor, action_traj: torch.Tensor) -> torch.Tensor:
        """Cost function for soft body tasks (GradSim)."""
        cost = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        energy_weight = self.cost_weights.get('energy', 1.0)
        action_weight = self.cost_weights.get('action', 0.01)
        
        for t in range(self.horizon):
            obs = obs_traj[t]
            action = action_traj[t]
            
            # Soft body energy cost
            energy_cost = energy_weight * torch.sum(obs**2)  # Placeholder
            
            # Action cost
            action_cost = action_weight * torch.sum(action**2)
            
            cost = cost + energy_cost + action_cost
            
        return cost
        
    def _generic_cost(self, obs_traj: torch.Tensor, action_traj: torch.Tensor) -> torch.Tensor:
        """Generic cost function for unknown environments."""
        cost = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        obs_weight = self.cost_weights.get('observation', 1.0)
        action_weight = self.cost_weights.get('action', 0.01)
        
        for t in range(self.horizon):
            obs = obs_traj[t]
            action = action_traj[t]
            
            # Generic observation cost
            obs_cost = obs_weight * torch.sum(obs**2)
            
            # Action cost
            action_cost = action_weight * torch.sum(action**2)
            
            cost = cost + obs_cost + action_cost
            
        return cost
        
    def _generic_locomotion_cost(self, obs_traj: torch.Tensor, action_traj: torch.Tensor) -> torch.Tensor:
        """Generic locomotion cost for unknown locomotion environments."""
        return self._locomotion_cost(obs_traj, action_traj)
        
    def _generic_manipulation_cost(self, obs_traj: torch.Tensor, action_traj: torch.Tensor) -> torch.Tensor:
        """Generic manipulation cost for unknown manipulation environments."""
        return self._manipulation_cost(obs_traj, action_traj)

    def simulate_with_gradients(self, x_u_flat: np.ndarray, current_obs: np.ndarray):
        """
        Unified simulation with gradients across all Rewarped environments.
        
        This is the key method that provides a unified interface regardless of
        the underlying physics engine (DFlex, Isaac Gym, MPM, etc.).
        
        Args:
            x_u_flat: Flattened optimization variables [states, actions]
            current_obs: Current observation from environment
            
        Returns:
            tuple: (cost_value, gradient_vector)
        """
        # Convert numpy arrays to torch tensors
        x_u_torch = torch.from_numpy(x_u_flat).float().to(self.device)
        current_obs_torch = torch.from_numpy(current_obs).float().to(self.device)
        
        # Split into states and actions
        state_size = (self.horizon + 1) * self.state_dim
        states_flat = x_u_torch[:state_size]
        actions_flat = x_u_torch[state_size:]
        
        # Reshape into trajectories
        states_traj = states_flat.view(self.horizon + 1, self.state_dim)
        actions_traj = actions_flat.view(self.horizon, self.action_dim)
        
        # Set initial state
        states_traj[0] = current_obs_torch
        
        # Enable gradients
        states_traj.requires_grad_(True)
        actions_traj.requires_grad_(True)
        
        # Create Warp tape for automatic differentiation
        tape = wp.Tape()
        
        with tape:
            # Simulate trajectory through Rewarped environment
            obs_traj = []
            current_state = states_traj[0]
            
            for t in range(self.horizon):
                action = actions_traj[t]
                
                # Convert to numpy for environment step
                current_state_np = current_state.detach().cpu().numpy()
                action_np = action.detach().cpu().numpy()
                
                # Step environment (this automatically uses the appropriate physics)
                next_obs, reward, done, info = self.env.step(
                    action_np.reshape(1, -1)  # Add batch dimension
                )
                
                # Convert back to torch
                next_obs_torch = torch.from_numpy(next_obs.flatten()).float().to(self.device)
                obs_traj.append(current_state)
                
                # Update state (this creates the computational graph)
                states_traj[t + 1] = next_obs_torch
                current_state = next_obs_torch
            
            obs_traj.append(current_state)  # Add final state
            obs_traj_tensor = torch.stack(obs_traj)
            
            # Compute cost using task-specific cost function
            cost_function = self.get_cost_function()
            total_cost = cost_function(obs_traj_tensor, actions_traj)
        
        # Backward pass to compute gradients
        tape.backward(loss=total_cost)
        
        # Extract gradients
        states_grad = tape.gradients[states_traj]
        actions_grad = tape.gradients[actions_traj]
        
        # Flatten gradients
        gradients = torch.cat([
            states_grad.view(-1),
            actions_grad.view(-1)
        ])
        
        # Convert to numpy
        cost_value = total_cost.detach().cpu().numpy()
        gradient_vector = gradients.detach().cpu().numpy()
        
        return cost_value, gradient_vector

    def cost_with_gradients(self, x_u: np.ndarray) -> tuple:
        """
        Cost function with gradients for scipy optimization.
        
        This method wraps simulate_with_gradients to provide the interface
        expected by scipy.optimize.minimize with jac=True.
        """
        try:
            cost, grad = self.simulate_with_gradients(x_u, self.current_obs)
            return cost, grad
        except Exception as e:
            print(f"Error in cost computation: {e}")
            # Return high cost and zero gradients on error
            return 1e6, np.zeros_like(x_u)

    def get_actions(self, obs, sample: bool = True):
        """
        Get actions using MPC optimization.
        
        This method works across all Rewarped environments by using the
        unified gradient interface.
        """
        self.current_obs = obs
        
        # Warm start optimization variables
        if hasattr(self, 'prev_solution'):
            # Shift previous solution (receding horizon)
            self.x_u_init = self._warm_start_solution()
        
        # Run optimization
        try:
            result = minimize(
                fun=self.cost_with_gradients,
                x0=self.x_u_init,
                method='SLSQP',
                jac=True,  # Use our computed gradients
                bounds=self.bounds,
                options={
                    'maxiter': self.max_iter,
                    'ftol': 1e-6,
                    'disp': False
                }
            )
            
            if result.success:
                self.prev_solution = result.x.copy()
                
                # Extract first action from solution
                state_size = (self.horizon + 1) * self.state_dim
                actions_flat = result.x[state_size:]
                actions_traj = actions_flat.reshape(self.horizon, self.action_dim)
                
                # Return first action
                action = actions_traj[0]
                
            else:
                print(f"Optimization failed: {result.message}")
                # Return zero action on failure
                action = np.zeros(self.action_dim)
                
        except Exception as e:
            print(f"MPC optimization error: {e}")
            action = np.zeros(self.action_dim)
        
        return action
    
    def _warm_start_solution(self) -> np.ndarray:
        """Warm start the optimization using previous solution (receding horizon)."""
        if not hasattr(self, 'prev_solution'):
            return self.x_u_init
            
        # Shift the previous solution for receding horizon
        prev_solution = self.prev_solution.copy()
        
        # Extract previous trajectories
        state_size = (self.horizon + 1) * self.state_dim
        prev_states = prev_solution[:state_size].reshape(self.horizon + 1, self.state_dim)
        prev_actions = prev_solution[state_size:].reshape(self.horizon, self.action_dim)
        
        # Create new initial guess by shifting
        new_states = np.zeros((self.horizon + 1, self.state_dim))
        new_actions = np.zeros((self.horizon, self.action_dim))
        
        # Shift states and actions
        new_states[:-1] = prev_states[1:]  # Shift states by 1
        new_states[-1] = prev_states[-1]   # Keep last state
        
        new_actions[:-1] = prev_actions[1:]  # Shift actions by 1
        new_actions[-1] = prev_actions[-1]   # Keep last action
        
        # Flatten and return
        return np.concatenate([new_states.flatten(), new_actions.flatten()])

    def explore_env(self, env, timesteps: int, random: bool = False, sample: bool = False):
        """Explore environment (not used for MPC, but required by Agent interface)."""
        pass

    def train(self):
        """Set agent to training mode (MPC doesn't require training)."""
        pass

    def eval(self):
        """Set agent to evaluation mode."""
        pass

    def set_train(self):
        """Set to training mode."""
        pass

    def set_eval(self):
        """Set to evaluation mode."""  
        pass

    def save(self, f):
        """Save agent state (MPC has no learnable parameters)."""
        pass

    def load(self, f, ckpt_keys=''):
        """Load agent state (MPC has no learnable parameters)."""
        pass 