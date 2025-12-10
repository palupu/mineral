"""DMS MPC using SLSQP optimizer with automatic differentiation.

This combines the best of both worlds:
- SLSQP's robust constrained optimization
- PyTorch's automatic differentiation through physics

Key insight: SLSQP can accept user-provided gradients via jac=True,
and we compute exact gradients through differentiable physics simulation.
"""

import time

import numpy as np
import torch
from scipy.optimize import minimize

from mineral.agents.agent import Agent


class DMSMPCScipyAutodiffAgent(Agent):
    r"""DMS MPC using SLSQP optimizer with PyTorch autodiff gradients.

    SLSQP (Sequential Least Squares Programming) is a robust gradient-based
    optimizer for constrained nonlinear optimization. Combined with PyTorch's
    automatic differentiation through physics, we get:
    - Exact gradients via backpropagation through the simulator
    - Efficient bound-constrained optimization
    - No finite differences or numerical approximations
    - Single rollout per iteration (cost + gradient computed together)
    """

    def __init__(self, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.params = full_cfg.agent.params
        self.num_actors = self.params.num_actors
        self.max_agent_steps = int(self.params.max_agent_steps)
        self.render_results = self.params.render_results

        # DMS MPC Parameters
        self.dms_mpc_params = full_cfg.agent.dms_mpc_params
        self.N = self.dms_mpc_params.N
        self.timesteps = self.dms_mpc_params.timesteps
        self.max_iter = self.dms_mpc_params.max_iter

        # Cost parameters
        self.cost_state = self.dms_mpc_params.cost_state
        self.cost_control = self.dms_mpc_params.cost_control
        self.cost_terminal = self.dms_mpc_params.cost_terminal

        # Derived parameters - determine dynamically from environment
        # These will be set after super().__init__ when self.env is available
        self.state_dim = None
        self.control_dim = None

        super().__init__(full_cfg, **kwargs)

        # Now that self.env is available, determine dimensions dynamically
        self._determine_dimensions()

        self.obs = None
        self.dones = None

        # Control bounds
        self.action_lower = -1.0 * np.ones(self.control_dim)
        self.action_upper = 1.0 * np.ones(self.control_dim)

    def _determine_dimensions(self):
        """Determine state and control dimensions dynamically from environment."""
        # Get control dimension from environment action space
        self.control_dim = self.env.action_space.shape[0]

        # Determine state dimension from observation space
        if hasattr(self.env.observation_space, 'spaces') and isinstance(self.env.observation_space.spaces, dict):
            # For dict observation spaces (like RollingPin), determine state dimension
            # by looking at the observation structure
            obs_space = self.env.observation_space.spaces

            # For RollingPin: joint_q (3) + com_q (3) = 6
            if 'joint_q' in obs_space and 'com_q' in obs_space:
                joint_dim = obs_space['joint_q'].shape[0] if len(obs_space['joint_q'].shape) > 0 else 1
                com_dim = obs_space['com_q'].shape[0] if len(obs_space['com_q'].shape) > 0 else 1
                self.state_dim = joint_dim + com_dim
            else:
                # Fallback: use total observation dimension
                total_dim = sum(space.shape[0] if len(space.shape) > 0 else 1
                              for space in obs_space.values())
                self.state_dim = total_dim
        else:
            # For simple observation spaces
            self.state_dim = self.env.observation_space.shape[0]

        print(f"Auto-detected dimensions: state_dim={self.state_dim}, control_dim={self.control_dim}")
        print(f"Environment info: env.num_envs={self.env.num_envs}, agent.num_actors={self.num_actors}")

        # Override with config values if they exist (for backward compatibility)
        if hasattr(self.dms_mpc_params, 'state_dim') and self.dms_mpc_params.state_dim is not None:
            self.state_dim = self.dms_mpc_params.state_dim
            print(f"Using config state_dim: {self.state_dim}")
        if hasattr(self.dms_mpc_params, 'control_dim') and self.dms_mpc_params.control_dim is not None:
            self.control_dim = self.dms_mpc_params.control_dim
            print(f"Using config control_dim: {self.control_dim}")

    def _obs_to_state(self, obs, detach=False):
        """Convert observation to state tensor.

        Args:
            obs: Observation from environment
            detach: If True, detach from computational graph (for non-gradient operations)
        """
        if isinstance(obs, dict):
            joint_q = obs.get('joint_q', torch.zeros(3, device=self.device))
            com_q = obs.get('com_q', torch.zeros(3, device=self.device))

            # Detach if requested to break computational graph
            if detach:
                joint_q = joint_q.detach() if isinstance(joint_q, torch.Tensor) else joint_q
                com_q = com_q.detach() if isinstance(com_q, torch.Tensor) else com_q

            state = torch.cat([joint_q.flatten(), com_q.flatten()])
            return state[: self.state_dim]

        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.detach() if detach else obs
            return obs_tensor[: self.state_dim]

        return torch.tensor(obs[: self.state_dim], device=self.device, dtype=torch.float32)

    def eval(self):
        if self.render_results:
            self.replay_trajectory()
        else:
            start_time = time.time()
            self.run_dms_mpc()
            end_time = time.time()
            print(f"Time taken: {end_time - start_time:.2f}s")

    def rollout_and_cost(self, controls_np, saved_state, compute_gradients=True):
        """Rollout trajectory and compute cost + gradients.

        Args:
            controls_np: Numpy array of shape (N * control_dim,)
            saved_state: Environment state to restore before rollout
            compute_gradients: If True, compute gradients; if False, only compute cost

        Returns:
            cost: Scalar cost value (numpy float)
            grad: Gradient w.r.t. controls (numpy array) or None
        """
        # Restore environment state before rollout using environment's built-in method
        # This properly detaches from any previous computational graphs
        self.env.clear_grad(checkpoint=saved_state)

        # Convert numpy to torch with gradient tracking
        if compute_gradients:
            controls_flat = torch.tensor(controls_np, device=self.device, dtype=torch.float32, requires_grad=True)
        else:
            controls_flat = torch.tensor(controls_np, device=self.device, dtype=torch.float32)

        # Reshape to (N, control_dim)
        controls = controls_flat.reshape(self.N, self.control_dim)

        # Rollout trajectory
        total_cost = 0.0

        for i in range(self.N):
            # Clip controls to bounds
            u = torch.clamp(
                controls[i],
                torch.tensor(self.action_lower, device=self.device, dtype=torch.float32),
                torch.tensor(self.action_upper, device=self.device, dtype=torch.float32),
            )

            # Don't expand - environment expects single action for single env
            # Shape should be (1, action_dim) for one environment
            actions = u.unsqueeze(0)

            # Step environment (differentiable)
            obs, reward, done, info = self.env.step(actions)

            # Extract state
            state = self._obs_to_state(obs)

            # Stage cost
            state_cost = self.cost_state * torch.sum(state**2)
            control_cost = self.cost_control * torch.sum(u**2)
            total_cost = total_cost + state_cost + control_cost

        # Terminal cost
        terminal_state = state
        terminal_cost = self.cost_terminal * torch.sum(terminal_state**2)
        total_cost = total_cost + terminal_cost

        # Extract cost
        cost_value = total_cost.item()

        # Compute gradients if requested
        if compute_gradients:
            # Backpropagate to get gradients
            total_cost.backward()
            grad_value = controls_flat.grad.detach().cpu().numpy()

            # Clear gradients to prevent accumulation across optimization iterations
            controls_flat.grad = None
        else:
            grad_value = None

        return cost_value, grad_value

    def cost_and_gradient(self, controls_np):
        """Combined cost and gradient function optimized for SLSQP.

        SLSQP efficiently handles functions that return both cost and gradient
        in a single call (jac=True), avoiding redundant rollouts.

        Returns:
            cost: Scalar cost value
            grad: Gradient array of same shape as controls_np
        """
        cost, grad = self.rollout_and_cost(controls_np, self._saved_state, compute_gradients=True)

        # Increment iteration counter and print for monitoring
        self._iter_count += 1
        print(f"    Iter {self._iter_count}/{self.max_iter}: Cost={cost:.6f}, ||grad||={np.linalg.norm(grad):.6f}", flush=True)

        return cost, grad

    def dms_plan(self, current_state):
        """Plan using SLSQP optimizer with autodiff gradients.

        SLSQP (Sequential Least Squares Programming) is a robust gradient-based
        optimizer that handles bound constraints efficiently. We use it with
        PyTorch autodiff to get exact gradients through the physics simulation.
        """
        # Initialize controls (flattened to 1D array for scipy)
        controls_init = np.zeros(self.N * self.control_dim)

        # Define bounds for each control variable at each horizon step
        bounds = [(self.action_lower[j], self.action_upper[j])
                  for _ in range(self.N)
                  for j in range(self.control_dim)]

        # Save environment state using environment's checkpoint system
        # This ensures each rollout starts fresh without sharing computational graphs
        self._saved_state = self.env.get_checkpoint(detach=True)

        # Initialize iteration counter for progress tracking
        self._iter_count = 0

        # Run SLSQP optimization with combined cost and gradient
        # jac=True tells scipy that the function returns (cost, gradient)
        result = minimize(
            fun=self.cost_and_gradient,
            x0=controls_init,
            method='SLSQP',
            jac=True,  # Function returns both cost and gradient
            bounds=bounds,
            options={
                'maxiter': self.max_iter,
                'ftol': 1e-6,  # Function tolerance
                'disp': False,  # Don't print scipy's own progress
            },
        )

        # Extract optimal control sequence
        optimal_controls = result.x.reshape(self.N, self.control_dim)
        optimal_action = optimal_controls[0]  # MPC: execute only first action

        # Print optimization summary
        print("\n  ✓ Optimization complete!", flush=True)
        print(f"    Success: {result.success} | Message: {result.message}", flush=True)
        print(f"    Final cost: {result.fun:.6f} | Iterations: {result.nit}", flush=True)
        print(f"    Optimal action (first step): {optimal_action}", flush=True)
        print(f"    Function evals: {result.nfev} | Gradient evals: {result.njev}", flush=True)

        return torch.tensor(optimal_action, device=self.device, dtype=torch.float32)

    def run_dms_mpc(self):
        """Main MPC loop."""
        # Initialize environment
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)
        total_reward = 0.0

        best_actions_list = []

        print(f"\n{'='*60}", flush=True)
        print("=== Starting DMS MPC with SLSQP + Autodiff ===", flush=True)
        print(f"  State dim: {self.state_dim}, Control dim: {self.control_dim}", flush=True)
        print(f"  Horizon (N): {self.N}, Total timesteps: {self.timesteps}", flush=True)
        print(f"  Optimizer: SLSQP, Max iter per step: {self.max_iter}", flush=True)
        print(f"  Cost weights - State: {self.cost_state}, Control: {self.cost_control}, Terminal: {self.cost_terminal}", flush=True)
        print(f"{'='*60}\n", flush=True)

        # Main control loop
        for timestep in range(self.timesteps):
            if self.dones[0]:
                break

            print(f"\n{'─'*60}", flush=True)
            print(f"Timestep {timestep + 1}/{self.timesteps}", flush=True)

            current_state = self._obs_to_state(obs)
            # Save state before optimization (will be modified during planning)
            saved_env_state = self.env.get_checkpoint(detach=True)

            # Enable gradients temporarily
            with torch.enable_grad():
                original_no_grad = self.env.no_grad
                self.env.no_grad = False

                optimal_action = self.dms_plan(current_state)

                self.env.no_grad = original_no_grad

            # Restore state and execute optimal action
            self.env.clear_grad(checkpoint=saved_env_state)
            # Shape should be (1, action_dim) for one environment
            actions = optimal_action.unsqueeze(0)
            obs, reward, done, info = self.env.step(actions)

            self.obs = self._convert_obs(obs)
            self.dones = done
            total_reward += reward[0].item() if isinstance(reward, torch.Tensor) else reward[0]

            best_actions_list.append(optimal_action.clone())

            print(f"    Reward: {reward[0]:.4f} | Cumulative reward: {total_reward:.4f}", flush=True)

        print(f"\n{'='*60}", flush=True)
        print("=== Evaluation Complete ===", flush=True)
        print(f"  Total reward: {total_reward:.4f}", flush=True)
        print(f"  Total timesteps: {len(best_actions_list)}", flush=True)
        print(f"  Average reward per step: {total_reward / len(best_actions_list):.4f}", flush=True)
        print(f"{'='*60}", flush=True)

        best_actions_tensor = torch.stack(best_actions_list)
        torch.save(best_actions_tensor, 'trajectory_scipy_autodiff.pt')
        print("Trajectory saved")

    def replay_trajectory(self):
        """Replay saved trajectory."""
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)

        actions_to_replay = torch.load('trajectory_scipy_autodiff.pt')

        print("Replaying trajectory...")
        for timestep, action in enumerate(actions_to_replay):
            actions = action.unsqueeze(0).repeat(self.num_actors, 1)
            obs, reward, done, info = self.env.step(actions)
            self.obs = self._convert_obs(obs)
            self.dones = done

            reward_val = reward[0].item() if isinstance(reward, torch.Tensor) else reward[0]
            print(f"Timestep {timestep + 1} | Reward: {reward_val:.3f}")

        print("Trajectory replay complete")
