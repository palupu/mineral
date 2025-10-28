import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from mineral.agents.agent import Agent


class DMSMPCDifferentiableAgent(Agent):
    r"""Direct Multiple Shooting MPC with Differentiable Physics.

    This implementation leverages Rewarped's automatic differentiation capabilities
    to optimize control sequences through gradient descent instead of constraint-based
    optimization. This is more efficient and natural for differentiable simulators.

    Key advantages:
    - No equality constraints needed (dynamics satisfied by simulation)
    - Exact gradients via backpropagation through time
    - Better suited for complex deformable object physics
    - Faster convergence with proper optimizer settings
    """

    def __init__(self, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.params = full_cfg.agent.params
        self.num_actors = self.params.num_actors
        self.max_agent_steps = int(self.params.max_agent_steps)
        self.render_results = self.params.render_results

        # Collect all DMS MPC Parameters
        self.dms_mpc_params = full_cfg.agent.dms_mpc_params
        self.N = self.dms_mpc_params.N  # Number of shooting nodes (horizon length)
        self.timesteps = self.dms_mpc_params.timesteps
        self.max_iter = self.dms_mpc_params.max_iter  # Optimization iterations per MPC step

        # Cost parameters
        self.cost_state = self.dms_mpc_params.cost_state
        self.cost_control = self.dms_mpc_params.cost_control
        self.cost_terminal = self.dms_mpc_params.cost_terminal

        # Optimizer settings
        self.learning_rate = self.dms_mpc_params.get('learning_rate', 0.01)
        self.optimizer_type = self.dms_mpc_params.get('optimizer', 'Adam')  # 'Adam' or 'LBFGS'

        # Derived parameters - determine dynamically from environment
        # These will be set after super().__init__ when self.env is available
        self.state_dim = None
        self.control_dim = None

        super().__init__(full_cfg, **kwargs)
        
        # Now that self.env is available, determine dimensions dynamically
        self._determine_dimensions()

        self.obs = None
        self.dones = None

        # Control bounds (normalized action space) - will be updated after dimensions are determined
        self.action_lower_bound = None
        self.action_upper_bound = None

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
        
        # Override with config values if they exist (for backward compatibility)
        if hasattr(self.dms_mpc_params, 'state_dim') and self.dms_mpc_params.state_dim is not None:
            self.state_dim = self.dms_mpc_params.state_dim
            print(f"Using config state_dim: {self.state_dim}")
        if hasattr(self.dms_mpc_params, 'control_dim') and self.dms_mpc_params.control_dim is not None:
            self.control_dim = self.dms_mpc_params.control_dim
            print(f"Using config control_dim: {self.control_dim}")
        
        # Update control bounds based on detected control dimension
        self.action_lower_bound = torch.tensor(
            [-1.0] * self.control_dim, device=self.device, dtype=torch.float32
        )
        self.action_upper_bound = torch.tensor(
            [1.0] * self.control_dim, device=self.device, dtype=torch.float32
        )

    def _obs_to_state(self, obs):
        """Convert observation to state vector for cost computation."""
        if isinstance(obs, dict):
            joint_q = obs.get('joint_q', torch.zeros(3, device=self.device))
            com_q = obs.get('com_q', torch.zeros(3, device=self.device))

            # Keep as torch tensors for differentiability
            state = torch.cat([joint_q.flatten(), com_q.flatten()])
            return state[: self.state_dim]

        if isinstance(obs, torch.Tensor):
            return obs[: self.state_dim]

        # Fallback to numpy conversion
        if isinstance(obs, np.ndarray):
            return torch.tensor(obs[: self.state_dim], device=self.device, dtype=torch.float32)

        return obs

    def eval(self):
        if self.render_results:
            self.replay_trajectory()
        else:
            start_time = time.time()
            self.run_dms_mpc()
            end_time = time.time()
            print(f"Time taken: {end_time - start_time:.2f}s")

    def rollout_trajectory(self, controls, target_state=None):
        """Rollout a trajectory using differentiable physics.

        Args:
            controls: Tensor of shape (N, control_dim) with requires_grad=True
            target_state: Optional target state for cost computation

        Returns:
            states: List of states (as tensors) at each node
            total_cost: Scalar tensor representing trajectory cost
        """
        states = []
        total_cost = 0.0

        # Ensure environment is in gradient mode
        # Rewarped will handle gradient flow automatically when tensors have requires_grad=True

        for i in range(self.N):
            # Clip controls to action bounds
            u = torch.clamp(controls[i], self.action_lower_bound, self.action_upper_bound)

            # Expand for all actors (environment expects batch dimension)
            actions = u.unsqueeze(0).repeat(self.env.num_envs, 1)

            # Step the differentiable environment
            obs, reward, done, info = self.env.step(actions)

            # Extract state (maintain gradient flow)
            state = self._obs_to_state(obs)
            states.append(state)

            # Compute stage cost
            # Q-cost on state
            state_cost = self.cost_state * torch.sum(state ** 2)

            # R-cost on control
            control_cost = self.cost_control * torch.sum(u ** 2)

            total_cost = total_cost + state_cost + control_cost

        # Terminal cost
        terminal_state = states[-1]
        terminal_cost = self.cost_terminal * torch.sum(terminal_state ** 2)
        total_cost = total_cost + terminal_cost

        # Optional: Add cost for deviation from target
        if target_state is not None:
            target_tensor = torch.tensor(target_state, device=self.device, dtype=torch.float32)
            tracking_cost = 10.0 * torch.sum((terminal_state - target_tensor) ** 2)
            total_cost = total_cost + tracking_cost

        return states, total_cost

    def dms_plan(self, current_state_tensor):
        """Direct Multiple Shooting planning using differentiable physics.

        Args:
            current_state_tensor: Current state as a tensor

        Returns:
            optimal_action: First control from optimized sequence (tensor)
        """
        # Initialize control sequence (decision variables)
        # Start with small random values or zeros
        controls = torch.zeros(
            (self.N, self.control_dim),
            device=self.device,
            dtype=torch.float32,
            requires_grad=True
        )

        # Alternatively, initialize with small random values for better exploration
        # controls = torch.randn(
        #     (self.N, self.control_dim),
        #     device=self.device,
        #     dtype=torch.float32,
        #     requires_grad=True
        # ) * 0.1

        # Choose optimizer
        if self.optimizer_type == 'Adam':
            optimizer = torch.optim.Adam([controls], lr=self.learning_rate)
        elif self.optimizer_type == 'LBFGS':
            optimizer = torch.optim.LBFGS([controls], lr=self.learning_rate, max_iter=20)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        # Optimization loop
        best_cost = float('inf')
        best_controls = None

        for iteration in range(self.max_iter):
            def closure():
                """Closure function for optimizers like LBFGS."""
                optimizer.zero_grad()

                # Rollout trajectory and compute cost
                states, cost = self.rollout_trajectory(controls)

                # Backpropagate
                cost.backward()

                return cost

            if self.optimizer_type == 'LBFGS':
                cost = optimizer.step(closure)
            else:
                optimizer.zero_grad()
                states, cost = self.rollout_trajectory(controls)
                cost.backward()
                optimizer.step()

            # Track best solution
            cost_value = cost.item() if isinstance(cost, torch.Tensor) else cost
            if cost_value < best_cost:
                best_cost = cost_value
                best_controls = controls.detach().clone()

            # Logging
            if iteration % 5 == 0 or iteration == self.max_iter - 1:
                print(f"  Iteration {iteration:3d} | Cost: {cost_value:.6f}")

            # Early stopping if cost is small enough
            if cost_value < 1e-3:
                print(f"  Converged at iteration {iteration}")
                break

        # Extract first control from optimized sequence
        optimal_action = best_controls[0] if best_controls is not None else controls[0].detach()

        print(f"  Final cost: {best_cost:.6f}")
        print(f"  Optimal action: {optimal_action.cpu().numpy()}")

        return optimal_action

    def run_dms_mpc(self):
        """Main MPC loop using differentiable planning."""
        # Initialize environment
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)
        total_reward = 0.0

        # Save trajectory
        best_actions_list = []

        # Main control loop
        for timestep in range(self.timesteps):
            if self.dones[0]:
                break

            print(f"\n=== Timestep {timestep + 1}/{self.timesteps} ===")

            # Get current state
            current_state = self._obs_to_state(obs)

            # Store the state before planning (for restoration after optimization)
            # Note: During optimization, the environment state will be modified by rollouts
            # We need to save and restore it for each MPC step
            saved_env_state = self.env.state_0  # Rewarped's current state

            # Plan optimal action using differentiable MPC
            # Note: Enable gradient mode in environment
            with torch.enable_grad():
                # Temporarily enable gradients in environment if needed
                original_no_grad = self.env.no_grad
                self.env.no_grad = False

                optimal_action = self.dms_plan(current_state)

                # Restore gradient setting
                self.env.no_grad = original_no_grad

            # Restore environment state after optimization
            self.env.state_0 = saved_env_state

            # Execute optimal action in the real environment
            actions = optimal_action.unsqueeze(0).repeat(self.env.num_envs, 1)
            obs, reward, done, info = self.env.step(actions)

            self.obs = self._convert_obs(obs)
            self.dones = done
            total_reward += reward[0].item() if isinstance(reward, torch.Tensor) else reward[0]

            # Save action
            best_actions_list.append(optimal_action.clone())

            print(f"Action executed: {optimal_action.cpu().numpy()}")
            print(f"Reward: {reward[0]:.4f}")

        print(f"\n=== Evaluation Complete ===")
        print(f"Total reward: {total_reward:.3f}")

        # Save trajectory
        best_actions_tensor = torch.stack(best_actions_list)
        torch.save(best_actions_tensor, 'trajectory_diff.pt')
        print("Trajectory saved to trajectory_diff.pt")

    def replay_trajectory(self):
        """Replay saved trajectory and generate visualizations."""
        # Initialize environment
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)

        rewards = []
        actions_to_replay = torch.load('trajectory_diff.pt')

        print("Replaying trajectory...")
        for timestep, action in enumerate(actions_to_replay):
            actions = action.unsqueeze(0).repeat(self.num_actors, 1)
            obs, reward, done, info = self.env.step(actions)
            self.obs = self._convert_obs(obs)
            self.dones = done

            reward_val = reward[0].item() if isinstance(reward, torch.Tensor) else reward[0]
            rewards.append(reward_val)

            print(f"Timestep {timestep + 1} | Action: {action.cpu().numpy()} | Reward: {reward_val:.3f}")

        print("Trajectory replay complete")

        # Plot rewards
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, marker='o', linewidth=2, markersize=4)
        plt.xlabel("Timestep", fontsize=12)
        plt.ylabel("Reward", fontsize=12)
        plt.title("Differentiable DMS MPC on Rewarped Deformable Task", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("reward_plot_diff.png", dpi=300)
        print("Saved reward plot to reward_plot_diff.png")

        # Create animation
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, len(rewards))
        ax.set_ylim(min(rewards) - 0.1 * abs(min(rewards)), max(rewards) + 0.1 * abs(max(rewards)))
        ax.set_xlabel("Timestep", fontsize=12)
        ax.set_ylabel("Reward", fontsize=12)
        ax.set_title("Differentiable DMS MPC Reward Evolution", fontsize=14)
        ax.grid(True, alpha=0.3)

        (line,) = ax.plot([], [], 'b-o', lw=2, markersize=6, label="Step Reward")
        ax.legend(fontsize=10)

        xdata, ydata = [], []

        def init():
            line.set_data([], [])
            return (line,)

        def update(frame):
            xdata.append(frame)
            ydata.append(rewards[frame])
            line.set_data(xdata, ydata)
            return (line,)

        ani = animation.FuncAnimation(
            fig, update, frames=len(rewards), init_func=init, blit=True, interval=100, repeat=False
        )

        ani.save("reward_animation_diff.gif", writer="pillow", fps=10)
        print("Saved animation to reward_animation_diff.gif")

