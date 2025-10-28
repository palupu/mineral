import copy
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import warp as wp

from mineral.agents.agent import Agent


class CEMMPCAgent(Agent):
    r"""Constant Control Test Agent.

    This is a test agent for exploring robot control by applying constant
    control inputs. Use this to learn what each control dimension does by
    observing the resulting robot movement.

    Usage Examples:
        # Test with default control (zeros - no input)
        agent = CEMMPCAgent(cfg)
        agent.eval()

        # Test with custom constant control values
        agent = CEMMPCAgent(cfg)
        agent.set_constant_control([0.5, -0.3, 0.0, 1.0])
        agent.eval()

        # Test one control at a time to isolate effects
        agent.set_constant_control([1.0, 0.0, 0.0, 0.0])  # Test first control
        agent.eval()

        # Test all maximum values
        agent.set_constant_control([1.0] * action_dim)
        agent.eval()
    """

    def __init__(self, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.params = full_cfg.agent.params
        self.num_actors = self.params.num_actors
        self.max_agent_steps = int(self.params.max_agent_steps)
        self.render_results = self.params.render_results

        # Get simulation parameters
        self.cem_mpc_params = full_cfg.agent.cem_mpc_params
        self.N = self.cem_mpc_params.N
        self.timesteps = self.cem_mpc_params.timesteps

        super().__init__(full_cfg, **kwargs)

        self.obs = None
        self.dones = None

        # Constant control values for testing
        # Set these using set_constant_control() method
        self.constant_control_values = None  # Will be set based on action_dim after env init

    def set_constant_control(self, values):
        """Set constant control values for testing.

        Args:
            values: list, tuple, numpy array, or torch tensor of control values
                   Should match the action dimension of the environment

        Example:
            agent.set_constant_control([0.5, -0.3, 0.0, 1.0])  # 4D control
            agent.set_constant_control([0.0] * action_dim)      # All zeros
            agent.set_constant_control([1.0] * action_dim)      # All max
        """
        if isinstance(values, (list, tuple)):
            self.constant_control_values = torch.tensor(values, device=self.device, dtype=torch.float32)
        elif isinstance(values, np.ndarray):
            self.constant_control_values = torch.from_numpy(values).float().to(self.device)
        elif isinstance(values, torch.Tensor):
            self.constant_control_values = values.float().to(self.device)
        else:
            raise ValueError("values must be a list, tuple, numpy array, or torch tensor")

        print(f"Constant control values set to: {self.constant_control_values}")

    def clone_state(self, state):
        s = type(state)()  # create a new empty State object of the same type

        # Attributes to clone with wp.clone
        wp_clone_attrs = [
            "body_f", "body_q", "body_qd", "joint_q", "joint_qd", "mpm_C", "mpm_F", "mpm_F_trial",
            "mpm_grid_m", "mpm_grid_mv", "mpm_grid_v", "mpm_stress", "mpm_x", "mpm_v",
        ]

        # Attributes to deep copy
        deepcopy_attrs = [ "particle_f", "particle_q", "particle_qd"]

        for attr in wp_clone_attrs:
            if hasattr(state, attr):
                setattr(s, attr, wp.clone(getattr(state, attr)))
        for attr in deepcopy_attrs:
            if hasattr(state, attr):
                setattr(s, attr, copy.deepcopy(getattr(state, attr)))

        return s


    def eval(self):
        """Run the test agent with constant control."""
        if self.render_results:
            self.replay_trajectory()
        else:
            start_time = time.time()
            self.run_constant_control_test()
            end_time = time.time()
            print(f"Time taken: {end_time - start_time:.2f}s")

    def run_constant_control_test(self):
        """Run the test with constant control values."""
        # Initialise environment
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)
        total_reward = 0.0

        # Initialize constant control values if not set
        if self.constant_control_values is None:
            # Default: zeros (no control input)
            self.constant_control_values = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            print(f"\n{'='*60}")
            print("CONSTANT CONTROL TEST MODE")
            print(f"Action dimension: {self.action_dim}")
            print(f"Control values: {self.constant_control_values}")
            print(f"{'='*60}\n")

        # Create list to save actions
        actions_list = []

        # Main loop - apply constant control at each timestep
        for timestep in range(self.timesteps):
            if self.dones[0]:
                break  # Stop if environment is done

            # Use constant control values
            action = self.constant_control_values.clone()
            actions = action.unsqueeze(0) #.repeat(self.N, 1)    # Shape: (N, action_dim)

            # Step the environment forward
            obs, reward, done, _ = self.env.step(actions)
            self.obs = self._convert_obs(obs)
            self.dones = done
            total_reward += reward[0]

            # Save action
            actions_list.append(action.clone())

            print(f"[TEST] Timestep {timestep + 1} | Action: {action} | Reward: {reward[0]:.3f}")

        print("\n" + "="*60)
        print("Test complete")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Average reward: {total_reward/len(actions_list):.3f}")
        print("="*60 + "\n")

        # Save trajectory as PyTorch file
        actions_tensor = torch.stack(actions_list)
        torch.save(actions_tensor, 'test_trajectory.pt')
        print("Test trajectory saved to 'test_trajectory.pt'")

    def replay_trajectory(self, trajectory_file='test_trajectory.pt'):
        """Replay a saved trajectory and visualize results."""
        # Initialise environment
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)

        print(f"\nReplaying trajectory from '{trajectory_file}'...")

        rewards = []
        actions_to_replay = torch.load(trajectory_file)

        timestep = 0
        for action in actions_to_replay:
            obs, reward, done, _ = self.env.step(action)
            self.obs = self._convert_obs(obs)
            self.dones = done

            print(f"Timestep {timestep + 1} | Action: {action} | Reward: {reward[0]:.3f}")
            timestep += 1
            rewards.append(reward[0].item())

        print("\nReplay complete!")
        print(f"Total reward: {sum(rewards):.3f}")
        print(f"Average reward: {np.mean(rewards):.3f}")

        # Plot static reward curve
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, linewidth=2)
        plt.xlabel("Timestep", fontsize=12)
        plt.ylabel("Reward", fontsize=12)
        plt.title("Constant Control Test - Reward over Time", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("test_reward_plot.png", dpi=300)
        print("\nStatic plot saved to 'test_reward_plot.png'")

        # Animated plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, len(rewards))
        reward_range = max(rewards) - min(rewards)
        ax.set_ylim(min(rewards) - 0.1 * reward_range, max(rewards) + 0.1 * reward_range)
        ax.set_xlabel("Timestep", fontsize=12)
        ax.set_ylabel("Reward", fontsize=12)
        ax.set_title("Constant Control Test - Reward Animation", fontsize=14)
        ax.grid(True, alpha=0.3)

        line, = ax.plot([], [], lw=2, label="Step Reward")
        ax.legend()

        xdata, ydata = [], []

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            xdata.append(frame)
            ydata.append(rewards[frame])
            line.set_data(xdata, ydata)
            return line,

        ani = animation.FuncAnimation(
            fig, update, frames=len(rewards),
            init_func=init, blit=True, interval=100, repeat=False
        )

        # Save animation as GIF
        ani.save("test_reward_animation.gif", writer="pillow", fps=30)
        print("Animation saved to 'test_reward_animation.gif'")
