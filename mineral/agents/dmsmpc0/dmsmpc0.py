import copy
import json
import os
import pickle
import time
from datetime import datetime, timedelta

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import warp as wp
from scipy.optimize import minimize

from mineral.agents.agent import Agent


class DMSMPC0Agent(Agent):
    r"""Direct Multiple Shooting Model Predictive Control."""

    def __init__(self, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.params = full_cfg.agent.params
        self.num_actors = self.params.num_actors
        self.max_agent_steps = int(self.params.max_agent_steps)
        self.render_results = self.params.render_results

        # Collect all DMS MPC Parameters
        self.dms_mpc_params = full_cfg.agent.dms_mpc_params
        # self.dt = 1.0 / 30.0 # Time step for rk4
        self.dt = 0.1 # Time step for H
        self.N = self.dms_mpc_params.N  # Number of shooting nodes
        self.H = self.dms_mpc_params.N * self.dt  # Horizon length
        self.timesteps = self.dms_mpc_params.timesteps
        self.max_iter = self.dms_mpc_params.max_iter  # Max optimization iterations

        # Cost parameters
        self.cost_state = self.dms_mpc_params.cost_state
        self.cost_control = self.dms_mpc_params.cost_control
        self.cost_terminal = self.dms_mpc_params.cost_terminal

        # Derived parameters
        self.state_dim = self.dms_mpc_params.state_dim
        self.control_dim = self.dms_mpc_params.control_dim
        self.dim = self.state_dim + self.control_dim

        super().__init__(full_cfg, **kwargs)

        self.obs = None
        self.dones = None

    def clone_state(self, state):
        s = type(state)()  # create a new empty State object of the same type

        # Attributes to clone with wp.clone
        wp_clone_attrs = [
            "body_f",
            "body_q",
            "body_qd",
            "joint_q",
            "joint_qd",
            "mpm_C",
            "mpm_F",
            "mpm_F_trial",
            "mpm_grid_m",
            "mpm_grid_mv",
            "mpm_grid_v",
            "mpm_stress",
            "mpm_x",
            "mpm_v",
        ]

        # Attributes to deep copy
        deepcopy_attrs = ["particle_f", "particle_q", "particle_qd"]

        for attr in wp_clone_attrs:
            if hasattr(state, attr):
                setattr(s, attr, wp.clone(getattr(state, attr)))
        for attr in deepcopy_attrs:
            if hasattr(state, attr):
                setattr(s, attr, copy.deepcopy(getattr(state, attr)))

        return s

    def _obs_to_state(self, obs):
        """Convert observation to state vector for DMS MPC."""
        # Handle RollingFlat environment observations
        if isinstance(obs, dict):
            # Extract joint_q and com_q from RollingFlat observations
            joint_q = obs.get('joint_q', torch.zeros(3, device=self.device))
            com_q = obs.get('com_q', torch.zeros(3, device=self.device))

            # Convert to numpy if they're torch tensors
            if isinstance(joint_q, torch.Tensor):
                joint_q = joint_q.detach().cpu().numpy()
            if isinstance(com_q, torch.Tensor):
                com_q = com_q.detach().cpu().numpy()

            # Flatten and concatenate
            state = np.concatenate([joint_q.flatten(), com_q.flatten()])
            return state[: self.state_dim]  # Ensure we don't exceed expected dimensions

        # Fallback for other observation formats
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()

        # Take the first state_dim elements as the state
        return obs[: self.state_dim]

    def eval(self):
        # render_results
        # True: use saved trajectory and render animation
        # False: perform evaluation with CEM MPC
        if self.render_results:
            self.replay_trajectory()

        else:
            start_time = time.time()
            self.run_dms_mpc()
            end_time = time.time()
            print(f"Time taken: {end_time - start_time}")

    def rk4_step(self, x_u, f, dt):
        """Perform a single step of the Runge-Kutta method."""
        x = x_u[: self.state_dim]
        u = x_u[self.state_dim :]

        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)

        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def dynamics(self, x, u):
        """Calculate the derivative of the state with respect to the state and control input."""
        # For RollingFlat environment: x = [joint_q(3), com_q(3)], u = [dx, dy, ry]
        # Simple dynamics model: joint_q_dot = u, com_q_dot = u (simplified)
        if len(x) == 6 and len(u) == 3:  # RollingFlat dimensions
            # Simple integrator model for RollingFlat
            # joint_q_dot = u (joint velocities = control inputs)
            # com_q_dot = u (center of mass velocity = control inputs, simplified)
            return np.concatenate([u, u])
        elif len(x) == 2:  # Simple 2D state (position, velocity)
            return np.array([x[1], u[0]])  # Simple double integrator
        else:
            # For other cases, use zero dynamics (placeholder)
            return np.zeros_like(x)

    def cost(self, x_u, current_x):
        """Cost function for the optimization problem using PyTorch for automatic differentiation."""
        # Convert numpy array to torch tensor with gradient tracking
        device = self.device
        x_u_torch = torch.tensor(x_u, dtype=torch.float64, requires_grad=True, device=device)

        # Reshape into state and control sequences
        last_x = x_u_torch[-self.state_dim :]
        x_u_torch_short = x_u_torch[: -self.state_dim].reshape(self.N, self.dim)

        x = torch.cat([x_u_torch_short[:, : self.state_dim], last_x.unsqueeze(0)], dim=0)
        u = x_u_torch_short[:, self.state_dim :]

        # Calculate stage costs using matrix operations
        # For RollingFlat: state_dim=6, control_dim=3
        Q = torch.diag(torch.tensor([self.cost_state] * self.state_dim, device=device, dtype=torch.float64))
        R = torch.diag(torch.tensor([self.cost_control] * self.control_dim, device=device, dtype=torch.float64))
        QN = torch.diag(torch.tensor([self.cost_terminal] * self.state_dim, device=device, dtype=torch.float64))

        cost_value = sum(x[i].T @ Q @ x[i] + u[i].T @ R @ u[i] for i in range(self.N))
        cost_value += x[self.N].T @ QN @ x[self.N]

        # Compute gradients
        cost_value.backward()

        return cost_value.item(), x_u_torch.grad.numpy()

    def eq_constraint(self, x_u, current_x):
        """Equality constraints for the optimization problem."""
        # Extract states and controls from the trajectory
        x = x_u[: -self.state_dim].reshape(self.N, self.dim)[:, : self.state_dim]
        x = np.vstack([x, x_u[-self.state_dim :]])
        u = x_u[: -self.state_dim].reshape(self.N, self.dim)[:, self.state_dim :]

        constraints = []

        # Initial constraint
        constraints.extend(x[0] - current_x)

        # Continuity constraints
        X = x[:-1]
        U = u
        X_next = x[1:]

        XU = np.hstack([X, U])
        X_next_pred = np.array([self.rk4_step(xu_i, self.dynamics, self.dt) for xu_i in XU])
        constraints.extend((X_next_pred - X_next).flatten())

        return np.array(constraints)

    def dms_plan(self, current_x):
        """Direct Multiple Shooting MPC planning."""
        # Initialize state and control vector
        x_u = np.zeros((self.state_dim + self.control_dim) * self.N + self.state_dim, dtype=np.float64)

        # Initialize state trajectory with current state
        for i in range(self.N):
            x_u[i * (self.state_dim + self.control_dim) : i * (self.state_dim + self.control_dim) + self.state_dim] = current_x

        x_u[self.N * self.dim : self.N * self.dim + self.state_dim] = current_x

        # Define bounds for RollingFlat environment
        # Control bounds: dx, dy, ry (translation and rotation)
        control_bounds = [(None, None), (None, None), (-2, 2)]  # dx, dy, ry bounds
        state_bounds = [(None, None)] * self.state_dim  # No bounds on states

        # Repeat bounds for each shooting node
        bounds = (control_bounds + state_bounds) * self.N + state_bounds

        # Create constraint wrapper
        def eq_constraint_wrapper(x_u):
            return self.eq_constraint(x_u, current_x)

        # Define constraints
        eq_cons = {'type': 'eq', 'fun': eq_constraint_wrapper}

        # Optimize
        result = minimize(
            lambda x_u: self.cost(x_u, current_x),
            x_u,
            method='SLSQP',
            jac=True,
            bounds=bounds,
            constraints=[eq_cons],
            options={'disp': False, 'maxiter': self.max_iter},
        )

        # Extract the first control input
        action = result.x[self.state_dim : self.state_dim + self.control_dim]
        return torch.tensor(action, device=self.device, dtype=torch.float32)

    def run_dms_mpc(self):
        # Initialise environment
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)
        total_reward = 0.0

        # Create list to save actions
        best_actions_list = []

        # Convert initial observation to state
        current_x = self._obs_to_state(obs)

        # Main loop
        for timestep in range(self.timesteps):
            if self.dones[0]:
                break  # Stop if real environment is done

            # Plan using DMS MPC
            best_action = self.dms_plan(current_x)
            actions = best_action.unsqueeze(0).repeat(self.num_actors, 1)  # Shape: (num_actors, action_dim)

            # Step the environment forward
            obs, reward, done, _ = self.env.step(actions)
            self.obs = self._convert_obs(obs)
            self.dones = done
            total_reward += reward[0]

            # Update current state for next iteration
            current_x = self._obs_to_state(obs)

            # Append actions to best action list
            best_actions_list.append(best_action.clone())

            print(f"Timestep {timestep + 1} | Action: {best_action} | Reward: {reward[0]:.3f}")

        print("Evaluation complete")
        print(f"Total reward: {total_reward:.3f}")

        # Save trajectory as PyTorch file
        best_actions_tensor = torch.stack(best_actions_list)
        torch.save(best_actions_tensor, 'trajectory.pt')
        print("Trajectory saved")

    def replay_trajectory(self):
        # Initialise environment
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)

        rewards = []
        actions_to_replay = torch.load('trajectory.pt')

        timestep = 0
        for action in actions_to_replay:
            obs, reward, done, _ = self.env.step(action)
            self.obs = self._convert_obs(obs)
            self.dones = done

            print(f"Timestep {timestep + 1} | Action: {action} | Reward: {reward[0]:.3f}")
            timestep += 1
            rewards.append(reward[0].item())
        print("Trajectory completed")

        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.xlabel("Timestep")
        plt.ylabel("Reward")
        plt.title("DMS MPC on Rewarped RollingFlat Task")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("reward_plot.png", dpi=300)

        # --- Animated Plot ---
        fig, ax = plt.subplots()
        ax.set_xlim(0, len(rewards))
        ax.set_ylim(min(rewards) - 0.1, max(rewards) + 0.1)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Reward")
        ax.set_title("DMS MPC on Rewarped RollingFlat Task")

        (line,) = ax.plot([], [], lw=2, label="Step Reward")
        ax.legend()

        xdata, ydata = [], []

        def init():
            line.set_data([], [])
            return (line,)

        def update(frame):
            xdata.append(frame)
            ydata.append(rewards[frame])
            line.set_data(xdata, ydata)
            return (line,)

        ani = animation.FuncAnimation(fig, update, frames=len(rewards), init_func=init, blit=True, interval=100, repeat=False)

        # Save animation as GIF
        ani.save("reward_animation.gif", writer="pillow", fps=30)
