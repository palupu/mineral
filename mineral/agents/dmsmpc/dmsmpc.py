import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import warp as wp
from scipy.optimize import minimize

from mineral.agents.agent import Agent


class DMSMPCAgent(Agent):
    r"""Direct Multiple Shooting Model Predictive Control with Real Physics.

    This implementation uses the actual physics simulator for trajectory rollouts
    instead of analytical dynamics models. This provides much better accuracy for
    complex physics like MPM-based deformable objects, at the cost of higher
    computational expense.

    Key differences from analytical DMS-MPC:
    - Uses clone_state() to save/restore full simulation state
    - Simulates trajectories using env.step() with real physics
    - More accurate for soft-body/plasticine deformation tasks
    """

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
        """Clone state by copying all warp arrays and MPM structures.

        This is based on the official Model_state implementation in rewarped
        (see rewarped/warp/model_monkeypatch.py::Model_state lines 93-116)
        but clones from an existing state instead of the model template.

        For MPM environments, mpm_particle and mpm_grid are special objects
        with their own .clone() methods. The individual arrays (mpm_x, mpm_v, etc.)
        are just references to mpm_particle attributes.
        """
        s = type(state)()  # create a new empty State object of the same type

        # Handle MPM particle and grid objects (they have .clone() methods)
        if hasattr(state, 'mpm_particle') and state.mpm_particle is not None:
            s.mpm_particle = state.mpm_particle.clone(requires_grad=False)
            # Add references to particle attributes
            s.mpm_x = s.mpm_particle.x
            s.mpm_v = s.mpm_particle.v
            s.mpm_C = s.mpm_particle.C
            s.mpm_F_trial = s.mpm_particle.F_trial
            s.mpm_F = s.mpm_particle.F
            s.mpm_stress = s.mpm_particle.stress

        if hasattr(state, 'mpm_grid') and state.mpm_grid is not None:
            s.mpm_grid = state.mpm_grid.clone(requires_grad=False)
            # Add references to grid attributes
            s.mpm_grid_v = s.mpm_grid.v
            s.mpm_grid_mv = s.mpm_grid.mv
            s.mpm_grid_m = s.mpm_grid.m

        # Clone all regular warp array attributes (bodies, joints, particles)
        for attr_name in dir(state):
            if attr_name.startswith('_') or attr_name.startswith('mpm_'):
                continue  # Skip private attrs and mpm attrs (already handled)
            attr = getattr(state, attr_name)
            if isinstance(attr, wp.array):
                setattr(s, attr_name, wp.clone(attr))

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
            obs = obs.detach().cpu().numpy()

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

    def simulate_trajectory(self, init_state, controls):
        """Simulate a trajectory using the real physics engine.

        Args:
            init_state: Initial environment state
            controls: Array of control inputs, shape (N, control_dim)

        Returns:
            states: List of states after each control is applied
        """
        states = []

        # Clone the initial state
        self.env.state_0 = self.clone_state(init_state)

        # Step through each control input
        for i in range(len(controls)):
            # Convert control to torch tensor and expand for all actors
            u = torch.tensor(controls[i], device=self.device, dtype=torch.float32)
            actions = u.unsqueeze(0).repeat(self.num_actors, 1)

            # Step the environment
            obs, _, _, _ = self.env.step(actions)

            # Extract state from observation
            state = self._obs_to_state(obs)
            states.append(state)

        return np.array(states)

    def simulate_single_step(self, init_state, control):
        """Simulate a single step using real physics.

        Args:
            init_state: Initial environment state
            control: Single control input, shape (control_dim,)

        Returns:
            next_state: State after applying control
        """
        # Clone the initial state
        self.env.state_0 = self.clone_state(init_state)

        # Convert control to torch tensor and expand for all actors
        u = torch.tensor(control, device=self.device, dtype=torch.float32)
        actions = u.unsqueeze(0).repeat(self.num_actors, 1)

        # Step the environment
        obs, _, _, _ = self.env.step(actions)

        # Extract state from observation
        next_state = self._obs_to_state(obs)

        return next_state

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

    def eq_constraint(self, x_u, current_x, init_state):
        """Equality constraints for the optimization problem using real physics.

        This implements a simplified multiple shooting where we simulate the entire
        trajectory from the initial state and check continuity.

        Args:
            x_u: Decision variables [x0, u0, x1, u1, ..., x_{N-1}, u_{N-1}, xN]
            current_x: Current observed state
            init_state: Initial full environment state
        """
        # Extract states and controls from the trajectory
        x = x_u[: -self.state_dim].reshape(self.N, self.dim)[:, : self.state_dim]
        x = np.vstack([x, x_u[-self.state_dim :]])
        u = x_u[: -self.state_dim].reshape(self.N, self.dim)[:, self.state_dim :]

        constraints = []

        # Initial constraint: first state must match current state
        constraints.extend(x[0] - current_x)

        # Continuity constraints: simulate trajectory and ensure consistency
        # Clone initial state for simulation
        self.env.state_0 = self.clone_state(init_state)

        for i in range(self.N):
            # Simulate one step with control u[i] from current environment state
            next_state_sim = self.simulate_single_step(self.env.state_0, u[i])

            # The constraint is: x[i+1] should equal the simulated next state
            constraints.extend(x[i + 1] - next_state_sim)

            # Note: In this simplified version, we continue from the simulated state
            # In true multiple shooting, we would "shoot" to x[i+1] instead
            # But since we can't directly set arbitrary states, we accept this approximation

        return np.array(constraints)

    def dms_plan(self, current_x, init_state):
        """Direct Multiple Shooting MPC planning with real physics.

        Args:
            current_x: Current observed state vector
            init_state: Current full environment state
        """
        # Initialize state and control vector
        x_u = np.zeros((self.state_dim + self.control_dim) * self.N + self.state_dim, dtype=np.float64)

        # Initialize state trajectory with current state
        for i in range(self.N):
            x_u[i * (self.state_dim + self.control_dim) : i * (self.state_dim + self.control_dim) + self.state_dim] = current_x

        x_u[self.N * self.dim : self.N * self.dim + self.state_dim] = current_x

        # Define bounds for RollingFlat environment
        # Control bounds: dx, dy, ry (translation and rotation)
        control_bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]  # Normalized action bounds
        state_bounds = [(None, None)] * self.state_dim  # No bounds on states

        # Repeat bounds for each shooting node
        bounds = (state_bounds + control_bounds) * self.N + state_bounds

        # Save the original state
        original_state = self.clone_state(init_state)

        # Create constraint wrapper that includes init_state
        def eq_constraint_wrapper(x_u):
            return self.eq_constraint(x_u, current_x, init_state)

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
            options={'disp': True, 'maxiter': self.max_iter, 'ftol': 1e-6},
        )

        # Restore the original state after optimization
        self.env.state_0 = original_state

        # Extract the first control input
        action = result.x[self.state_dim : self.state_dim + self.control_dim]

        print(f"Optimization success: {result.success}, message: {result.message}")
        print(f"Final cost: {result.fun}, iterations: {result.nit}")

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

            # Save the current full state
            init_state = self.clone_state(self.env.state_0)

            # Plan using DMS MPC with full state
            best_action = self.dms_plan(current_x, init_state)
            actions = best_action.unsqueeze(0).repeat(self.num_actors, 1)  # Shape: (num_actors, action_dim)

            # Restore state before executing the action
            self.env.state_0 = init_state

            # Step the environment forward with the best action
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
