"""TRUE Direct Multiple Shooting MPC with PyTorch Adam Optimizer.

This implementation combines TRUE Direct Multiple Shooting (DMS) with penalty-based
optimization using PyTorch's Adam optimizer and differentiable physics simulation.

Key Features:
1. TRUE DMS: Each shooting node x[i] is set independently before simulating to x[i+1]
2. PyTorch Adam: First-order gradient-based optimization (replaces scipy SLSQP)
3. Penalty Method: Hard constraints converted to soft penalties in loss function
4. Analytical Gradients: All gradients via PyTorch autograd through Rewarped
5. Full State Representation: All MPM particle positions + rigid body configuration

Direct Multiple Shooting with Penalty Method:
- Loss = task_cost + penalty_weight * ||x[i+1] - f(x[i], u[i])||^2
- Each interval independently shoots from x[i] to verify x[i+1]
- No hard constraints; penalties guide optimizer to feasible solutions
- More flexible and scalable than constrained optimization

Gradient Computation:
- Combined loss function: Automatic differentiation with PyTorch (.backward())
- Adam optimizer uses first-order gradients only (no Hessian/Jacobian needed)
- Differentiable physics simulation via Rewarped enables end-to-end gradients

State Representation:
- Uses ALL MPM particle positions (2592 particles for RollingPin)
- Includes rigid body configuration (roller position and orientation)
- State vector: [x1, y1, z1, ..., xN, yN, zN, body_x, body_y, body_z, qw, qx, qy, qz]
- No dimensionality reduction for mathematical correctness
"""

import time
import warnings
from typing import Any, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import warp as wp
# from scipy.optimize import minimize

from mineral.agents.agent import Agent

# warnings.simplefilter('always')
warnings.filterwarnings('ignore', message='.*grad attribute of a Tensor that is not a leaf.*')


class TrueDMSMPCAgent(Agent):
    r"""TRUE Direct Multiple Shooting MPC with PyTorch Adam Optimizer.

    This agent combines TRUE Direct Multiple Shooting (DMS) with penalty-based
    optimization using PyTorch's Adam optimizer. Hard dynamics constraints are
    converted to soft penalties, allowing efficient first-order optimization.

    Optimization Problem (Penalty Method):
        minimize_{x, u}  L(x, u) = task_cost(x, u) + λ * constraint_penalty(x, u)

        where:
            task_cost = Σ cost(x[i], u[i]) + terminal_cost(x[N])
            constraint_penalty = ||x[0] - x_current||^2 + Σ ||x[i+1] - f(x[i], u[i])||^2
            λ = penalty_weight (large value enforces constraints)

    State Representation:
        - State x: ALL MPM particle positions + rigid body configuration
        - Dimension: state_dim = num_particles × 3 + 7 (particle coords + body_q)
        - body_q: [x, y, z, qw, qx, qy, qz] (position + quaternion orientation)
        - Example: 2592 particles → 7783-dimensional state vector
        - Control u: 3D input [dx, dy, ry] for RollingPin task

    Optimization Approach:
        - Optimizer: PyTorch Adam (first-order, adaptive learning rate)
        - Gradients: ∇_xu L computed via PyTorch autograd (loss.backward())
        - No Jacobian/Hessian needed: Adam only requires first-order gradients
        - Leverages Rewarped's differentiable physics for exact gradients
    """

    def __init__(self, full_cfg: Any, **kwargs: Any) -> None:
        self.network_config = full_cfg.agent.network
        self.params = full_cfg.agent.params
        self.num_actors = self.params.num_actors
        self.max_agent_steps = int(self.params.max_agent_steps)
        self.render_results = self.params.render_results

        # DMS MPC Parameters
        self.dms_mpc_params = full_cfg.agent.dms_mpc_params
        self.dt = 0.1
        self.N = self.dms_mpc_params.N  # Number of shooting nodes
        self.H = self.dms_mpc_params.N * self.dt  # Horizon length
        self.timesteps = self.dms_mpc_params.timesteps
        self.max_iter = self.dms_mpc_params.max_iter

        # Cost parameters
        self.cost_state = self.dms_mpc_params.cost_state
        self.cost_control = self.dms_mpc_params.cost_control
        self.cost_terminal = self.dms_mpc_params.cost_terminal
        self.penalty_weight = self.dms_mpc_params.get('penalty_weight', 1000.0)  # Penalty for constraint violations
        self.learning_rate = self.dms_mpc_params.get('learning_rate', 0.01)  # Adam learning rate

        # State and control dimensions - determined dynamically after env init
        # self.control_dim = 3  # Fixed: dx, dy, ry for RollingPin
        self.state_dim = None  # Will be set after first observation
        self.num_particles = None  # Number of MPM particles
        self.mpm_pos_dim = None  # MPM particle positions (num_particles * 3)
        self.mpm_vel_dim = None  # MPM particle velocities (num_particles * 3)
        self.body_q_dim = None  # Body configuration dimension (7 for [x,y,z,qw,qx,qy,qz])
        self.body_qd_dim = None  # Body velocity dimension (6 for [vx,vy,vz,wx,wy,wz])
        self.dim = None  # Will be state_dim + control_dim

        super().__init__(full_cfg, **kwargs)

        # Get control dimension from parent Agent class
        self.control_dim = self.action_dim

        self.obs = None
        self.dones = None

        # Diagnostic attributes for loss breakdown
        self._last_task_cost = 0.0
        self._last_constraint_penalty = 0.0

    def clone_state(self, state: Any) -> Any:
        """Clone state by copying all warp arrays and MPM structures.

        This creates a complete independent copy of the simulation state,
        including all MPM particles, grid data, and rigid body states.
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

        # Ensure body_q is cloned (rigid body position and orientation: [x, y, z, qw, qx, qy, qz])
        if hasattr(state, 'body_q') and state.body_q is not None:
            s.body_q = wp.clone(state.body_q)

        return s

    def _obs_to_state(self, return_torch: bool = False):
        """Extract full state vector from simulation (positions + velocities + body config).

        For TRUE DMS, we use COMPLETE state including velocities:
        - All MPM particle positions and velocities
        - Rigid body position, orientation, and velocities

        RollingPin: 2592 particles in simulation, 250 in observations
        We use ALL 2592 for TRUE DMS to be mathematically correct.

        State vector: [x1,y1,z1,...,xN,yN,zN, vx1,vy1,vz1,...,vxN,vyN,vzN, body_q (7), body_qd (6)]

        Args:
            return_torch: If True, return torch.Tensor (for differentiable ops).
                         If False, return numpy array (for compatibility).

        Returns:
            state: State vector as numpy array or torch tensor
        """
        # Extract ALL particles from the full simulation state
        # NOTE: We do NOT use obs['particle_q'] because it's downsampled (250 particles)
        # Instead, we access the full simulation state (2592 particles)

        if not hasattr(self.env, 'state_0'):
            raise ValueError("Environment does not have state_0 attribute!")

        full_state = self.env.state_0

        if not hasattr(full_state, 'mpm_x') or full_state.mpm_x is None:
            raise ValueError("State does not have mpm_x attribute for particle positions")

        # Extract particle positions
        mpm_x = wp.to_torch(full_state.mpm_x)
        if mpm_x.device != self.device:
            mpm_x = mpm_x.to(self.device)

        # Handle batch/environment dimension
        if mpm_x.ndim == 3:
            mpm_x = mpm_x[0]  # Extract first env
        elif mpm_x.ndim != 2:
            raise ValueError(f"Expected mpm_x with 2 or 3 dims, got {mpm_x.ndim}")

        mpm_x_flat = mpm_x.flatten()

        # Extract particle velocities
        if not hasattr(full_state, 'mpm_v') or full_state.mpm_v is None:
            raise ValueError("State does not have mpm_v attribute for particle velocities")

        mpm_v = wp.to_torch(full_state.mpm_v)
        if mpm_v.device != self.device:
            mpm_v = mpm_v.to(self.device)

        if mpm_v.ndim == 3:
            mpm_v = mpm_v[0]
        mpm_v_flat = mpm_v.flatten()

        # Extract body_q (rigid body position and orientation: [x, y, z, qw, qx, qy, qz])
        state_components = [mpm_x_flat, mpm_v_flat]

        if hasattr(full_state, 'body_q') and full_state.body_q is not None:
            body_q = wp.to_torch(full_state.body_q)
            if body_q.device != self.device:
                body_q = body_q.to(self.device)
            if body_q.ndim == 2:
                body_q = body_q[0]  # Extract first body
            state_components.append(body_q)

        # Extract body_qd (rigid body velocity: [vx, vy, vz, wx, wy, wz])
        if hasattr(full_state, 'body_qd') and full_state.body_qd is not None:
            body_qd = wp.to_torch(full_state.body_qd)
            if body_qd.device != self.device:
                body_qd = body_qd.to(self.device)
            if body_qd.ndim == 2:
                body_qd = body_qd[0]  # Extract first body
            state_components.append(body_qd)

        # Concatenate all components
        state = torch.cat(state_components)

        # Initialize dimensions if this is the first call
        if self.state_dim is None:
            self.num_particles = mpm_x.shape[0]
            self.mpm_pos_dim = self.num_particles * 3
            self.mpm_vel_dim = self.num_particles * 3
            self.body_q_dim = 7 if hasattr(full_state, 'body_q') and full_state.body_q is not None else 0
            self.body_qd_dim = 6 if hasattr(full_state, 'body_qd') and full_state.body_qd is not None else 0
            self.state_dim = self.mpm_pos_dim + self.mpm_vel_dim + self.body_q_dim + self.body_qd_dim
            self.dim = self.state_dim + self.control_dim
            print(f"\n{'=' * 70}")
            print("State Dimensions Initialized (FULL STATE + VELOCITIES):")
            print(f"  Number of particles: {self.num_particles}")
            print(f"  MPM positions: {self.mpm_pos_dim} (particles × 3)")
            print(f"  MPM velocities: {self.mpm_vel_dim} (particles × 3)")
            print(f"  Body config (q): {self.body_q_dim} (x, y, z, qw, qx, qy, qz)")
            print(f"  Body velocity (qd): {self.body_qd_dim} (vx, vy, vz, wx, wy, wz)")
            print(f"  Total state dimension: {self.state_dim}")
            print(f"  Control dimension: {self.control_dim}")
            print(f"  Total decision vars per node: {self.dim}")
            print(f"{'=' * 70}\n")

        # Return torch tensor or numpy array based on flag
        if return_torch:
            return state
        else:
            return state.detach().cpu().numpy()

    def _set_state_from_vector(self, state_vec, template_state: Any, zero_internal_states: bool = True) -> Any:
        """Set full environment state from state vector (positions + velocities + body config).

        This is THE KEY FUNCTION for true DMS. It allows the optimizer to
        propose arbitrary states x[i] at each shooting node, which we then
        set in the simulator before shooting to x[i+1].

        Args:
            state_vec: Full state vector, shape (state_dim,)
                      Can be numpy array or torch.Tensor
                      Contains: [mpm_x, mpm_v, body_q, body_qd]
            template_state: Full simulation state to use as template
            zero_internal_states: If True, zero out MPM internal states (C, F, stress)
                                 Should be True for shooting nodes, False for initial state

        Returns:
            new_state: Full simulation state with updated positions/velocities
        """
        # Verify template state
        if not hasattr(template_state, 'mpm_x') or template_state.mpm_x is None:
            raise ValueError("Template state does not have mpm_x attribute")

        # Get expected dimensions
        num_particles = template_state.mpm_x.shape[0]
        mpm_pos_dim = num_particles * 3
        mpm_vel_dim = num_particles * 3
        has_body_q = hasattr(template_state, 'body_q') and template_state.body_q is not None
        has_body_qd = hasattr(template_state, 'body_qd') and template_state.body_qd is not None
        body_q_dim = 7 if has_body_q else 0
        body_qd_dim = 6 if has_body_qd else 0
        expected_size = mpm_pos_dim + mpm_vel_dim + body_q_dim + body_qd_dim

        # Handle both numpy and torch inputs
        if isinstance(state_vec, np.ndarray):
            state_size = len(state_vec)
        else:  # torch.Tensor
            state_size = state_vec.numel()

        # Check dimensions match
        if state_size != expected_size:
            raise ValueError(
                f"State vector size mismatch! "
                f"state_vec has {state_size} elements, "
                f"but expected {expected_size} (mpm_x: {mpm_pos_dim}, mpm_v: {mpm_vel_dim}, "
                f"body_q: {body_q_dim}, body_qd: {body_qd_dim})"
            )

        # Clone full state
        new_state = self.clone_state(template_state)

        # Convert to torch tensor if needed
        if isinstance(state_vec, np.ndarray):
            state_vec_torch = torch.from_numpy(state_vec.astype(np.float32)).to(self.device)
        else:
            state_vec_torch = state_vec
            if state_vec_torch.dtype != torch.float32:
                state_vec_torch = state_vec_torch.float()
            if state_vec_torch.device != self.device:
                state_vec_torch = state_vec_torch.to(self.device)

        # Parse state vector
        idx = 0

        # Extract and set particle positions
        mpm_x_flat = state_vec_torch[idx:idx + mpm_pos_dim]
        particle_positions = mpm_x_flat.reshape(num_particles, 3)
        new_state.mpm_x.assign(wp.from_torch(particle_positions))
        idx += mpm_pos_dim

        # Extract and set particle velocities
        mpm_v_flat = state_vec_torch[idx:idx + mpm_vel_dim]
        particle_velocities = mpm_v_flat.reshape(num_particles, 3)
        new_state.mpm_v.assign(wp.from_torch(particle_velocities))
        idx += mpm_vel_dim

        # Extract and set body_q if present
        if has_body_q:
            body_q_vec = state_vec_torch[idx:idx + body_q_dim]
            new_state.body_q.assign(wp.from_torch(body_q_vec.unsqueeze(0)))  # Add batch dimension
            idx += body_q_dim

        # Extract and set body_qd if present
        if has_body_qd:
            body_qd_vec = state_vec_torch[idx:idx + body_qd_dim]
            new_state.body_qd.assign(wp.from_torch(body_qd_vec.unsqueeze(0)))  # Add batch dimension
            idx += body_qd_dim

        # Zero out internal MPM states for physical consistency
        # These should be recomputed by the simulator from positions/velocities
        if zero_internal_states:
            if hasattr(new_state, 'mpm_C') and new_state.mpm_C is not None:
                new_state.mpm_C.zero_()
            if hasattr(new_state, 'mpm_F') and new_state.mpm_F is not None:
                # F should be identity, not zero
                new_state.mpm_particle.init_F()
            if hasattr(new_state, 'mpm_stress') and new_state.mpm_stress is not None:
                new_state.mpm_stress.zero_()
            # Zero grid state (will be recomputed)
            if hasattr(new_state, 'mpm_grid') and new_state.mpm_grid is not None:
                new_state.mpm_grid.clear()

        return new_state

    def simulate_single_step(self, init_state: Any, control, return_torch: bool = False):
        """Simulate a single step using real physics.

        Args:
            init_state: Initial environment state (full simulation state)
            control: Single control input, shape (control_dim,) (numpy or torch)
            return_torch: If True, return torch.Tensor; else numpy array

        Returns:
            next_state: State after applying control (numpy array or torch tensor)
        """
        # Set the environment state
        self.env.state_0 = self.clone_state(init_state)

        # Convert control to torch tensor if needed
        if isinstance(control, np.ndarray):
            u = torch.tensor(control, device=self.device, dtype=torch.float32)
        else:  # Already torch.Tensor
            u = control
            if u.dtype != torch.float32:
                u = u.float()
            if u.device != self.device:
                u = u.to(self.device)

        # Expand for all actors
        actions = u.unsqueeze(0).repeat(self.num_actors, 1)

        # Step the environment (differentiable with Rewarped!)
        obs, _, _, _ = self.env.step(actions)

        # Extract state from simulation (env.state_0 updated after step)
        next_state = self._obs_to_state(return_torch=return_torch)

        return next_state

    def compute_loss(self, x_u: torch.Tensor, current_x: torch.Tensor, template_state: Any) -> torch.Tensor:
        """Combined loss function: task cost + penalty for constraint violations.

        This replaces hard constraints with soft penalties (penalty method).
        Loss = task_cost + penalty_weight * constraint_violations

        Args:
            x_u: Decision variables [x0, u0, x1, u1, ..., x_{N-1}, u_{N-1}, xN]
            current_x: Current observed state (for initial constraint)
            template_state: Template simulation state

        Returns:
            total_loss: Scalar tensor to minimize
        """
        # Reshape into state and control sequences
        last_x = x_u[-self.state_dim :]
        x_u_short = x_u[: -self.state_dim].reshape(self.N, self.dim)

        x = torch.cat([x_u_short[:, : self.state_dim], last_x.unsqueeze(0)], dim=0)
        u = x_u_short[:, self.state_dim :]

        # ========== TASK COST (VECTORIZED) ==========
        task_cost = self._compute_task_cost_vectorized(x, u)

        # ========== CONSTRAINT PENALTIES (VECTORIZED) ==========
        constraint_penalty = self._compute_constraint_penalty_vectorized(x, u, current_x, template_state)

        # Total loss
        total_loss = task_cost + self.penalty_weight * constraint_penalty

        # Store components for diagnostics
        self._last_task_cost = task_cost.item()
        self._last_constraint_penalty = constraint_penalty.item()

        return total_loss

    def _compute_task_cost_vectorized(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Vectorized computation of task cost (stage costs + terminal cost).

        Args:
            x: States, shape (N+1, state_dim)
            u: Controls, shape (N, control_dim)

        Returns:
            task_cost: Scalar tensor
        """
        h_ref = 0.125

        # ===== VECTORIZED STAGE COSTS =====
        # Extract positions from all N stage states at once: (N, mpm_pos_dim)
        x_stage = x[:self.N]  # (N, state_dim)
        mpm_x_stage = x_stage[:, :self.mpm_pos_dim]  # (N, mpm_pos_dim)

        # Reshape to (N, num_particles, 3)
        particles_stage = mpm_x_stage.reshape(self.N, self.num_particles, 3)

        # Extract heights (y-coordinates): (N, num_particles)
        heights_stage = particles_stage[:, :, 1]

        # Cost 1: Average height per node (N,)
        mean_heights = heights_stage.mean(dim=1)  # (N,)
        height_costs = (mean_heights / h_ref) ** 2  # (N,)

        # Cost 2: Height variance per node (N,)
        variance_costs = heights_stage.var(dim=1)  # (N,)

        # Cost 3: Control effort per node (N,)
        control_costs = self.cost_control * (u * u).sum(dim=1)  # (N,)

        # Total stage costs: sum over all N nodes
        stage_cost = self.cost_state * (height_costs.sum() + variance_costs.sum()) + control_costs.sum()

        # ===== TERMINAL COST =====
        mpm_x_final = x[self.N][:self.mpm_pos_dim]  # (mpm_pos_dim,)
        particles_final = mpm_x_final.reshape(self.num_particles, 3)
        heights_final = particles_final[:, 1]

        mean_height_final = heights_final.mean()
        variance_final = heights_final.var()

        terminal_cost = self.cost_terminal * ((mean_height_final / h_ref) ** 2 + variance_final)

        return stage_cost + terminal_cost

    def _compute_constraint_penalty_vectorized(
        self, x: torch.Tensor, u: torch.Tensor, current_x: torch.Tensor, template_state: Any
    ) -> torch.Tensor:
        """Vectorized computation of constraint penalties.

        Args:
            x: States, shape (N+1, state_dim)
            u: Controls, shape (N, control_dim)
            current_x: Current state
            template_state: Template simulation state

        Returns:
            constraint_penalty: Scalar tensor
        """
        # Initial constraint: first state must match current state
        initial_violation = torch.sum((x[0] - current_x) ** 2)

        # Dynamics constraints: vectorized simulation
        # Simulate all N shooting intervals in parallel
        next_states_sim = self._simulate_batch_parallel(x[:self.N], u, template_state)  # (N, state_dim)

        # Compute violations: ||x[i+1] - f(x[i], u[i])||^2 for all i
        dynamics_violations = torch.sum((x[1:self.N+1] - next_states_sim) ** 2, dim=1)  # (N,)
        dynamics_penalty = dynamics_violations.sum()

        return initial_violation + dynamics_penalty

    def _simulate_batch_parallel(
        self, x_batch: torch.Tensor, u_batch: torch.Tensor, template_state: Any
    ) -> torch.Tensor:
        """Simulate N shooting intervals in parallel using batched environments.

        Args:
            x_batch: Initial states for each interval, shape (N, state_dim)
            u_batch: Controls for each interval, shape (N, control_dim)
            template_state: Template simulation state

        Returns:
            next_states: Resulting states after simulation, shape (N, state_dim)
        """
        # TODO: This requires environment to support parallel execution with different initial states
        # For now, fall back to sequential execution (will be optimized later)

        next_states = []
        for i in range(self.N):
            state_i = self._set_state_from_vector(x_batch[i], template_state, zero_internal_states=(i > 0))
            next_state = self.simulate_single_step(state_i, u_batch[i], return_torch=True)
            next_states.append(next_state)

        return torch.stack(next_states)  # (N, state_dim)


    def dms_plan(self, current_x: np.ndarray, init_state: Any) -> torch.Tensor:
        """Direct Multiple Shooting MPC planning with PyTorch Adam optimizer.

        Uses penalty method: constraints are enforced as soft penalties in the loss.

        Args:
            current_x: Current observed state vector (numpy)
            init_state: Current full environment state (template for state setting)

        Returns:
            action: Optimal action tensor to execute (first control from trajectory)
        """
        # Ensure dimensions are properly set
        if self.state_dim is None or self.dim is None:
            raise ValueError("State dimensions not initialized! Call _obs_to_state first.")

        # Verify dimensions match current state
        if len(current_x) != self.state_dim:
            raise ValueError(
                f"State dimension mismatch! current_x has {len(current_x)} elements but self.state_dim={self.state_dim}"
            )

        # Convert current state to torch tensor
        current_x_torch = torch.tensor(current_x, dtype=torch.float32, device=self.device)

        # Initialize decision variables as torch tensor with gradient
        x_u_init = torch.zeros(
            (self.state_dim + self.control_dim) * self.N + self.state_dim,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )

        # Warm start: initialize state trajectory with current state
        # IMPORTANT: Initialize controls with small random values (not zero!)
        with torch.no_grad():
            for i in range(self.N):
                # Initialize states
                x_u_init[i * self.dim : i * self.dim + self.state_dim] = current_x_torch
                # Initialize controls with small random values
                control_start = i * self.dim + self.state_dim
                control_end = control_start + self.control_dim
                x_u_init[control_start:control_end] = torch.randn(self.control_dim, device=self.device) * 0.1
            # Final state
            x_u_init[self.N * self.dim : self.N * self.dim + self.state_dim] = current_x_torch

        # Save the original state
        original_state = self.clone_state(init_state)

        # Create optimizer
        x_u = torch.nn.Parameter(x_u_init.clone())
        optimizer = torch.optim.Adam([x_u], lr=self.learning_rate)

        print(f"\n{'=' * 70}")
        print("Starting TRUE DMS Optimization (PyTorch Adam + Penalty Method)")
        print(f"  Particles: {self.num_particles}")
        print(f"  Nodes: {self.N}, State dim: {self.state_dim}, Control dim: {self.control_dim}")
        print(f"  Decision variables: {len(x_u)}")
        print(f"  Penalty weight: {self.penalty_weight}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Max iterations: {self.max_iter}")
        print("  Using differentiable simulation with Adam optimizer ✓")

        # Print initial controls
        with torch.no_grad():
            initial_controls = []
            for i in range(self.N):
                control_start = i * self.dim + self.state_dim
                control_end = control_start + self.control_dim
                initial_controls.append(x_u[control_start:control_end].cpu().numpy())
            print(f"  Initial controls: {[f'[{c[0]:.3f} {c[1]:.3f} {c[2]:.3f}]' for c in initial_controls]}")
        print(f"{'=' * 70}\n")

        # Optimization loop
        start_time = time.time()
        best_loss = float('inf')
        best_x_u = None

        for iteration in range(self.max_iter):
            optimizer.zero_grad()

            # Compute loss (task cost + constraint penalties)
            loss = self.compute_loss(x_u, current_x_torch, init_state)

            # Backward pass
            loss.backward()

            # Check gradient norms for debugging
            if iteration % 10 == 0:
                grad_norm = x_u.grad.norm().item() if x_u.grad is not None else 0.0
                # Check control gradients specifically
                control_grads = []
                for i in range(self.N):
                    control_start = i * self.dim + self.state_dim
                    control_end = control_start + self.control_dim
                    if x_u.grad is not None:
                        control_grads.append(x_u.grad[control_start:control_end].norm().item())
                avg_control_grad = sum(control_grads) / len(control_grads) if control_grads else 0.0

            # Optimizer step (BEFORE clipping!)
            optimizer.step()

            # Apply control bounds by clipping AFTER optimizer step
            with torch.no_grad():
                for i in range(self.N):
                    control_start = i * self.dim + self.state_dim
                    control_end = control_start + self.control_dim
                    # Clip control values to [-1, 1]
                    x_u.data[control_start:control_end].clamp_(-1.0, 1.0)

            # Track best solution
            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_x_u = x_u.data.clone()

            # Print progress with gradient info and loss breakdown
            if iteration % 10 == 0 or iteration == self.max_iter - 1:
                print(f"Iteration {iteration:4d}/{self.max_iter} | Loss: {loss_val:.6e} | "
                      f"TaskCost: {self._last_task_cost:.4e} | ConstraintPenalty: {self._last_constraint_penalty:.4e} | "
                      f"Grad: {grad_norm:.4e} | ControlGrad: {avg_control_grad:.4e}", flush=True)

        elapsed_time = time.time() - start_time

        # Use best solution
        if best_x_u is not None:
            x_u.data = best_x_u

        # Restore the original state after optimization
        self.env.state_0 = original_state

        # Extract the first control input (MPC receding horizon principle)
        with torch.no_grad():
            action = x_u[self.state_dim : self.state_dim + self.control_dim].clone()

        # Report optimization results
        print(f"\n{'=' * 70}")
        print("Optimization Results")
        print(f"{'=' * 70}")
        print(f"  Final Loss: {best_loss:.6e}")
        print(f"  Iterations: {self.max_iter}")
        print(f"  Optimization Time: {elapsed_time:.2f}s")
        print(f"  Optimal Action: {action.cpu().numpy()}")
        print(f"{'=' * 70}\n")

        return action

    def eval(self) -> None:
        if self.render_results:
            self.replay_trajectory()
        else:
            start_time = time.time()
            self.run_dms_mpc()
            end_time = time.time()
            print(f"\nTotal time: {end_time - start_time:.2f}s")

    def run_dms_mpc(self) -> None:
        """Main MPC loop using TRUE DMS."""
        # Initialize environment
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)
        total_reward = 0.0

        # Create list to save actions
        best_actions_list = []

        # Extract initial state from simulation
        current_x = self._obs_to_state()

        print(f"\n{'#' * 70}")
        print(f"{'TRUE DIRECT MULTIPLE SHOOTING MPC (PyTorch Adam)':^70}")
        print(f"{'#' * 70}")
        print(f"Full State Representation: All {self.num_particles} MPM particles")
        print(f"Horizon: N={self.N} nodes, H={self.H:.2f}s")
        print(f"State dim: {self.state_dim}, Control dim: {self.control_dim}")
        print(f"Optimizer: Adam (lr={self.learning_rate}, penalty_weight={self.penalty_weight})")
        print(f"Timesteps: {self.timesteps}")
        print(f"{'#' * 70}\n")

        # Main loop
        for timestep in range(self.timesteps):
            if self.dones[0]:
                break

            print(f"\n{'─' * 70}")
            print(f"Timestep {timestep + 1}/{self.timesteps}")
            print(f"{'─' * 70}")

            # Save the current full state
            init_state = self.clone_state(self.env.state_0)

            # Disable USD recording during optimization (we only want actual executed steps recorded)
            saved_renderer = self.env.renderer
            self.env.renderer = None

            # Plan using TRUE DMS MPC
            best_action = self.dms_plan(current_x, init_state)
            actions = best_action.unsqueeze(0).repeat(self.num_actors, 1)

            # Re-enable USD recording for actual execution
            self.env.renderer = saved_renderer

            # Restore state before executing the action
            self.env.state_0 = init_state

            # Step the environment forward with the best action (this WILL be recorded)
            obs, reward, done, _ = self.env.step(actions)
            self.obs = self._convert_obs(obs)
            self.dones = done

            reward_val = reward[0].item() if isinstance(reward, torch.Tensor) else reward[0]
            total_reward += reward_val

            # Update current state for next iteration
            current_x = self._obs_to_state()

            # Append actions to best action list
            best_actions_list.append(best_action.clone())

            print(f"Action executed: {best_action.cpu().numpy()}")
            print(f"Reward: {reward_val:.4f} | Cumulative: {total_reward:.4f}")

        print(f"\n{'#' * 70}")
        print(f"{'EVALUATION COMPLETE':^70}")
        print(f"{'#' * 70}")
        print(f"Total reward: {total_reward:.4f}")
        print(f"Average reward per step: {total_reward / len(best_actions_list):.4f}")
        print(f"{'#' * 70}\n")

        # Save trajectory
        best_actions_tensor = torch.stack(best_actions_list)
        torch.save(best_actions_tensor, 'trajectory_true_dms.pt')
        print("✓ Trajectory saved to trajectory_true_dms.pt")

    def replay_trajectory(self) -> None:
        """Replay saved trajectory."""
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)

        rewards = []
        actions_to_replay = torch.load('trajectory_true_dms.pt')

        print("\nReplaying TRUE DMS trajectory...")
        timestep = 0
        for action in actions_to_replay:
            actions = action.unsqueeze(0).repeat(self.num_actors, 1)
            obs, reward, done, _ = self.env.step(actions)
            self.obs = self._convert_obs(obs)
            self.dones = done

            reward_val = reward[0].item() if isinstance(reward, torch.Tensor) else reward[0]
            print(f"Timestep {timestep + 1} | Reward: {reward_val:.3f}")
            timestep += 1
            rewards.append(reward_val)

        print("✓ Trajectory replay complete")

        # Plot rewards
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, marker='o', linewidth=2, markersize=5)
        plt.xlabel("Timestep")
        plt.ylabel("Reward")
        plt.title("TRUE DMS MPC Performance")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("reward_plot_true_dms.png", dpi=300)
        print("✓ Reward plot saved to reward_plot_true_dms.png")

        # Animated plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, len(rewards))
        ax.set_ylim(min(rewards) - 0.1, max(rewards) + 0.1)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Reward")
        ax.set_title("TRUE DMS MPC Performance")
        ax.grid(True, alpha=0.3)

        (line,) = ax.plot([], [], lw=2, marker='o', label="Step Reward")
        ax.legend()

        xdata, ydata = [], []

        def init() -> Tuple[Any]:
            line.set_data([], [])
            return (line,)

        def update(frame: int) -> Tuple[Any]:
            xdata.append(frame)
            ydata.append(rewards[frame])
            line.set_data(xdata, ydata)
            return (line,)

        ani = animation.FuncAnimation(fig, update, frames=len(rewards), init_func=init, blit=True, interval=100, repeat=False)

        ani.save("reward_animation_true_dms.gif", writer="pillow", fps=30)
        print("✓ Animation saved to reward_animation_true_dms.gif")
