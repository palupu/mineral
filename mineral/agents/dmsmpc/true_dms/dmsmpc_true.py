"""TRUE Direct Multiple Shooting MPC with Independent Shooting Nodes.

This is a proper DMS implementation where each shooting interval is truly independent,
unlike the pseudo-DMS in dmsmpc.py which simulates sequentially.

Key Differences from Pseudo-DMS:
1. Each shooting node x[i] is set independently before simulating to x[i+1]
2. Constraints enforce: x[i+1] = f(x[i], u[i]) for EACH node independently
3. No sequential dependency between nodes during constraint evaluation

The Challenge:
- Setting arbitrary states in complex physics (MPM particles) is non-trivial
- We provide multiple strategies for state setting (joint-only, joint+COM translation)

Performance Implications:
- More accurate for unstable/nonlinear dynamics
- Computationally expensive: N independent simulations per constraint evaluation
- Each SLSQP iteration requires multiple constraint evaluations
"""

import time
from typing import Any, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import warp as wp
from scipy.optimize import minimize

from mineral.agents.agent import Agent


class TrueDMSMPCAgent(Agent):
    r"""TRUE Direct Multiple Shooting MPC with Independent Shooting Intervals.

    This implementation properly decouples shooting nodes, making each interval
    independent. The optimizer proposes states x[i] at each node, and we verify
    that simulating from x[i] with u[i] reaches the proposed x[i+1].

    Full State Representation (Scientific Research):
    - State x contains ALL MPM particle positions (no dimensionality reduction)
    - For N particles: state_dim = N × 3 (x, y, z coordinates)
    - Control u: 3D control input [dx, dy, ry] matching environment action space
    - This provides complete physics information with no approximations
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

        # State and control dimensions - determined dynamically after env init
        self.control_dim = 3  # Fixed: dx, dy, ry for RollingPin
        self.state_dim = None  # Will be set after first observation
        self.num_particles = None  # Number of MPM particles
        self.dim = None  # Will be state_dim + control_dim

        super().__init__(full_cfg, **kwargs)

        self.obs = None
        self.dones = None

        # Performance tracking
        self.constraint_eval_count = 0
        self.cost_eval_count = 0

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

        return s

    def _obs_to_state(self, return_torch: bool = False):
        """Extract full state vector from simulation (ALL MPM particle positions).

        For scientific research, we use the COMPLETE state: ALL particle positions
        directly from the simulation state.

        RollingPin: 2592 particles in simulation, 250 in observations
        We use ALL 2592 for TRUE DMS to be mathematically correct.

        State vector: [x1, y1, z1, x2, y2, z2, ..., xN, yN, zN] for ALL N particles

        Args:
            return_torch: If True, return torch.Tensor (for differentiable ops).
                         If False, return numpy array (for scipy optimization).

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

        # Extract all particle positions from simulation state
        # Keep gradients if return_torch=True
        mpm_x = wp.to_torch(full_state.mpm_x)

        # Move to correct device if needed
        if mpm_x.device != self.device:
            mpm_x = mpm_x.to(self.device)

        # Handle batch/environment dimension if present
        if mpm_x.ndim == 2:
            # Shape: (num_particles, 3) - already correct
            pass
        elif mpm_x.ndim == 3:
            # Shape: (num_envs, num_particles, 3) - extract first env
            mpm_x = mpm_x[0]
        else:
            raise ValueError(f"Expected mpm_x with 2 or 3 dims, got {mpm_x.ndim}")

        # Flatten to 1D: (num_particles, 3) -> (num_particles * 3,)
        state = mpm_x.flatten()

        # Initialize dimensions if this is the first call
        if self.state_dim is None:
            self.num_particles = mpm_x.shape[0]
            self.state_dim = self.num_particles * 3
            self.dim = self.state_dim + self.control_dim
            print(f"\n{'=' * 70}")
            print("State Dimensions Initialized (FULL PARTICLES - Research Mode):")
            print(f"  Number of particles: {self.num_particles}")
            print(f"  State dimension: {self.state_dim} (particles × 3)")
            print(f"  Control dimension: {self.control_dim}")
            print(f"  Total decision vars per node: {self.dim}")
            print(f"{'=' * 70}\n")

        # Return torch tensor or numpy array based on flag
        if return_torch:
            return state
        else:
            return state.detach().cpu().numpy()

    def _set_state_from_vector(self, state_vec, template_state: Any) -> Any:
        """Set full environment state from state vector (all particle positions).

        This is THE KEY FUNCTION for true DMS. It allows the optimizer to
        propose arbitrary states x[i] at each shooting node, which we then
        set in the simulator before shooting to x[i+1].

        For scientific research: We use the FULL state - all MPM particle positions.
        No approximations, no reduced representations.

        Args:
            state_vec: Full state vector, shape (state_dim,) = (num_particles * 3,)
                      Can be numpy array or torch.Tensor
                      Contains all particle positions: [x1, y1, z1, x2, y2, z2, ...]
            template_state: Full simulation state to use as template
                           (contains all MPM particles, grid, velocities, etc.)

        Returns:
            new_state: Full simulation state with updated particle positions
        """
        # Verify template state has mpm_x
        if not hasattr(template_state, 'mpm_x') or template_state.mpm_x is None:
            raise ValueError("Template state does not have mpm_x attribute")

        # Get template mpm_x shape for validation
        num_particles = template_state.mpm_x.shape[0]
        expected_size = num_particles * 3  # num_particles * 3

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
                f"but template_state.mpm_x requires {expected_size} "
                f"({num_particles} particles × 3)"
            )

        # Clone full state (including all MPM particles, velocities, stresses, etc.)
        new_state = self.clone_state(template_state)

        # Convert to torch tensor if needed
        if isinstance(state_vec, np.ndarray):
            particle_positions = torch.from_numpy(
                state_vec.reshape(self.num_particles, 3).astype(np.float32)
            ).to(self.device)
        else:  # Already torch.Tensor
            particle_positions = state_vec.reshape(self.num_particles, 3)
            if particle_positions.dtype != torch.float32:
                particle_positions = particle_positions.float()
            if particle_positions.device != self.device:
                particle_positions = particle_positions.to(self.device)

        # Set all particle positions directly - maintains gradients for torch tensors
        new_state.mpm_x.assign(wp.from_torch(particle_positions))

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

    def cost(self, x_u : np.ndarray) -> tuple[float, np.ndarray]:
        """Cost function for the optimization problem.

        Matches RollingPin reward structure (cost = -reward):
        - Minimize average particle height (flatten the dough)
        - Minimize height variance (uniform flatness)
        - Minimize control effort

        State x contains all particle positions: [x1,y1,z1, x2,y2,z2, ..., xN,yN,zN]
        We extract y-coordinates (height) and compute mean + variance.
        """
        self.cost_eval_count += 1
        # Convert numpy array to torch tensor with gradient tracking
        device = self.device
        x_u_torch = torch.tensor(x_u, dtype=torch.float64, requires_grad=True, device=device)

        # Reshape into state and control sequences
        last_x = x_u_torch[-self.state_dim :]
        x_u_torch_short = x_u_torch[: -self.state_dim].reshape(self.N, self.dim)

        x = torch.cat([x_u_torch_short[:, : self.state_dim], last_x.unsqueeze(0)], dim=0)
        u = x_u_torch_short[:, self.state_dim :]

        # Reference height from RollingPin (self.h = 0.125)
        h_ref = 0.125

        # Compute stage costs matching RollingPin reward
        cost_value = 0.0

        # Vecotirzien 
        # dummy set 
        for i in range(self.N):
            # Extract particle positions: (num_particles * 3,) -> (num_particles, 3)
            particles = x[i].reshape(self.num_particles, 3)

            # Extract heights (y-coordinates, index 1)
            heights = particles[:, 1]

            # Cost 1: Average height (penalize high dough)
            # RollingPin reward: 1.0 / (1.0 + mean_height/h_ref)^2
            # Cost (minimize): mean_height / h_ref
            mean_height = heights.mean()
            height_cost = (mean_height / h_ref) ** 2  # Squared to match reward scaling

            # Cost 2: Height variance (penalize non-uniform height)
            # RollingPin reward: -variance
            # Cost (minimize): variance
            variance_cost = heights.var()

            # Cost 3: Control effort
            control_cost = self.cost_control * (u[i] @ u[i])

            # Total stage cost
            stage_cost = self.cost_state * (height_cost + variance_cost) + control_cost
            cost_value += stage_cost

        # Terminal cost: emphasize final state (flat + uniform)
        particles_final = x[self.N].reshape(self.num_particles, 3)
        heights_final = particles_final[:, 1]
        mean_height_final = heights_final.mean()
        variance_final = heights_final.var()

        terminal_cost = self.cost_terminal * ((mean_height_final / h_ref) ** 2 + variance_final)
        cost_value += terminal_cost

        # Compute gradients
        cost_value.backward()

        print(f"Cost call #{self.cost_eval_count}: cost={cost_value.item():.6f}", flush=True)
        # if self.cost_eval_count == 2:
        #     raise RuntimeError("Testwise error thrown as requested")

        return cost_value.item(), x_u_torch.grad.cpu().numpy()

    def eq_constraint_differentiable(self, x_u: torch.Tensor, current_x: torch.Tensor, template_state: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """TRUE Direct Multiple Shooting Equality Constraints with Jacobian computation.

        This version uses Rewarped's differentiable simulation to compute analytical
        gradients (Jacobian) of the constraints with respect to decision variables.

        Args:
            x_u: Decision variables as torch.Tensor with requires_grad=True
            current_x: Current observed state as torch.Tensor
            template_state: Template simulation state (with all MPM data)

        Returns:
            constraints: Constraint violations as torch.Tensor
            jacobian: Jacobian matrix d(constraints)/d(x_u)
        """
        # Extract states and controls from the trajectory
        x = x_u[:-self.state_dim].reshape(self.N, self.dim)[:, :self.state_dim]
        x = torch.cat([x, x_u[-self.state_dim:].unsqueeze(0)], dim=0)
        u = x_u[:-self.state_dim].reshape(self.N, self.dim)[:, self.state_dim:]

        # Pre-allocate constraints tensor
        constraints = torch.zeros((self.N + 1) * self.state_dim,
                                 dtype=torch.float32, device=self.device)

        # Initial constraint: first state must match current state
        constraints[:self.state_dim] = x[0] - current_x

        # TRUE MULTIPLE SHOOTING: Each interval is independent!
        for i in range(self.N):
            # 1. Set environment to the optimizer's proposed state x[i]
            state_at_node_i = self._set_state_from_vector(x[i], template_state)

            # 2. Simulate one step with control u[i] - DIFFERENTIABLE!
            next_state_sim = self.simulate_single_step(state_at_node_i, u[i], return_torch=True)

            # 3. Constraint: simulated state must match optimizer's x[i+1]
            start_idx = (i + 1) * self.state_dim
            end_idx = (i + 2) * self.state_dim
            constraints[start_idx:end_idx] = x[i + 1] - next_state_sim

        # Compute Jacobian using autograd - this is where Rewarped shines!
        # We need the full Jacobian matrix: (num_constraints, num_variables)
        # Note: For high-dimensional problems (23k+ constraints), this is computationally expensive
        # but provides exact analytical gradients

        # Use torch.autograd.functional.jacobian for efficient computation
        # This computes the full Jacobian matrix in one call
        def constraint_fn(xu):
            # Extract states and controls from the trajectory
            x_local = xu[:-self.state_dim].reshape(self.N, self.dim)[:, :self.state_dim]
            x_local = torch.cat([x_local, xu[-self.state_dim:].unsqueeze(0)], dim=0)
            u_local = xu[:-self.state_dim].reshape(self.N, self.dim)[:, self.state_dim:]

            # Pre-allocate constraints tensor
            constraints_local = torch.zeros((self.N + 1) * self.state_dim,
                                           dtype=torch.float32, device=self.device)

            # Initial constraint
            constraints_local[:self.state_dim] = x_local[0] - current_x

            # TRUE MULTIPLE SHOOTING: Each interval is independent!
            for i in range(self.N):
                state_at_node_i = self._set_state_from_vector(x_local[i], template_state)
                next_state_sim = self.simulate_single_step(state_at_node_i, u_local[i], return_torch=True)
                start_idx = (i + 1) * self.state_dim
                end_idx = (i + 2) * self.state_dim
                constraints_local[start_idx:end_idx] = x_local[i + 1] - next_state_sim

            return constraints_local

        # Compute full Jacobian matrix efficiently
        print(f"Computing Jacobian ({(self.N + 1) * self.state_dim}x{len(x_u)})...", end='', flush=True)
        jacobian = torch.autograd.functional.jacobian(constraint_fn, x_u, create_graph=False, strict=True)
        print(" done!", flush=True)

        return constraints, jacobian

    def eq_constraint(self, x_u: np.ndarray, current_x: np.ndarray, template_state: Any) -> Tuple[np.ndarray, np.ndarray]:
        """TRUE Direct Multiple Shooting Equality Constraints with analytical Jacobian.

        This wrapper uses Rewarped's differentiable simulation to compute both
        constraints and their analytical Jacobian for SLSQP optimization.

        This is the KEY difference from pseudo-DMS. Each shooting interval is
        evaluated INDEPENDENTLY:

        For each i = 0, ..., N-1:
            1. Set environment to state x[i] (using _set_state_from_vector)
            2. Simulate one step with control u[i] (differentiable!)
            3. Constraint: simulated_state must equal x[i+1]

        Critically: We do NOT carry forward the simulated state to the next iteration.
        Each interval shoots from the optimizer's proposed x[i], not from the
        simulation result.

        Args:
            x_u: Decision variables [x0, u0, x1, u1, ..., x_{N-1}, u_{N-1}, xN]
            current_x: Current observed state (for initial constraint)
            template_state: Template simulation state (with all MPM data)

        Returns:
            constraints: Array of constraint violations, shape ((N+1) * state_dim,)
            jacobian: Jacobian matrix d(constraints)/d(x_u), shape ((N+1)*state_dim, len(x_u))
        """
        self.constraint_eval_count += 1
        print(f"Constraint #{self.constraint_eval_count} starting...")

        # Convert to torch tensors for differentiable computation
        x_u_torch = torch.tensor(x_u, dtype=torch.float32, requires_grad=True, device=self.device)
        current_x_torch = torch.tensor(current_x, dtype=torch.float32, device=self.device)

        # Compute constraints and Jacobian using differentiable simulation
        constraints, jacobian = self.eq_constraint_differentiable(x_u_torch, current_x_torch, template_state)

        # Convert back to numpy for scipy
        constraints_np = constraints.detach().cpu().numpy().astype(np.float64)
        jacobian_np = jacobian.detach().cpu().numpy().astype(np.float64)

        max_violation = np.max(np.abs(constraints_np))
        rms_violation = np.sqrt(np.mean(constraints_np**2))

        print(
            f"Constraint #{self.constraint_eval_count} done: "
            f"max={max_violation:.4e} rms={rms_violation:.4e} "
            f"Jacobian shape={jacobian_np.shape}",
            flush=True,
        )

        return constraints_np, jacobian_np

    def dms_plan(self, current_x: np.ndarray, init_state: Any) -> torch.Tensor:
        """Direct Multiple Shooting MPC planning with TRUE independent shooting.

        Args:
            current_x: Current observed state vector
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

        # Initialize state and control vector
        x_u = np.zeros((self.state_dim + self.control_dim) * self.N + self.state_dim) # dtype=np.float64

        # Initialize state trajectory with current state (warm start)
        for i in range(self.N):
            x_u[i * self.dim : i * self.dim + self.state_dim] = current_x
        x_u[self.N * self.dim : self.N * self.dim + self.state_dim] = current_x

        # Define bounds
        control_bounds = [(-1.0, 1.0)] * self.control_dim  # Control bounds
        state_bounds = [(None, None)] * self.state_dim  # No bounds on states

        # Repeat bounds for each shooting node
        bounds = (state_bounds + control_bounds) * self.N + state_bounds

        # Save the original state
        original_state = self.clone_state(init_state)

        # Reset constraint evaluation counter
        self.constraint_eval_count = 0
        self.cost_eval_count = 0
        self.iteration_count = 0

        # Create constraint wrapper that returns both constraints and Jacobian
        def eq_constraint_wrapper(x_u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return self.eq_constraint(x_u, current_x, init_state)

        # Define constraints with analytical Jacobian from Rewarped
        eq_cons = {
            'type': 'eq',
            'fun': lambda x: eq_constraint_wrapper(x)[0],  # Constraint values
            'jac': lambda x: eq_constraint_wrapper(x)[1]   # Analytical Jacobian
        }

        # Callback function to monitor SLSQP iterations
        def callback(x_u: np.ndarray) -> None:
            self.iteration_count += 1
            # Compute current constraint violation (only need values for monitoring)
            c, _ = eq_constraint_wrapper(x_u)
            max_c = np.max(np.abs(c))
            rms_c = np.sqrt(np.mean(c**2))
            # Compute current cost
            cost_val, _ = self.cost(x_u)
            print(
                f"\n>>> ITERATION {self.iteration_count}: cost={cost_val:.6f} | constraint max={max_c:.4e} rms={rms_c:.4e}",
                flush=True,
            )

        print(f"\n{'=' * 70}")
        print("Starting TRUE DMS Optimization (with Rewarped Analytical Jacobians)")
        print(f"  Particles: {self.num_particles}")
        print(f"  Nodes: {self.N}, State dim: {self.state_dim}, Control dim: {self.control_dim}")
        print(f"  Decision variables: {len(x_u)}")
        print(f"  Constraints: {(self.N + 1) * self.state_dim} equality constraints")
        print("  Using differentiable simulation for analytical gradients ✓")
        print(f"{'=' * 70}\n")

        # Optimize
        start_time = time.time()
        result = minimize(
            self.cost,  # Cost function (returns cost and gradient)
            x_u,
            method='SLSQP',
            jac=True,
            bounds=bounds,
            constraints=[eq_cons],
            # callback=callback,  # Monitor each iteration
            options={
                'disp': True,
                'maxiter': self.max_iter,
                # 'ftol': 1e-9,  # Tighter function tolerance (cost change)
                # 'eps': 1e-8,   # Finite difference step for gradient
            },
        )
        elapsed_time = time.time() - start_time

        # Manually check constraint satisfaction
        # SLSQP may report failure even if constraints are "good enough" for high-D problems
        final_constraints, _ = eq_constraint_wrapper(result.x)
        max_constraint_viol = np.max(np.abs(final_constraints))
        mean_constraint_viol = np.mean(np.abs(final_constraints))

        # Restore the original state after optimization
        self.env.state_0 = original_state

        # Extract the first control input
        action = result.x[self.state_dim : self.state_dim + self.control_dim]

        print(f"\n{'=' * 70}")
        print("Optimization Complete")
        print(f"  SLSQP Success: {result.success} | Message: {result.message}")
        print(f"  Final cost: {result.fun:.6f}")
        print(f"  Iterations: {result.nit}")
        print(f"  Function evals: {result.nfev}")
        print(f"  Constraint evals: {self.constraint_eval_count}")
        print(f"  Time: {elapsed_time:.2f}s")
        print("\n  Constraint Satisfaction:")
        print(f"    Max violation: {max_constraint_viol:.4e}")
        print(f"    Mean violation: {mean_constraint_viol:.4e}")

        # Accept solution if constraints are "good enough" even if SLSQP says failure
        constraint_tolerance = 0.01  # Practical tolerance for 7776D problem
        constraints_satisfied = max_constraint_viol < constraint_tolerance

        if constraints_satisfied:
            print(f"    ✓ Constraints satisfied (< {constraint_tolerance})")
        else:
            print(f"    ✗ Constraints NOT satisfied (threshold: {constraint_tolerance})")
            print("    ⚠️  Using best solution anyway (TRUE DMS may not converge in 7776D)")

        print(f"\n  Optimal action: {action}")
        print(f"{'=' * 70}\n")

        return torch.tensor(action, device=self.device, dtype=torch.float32)

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
        print(f"{'TRUE DIRECT MULTIPLE SHOOTING MPC':^70}")
        print(f"{'#' * 70}")
        print(f"Full State Representation: All {self.num_particles} MPM particles")
        print(f"Horizon: N={self.N} nodes, H={self.H:.2f}s")
        print(f"State dim: {self.state_dim}, Control dim: {self.control_dim}")
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

            # Plan using TRUE DMS MPC
            best_action = self.dms_plan(current_x, init_state)
            actions = best_action.unsqueeze(0).repeat(self.num_actors, 1)

            # Restore state before executing the action
            self.env.state_0 = init_state

            # Step the environment forward with the best action
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
