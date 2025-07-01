import numpy as np
import torch
import warp as wp
from scipy.optimize import minimize
from mineral.agents.agent import Agent

# Initialize Warp
wp.init()

class DMS1WithWarp(Agent):
    r"""DMS1 Agent with Rewarped/Warp Differentiable Simulation."""

    def __init__(self, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.dms1_config = full_cfg.agent.dms1
        self.num_actors = self.dms1_config.num_actors
        self.max_agent_steps = int(self.dms1_config.max_agent_steps)

        self.N = self.dms1_config.get('N', 40)
        self.DT = self.dms1_config.get('DT', 0.05)

        self.CONTROL_DIM = 1
        self.STATE_DIM = 2
        self.DIM = self.CONTROL_DIM + self.STATE_DIM
        
        self.COST_THETA = self.dms1_config.get('COST_THETA', 1.0)
        self.COST_THETA_DOT = self.dms1_config.get('COST_THETA_DOT', 0.1)
        self.COST_CONTROL = self.dms1_config.get('COST_CONTROL', 0.001)
        self.COST_TERMINAL = self.dms1_config.get('COST_TERMINAL', 3.0)
        
        # Pendulum physical parameters
        self.GRAVITY = self.dms1_config.get('GRAVITY', 10.0)
        self.LENGTH = self.dms1_config.get('LENGTH', 1.0)
        self.MASS = self.dms1_config.get('MASS', 1.0)

        self.MAX_ITER = self.dms1_config.get('MAX_ITER', 100)

        # Call parent constructor
        super().__init__(full_cfg, **kwargs)

        # Initialize optimization variables
        self.x_u = None
        self.bounds = None
        
        # Setup Warp arrays for differentiable simulation
        self._setup_warp_simulation()
        
    def _setup_warp_simulation(self):
        """Initialize Warp arrays and kernels for differentiable simulation."""
        # Create Warp arrays with gradients enabled
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # State trajectory [theta, theta_dot] for N+1 time steps
        self.warp_states = wp.zeros((self.N + 1, self.STATE_DIM), dtype=wp.float32, 
                                   device=device, requires_grad=True)
        
        # Control trajectory [torque] for N time steps  
        self.warp_controls = wp.zeros((self.N, self.CONTROL_DIM), dtype=wp.float32,
                                     device=device, requires_grad=True)
        
        # Cost output (scalar)
        self.warp_cost = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        
        # Physical parameters as Warp arrays
        self.warp_gravity = wp.array([self.GRAVITY], dtype=wp.float32, device=device)
        self.warp_length = wp.array([self.LENGTH], dtype=wp.float32, device=device)
        self.warp_mass = wp.array([self.MASS], dtype=wp.float32, device=device)
        self.warp_dt = wp.array([self.DT], dtype=wp.float32, device=device)

    @wp.kernel
    def pendulum_dynamics_kernel(
        states: wp.array(dtype=wp.vec2),
        controls: wp.array(dtype=float),
        next_states: wp.array(dtype=wp.vec2),
        gravity: wp.array(dtype=float),
        length: wp.array(dtype=float),
        mass: wp.array(dtype=float),
        dt: wp.array(dtype=float),
        n_steps: int
    ):
        """Warp kernel to compute pendulum dynamics using RK4 integration."""
        tid = wp.tid()
        
        if tid < n_steps:
            # Current state [theta, theta_dot]
            state = states[tid]
            theta = state[0]
            theta_dot = state[1]
            
            # Control input [torque]
            u = controls[tid]
            
            # Physical parameters
            g = gravity[0]
            l = length[0]
            m = mass[0]
            dt_val = dt[0]
            
            # RK4 integration for pendulum dynamics
            # dx/dt = [theta_dot, (3*g/(2*l))*sin(theta) + (3/(m*l^2))*u]
            
            # k1
            k1_theta = theta_dot
            k1_theta_dot = (3.0 * g / (2.0 * l)) * wp.sin(theta) + (3.0 / (m * l * l)) * u
            
            # k2 
            theta_k2 = theta + 0.5 * dt_val * k1_theta
            theta_dot_k2 = theta_dot + 0.5 * dt_val * k1_theta_dot
            k2_theta = theta_dot_k2
            k2_theta_dot = (3.0 * g / (2.0 * l)) * wp.sin(theta_k2) + (3.0 / (m * l * l)) * u
            
            # k3
            theta_k3 = theta + 0.5 * dt_val * k2_theta
            theta_dot_k3 = theta_dot + 0.5 * dt_val * k2_theta_dot
            k3_theta = theta_dot_k3
            k3_theta_dot = (3.0 * g / (2.0 * l)) * wp.sin(theta_k3) + (3.0 / (m * l * l)) * u
            
            # k4
            theta_k4 = theta + dt_val * k3_theta
            theta_dot_k4 = theta_dot + dt_val * k3_theta_dot
            k4_theta = theta_dot_k4
            k4_theta_dot = (3.0 * g / (2.0 * l)) * wp.sin(theta_k4) + (3.0 / (m * l * l)) * u
            
            # Final RK4 step
            next_theta = theta + (dt_val / 6.0) * (k1_theta + 2.0 * k2_theta + 2.0 * k3_theta + k4_theta)
            next_theta_dot = theta_dot + (dt_val / 6.0) * (k1_theta_dot + 2.0 * k2_theta_dot + 2.0 * k3_theta_dot + k4_theta_dot)
            
            # Store next state
            next_states[tid + 1] = wp.vec2(next_theta, next_theta_dot)

    @wp.kernel 
    def cost_kernel(
        states: wp.array(dtype=wp.vec2),
        controls: wp.array(dtype=float),
        cost_output: wp.array(dtype=float),
        cost_theta: float,
        cost_theta_dot: float,
        cost_control: float,
        cost_terminal: float,
        n_steps: int
    ):
        """Warp kernel to compute the MPC cost function."""
        # Single thread computes total cost
        if wp.tid() == 0:
            total_cost = 0.0
            
            # Stage costs: sum over time steps
            for i in range(n_steps):
                state = states[i]
                control = controls[i]
                
                # Normalize theta to [-pi, pi]
                theta = ((state[0] + wp.pi) % (2.0 * wp.pi)) - wp.pi
                theta_dot = state[1]
                
                # Quadratic cost: x^T Q x + u^T R u
                stage_cost = (cost_theta * theta * theta + 
                             cost_theta_dot * theta_dot * theta_dot +
                             cost_control * control * control)
                total_cost += stage_cost
            
            # Terminal cost
            final_state = states[n_steps]
            final_theta = ((final_state[0] + wp.pi) % (2.0 * wp.pi)) - wp.pi
            final_theta_dot = final_state[1]
            
            terminal_cost = (cost_terminal * final_theta * final_theta +
                           cost_terminal * cost_theta_dot * final_theta_dot * final_theta_dot)
            total_cost += terminal_cost
            
            cost_output[0] = total_cost

    def simulate_with_gradients(self, x_u_flat, current_state):
        """Run differentiable simulation and return cost + gradients."""
        
        # Convert flat optimization vector to Warp arrays
        self._convert_flat_to_warp(x_u_flat, current_state)
        
        # Create tape for automatic differentiation
        tape = wp.Tape()
        
        with tape:
            # Forward simulation using Warp kernels
            wp.launch(
                kernel=self.pendulum_dynamics_kernel,
                dim=self.N,
                inputs=[
                    self.warp_states, self.warp_controls, self.warp_states,
                    self.warp_gravity, self.warp_length, self.warp_mass, self.warp_dt,
                    self.N
                ]
            )
            
            # Compute cost
            wp.launch(
                kernel=self.cost_kernel,
                dim=1,
                inputs=[
                    self.warp_states, self.warp_controls, self.warp_cost,
                    self.COST_THETA, self.COST_THETA_DOT, self.COST_CONTROL, 
                    self.COST_TERMINAL, self.N
                ]
            )
        
        # Backward pass to compute gradients
        tape.backward(loss=self.warp_cost)
        
        # Extract cost value and gradients
        cost_value = self.warp_cost.numpy()[0]
        
        # Get gradients w.r.t. controls and states
        states_grad = tape.gradients[self.warp_states].numpy()
        controls_grad = tape.gradients[self.warp_controls].numpy()
        
        # Convert gradients back to flat format matching x_u_flat
        grad_flat = self._convert_warp_to_flat_grad(states_grad, controls_grad)
        
        # Reset tape for next iteration
        tape.reset()
        
        return cost_value, grad_flat
    
    def _convert_flat_to_warp(self, x_u_flat, current_state):
        """Convert flat optimization vector to Warp arrays."""
        # Set initial state
        self.warp_states.assign(0, current_state)
        
        # Unpack states and controls from flat vector
        for i in range(self.N):
            idx = i * self.DIM
            # State at time i+1 (skip initial state)
            state = x_u_flat[idx:idx + self.STATE_DIM] 
            control = x_u_flat[idx + self.STATE_DIM:idx + self.DIM]
            
            if i < self.N - 1:  # Not the last control
                self.warp_states.assign(i + 1, state)
            self.warp_controls.assign(i, control)
        
        # Final state
        final_state = x_u_flat[self.N * self.DIM:self.N * self.DIM + self.STATE_DIM]
        self.warp_states.assign(self.N, final_state)
    
    def _convert_warp_to_flat_grad(self, states_grad, controls_grad):
        """Convert Warp gradients back to flat format."""
        grad_flat = np.zeros_like(self.x_u)
        
        for i in range(self.N):
            idx = i * self.DIM
            # Gradient w.r.t. state at time i+1
            grad_flat[idx:idx + self.STATE_DIM] = states_grad[i + 1]
            # Gradient w.r.t. control at time i  
            grad_flat[idx + self.STATE_DIM:idx + self.DIM] = controls_grad[i]
        
        # Final state gradient
        grad_flat[self.N * self.DIM:self.N * self.DIM + self.STATE_DIM] = states_grad[self.N]
        
        return grad_flat

    def cost_with_warp_gradients(self, x_u):
        """Cost function that uses Warp for automatic differentiation."""
        # Get current state from first elements of x_u
        current_x = x_u[0:self.STATE_DIM]
        
        # Run differentiable simulation
        cost_value, grad_flat = self.simulate_with_gradients(x_u, current_x)
        
        return cost_value, grad_flat

    def eq_constraint_with_warp(self, x_u, current_x):
        """Equality constraints using Warp simulation results."""
        # Convert to Warp format and simulate
        self._convert_flat_to_warp(x_u, current_x)
        
        # Run forward simulation without tape (no gradients needed for constraints)
        wp.launch(
            kernel=self.pendulum_dynamics_kernel,
            dim=self.N,
            inputs=[
                self.warp_states, self.warp_controls, self.warp_states,
                self.warp_gravity, self.warp_length, self.warp_mass, self.warp_dt,
                self.N
            ]
        )
        
        # Extract results and compute constraint violations
        states_np = self.warp_states.numpy()
        constraints = []
        
        # Initial constraint: x0 must equal current state
        constraints.extend(states_np[0] - current_x)
        
        # Note: Dynamics constraints are automatically satisfied by the simulation
        # so we don't need explicit continuity constraints here
        
        return np.array(constraints)

    def get_actions(self, obs, sample: bool = True):
        """Generate actions using MPC optimization with Warp gradients."""
        # Convert observation to current state
        if isinstance(obs, dict):
            obs = obs['obs']
        
        current_theta = np.arctan2(obs[1], obs[0])
        current_x = np.array([current_theta, obs[2]], dtype=np.float32)
        
        # Initialize optimization variables if needed
        if self.x_u is None:
            self.x_u = np.zeros((self.STATE_DIM + self.CONTROL_DIM) * self.N + self.STATE_DIM, dtype=np.float32)
            
            # Warm start with current state
            for i in range(self.N):
                self.x_u[i*(self.STATE_DIM + self.CONTROL_DIM):
                    i*(self.STATE_DIM + self.CONTROL_DIM) + self.STATE_DIM] = current_x
            self.x_u[self.N*self.DIM:self.N*self.DIM + self.STATE_DIM] = current_x
            
            # Set bounds: states unbounded, controls bounded
            self.bounds = [(None, None), (None, None), (-2, 2)] * self.N + [(None, None), (None, None)]
        
        # Update initial state
        self.x_u[0:self.STATE_DIM] = current_x
        
        # Constraint wrapper
        def eq_constraint_wrapper(x_u):
            return self.eq_constraint_with_warp(x_u, current_x)
        
        eq_cons = {'type': 'eq', 'fun': eq_constraint_wrapper}
        
        # Optimize using gradients from Warp
        result = minimize(
            self.cost_with_warp_gradients,  # Uses automatic gradients
            self.x_u,
            method='SLSQP',
            jac=True,  # Gradients provided by cost function
            bounds=self.bounds,
            constraints=[eq_cons],
            options={'disp': False, 'maxiter': self.MAX_ITER}
        )
        
        # Update solution
        self.x_u = result.x
        
        # Extract first control action
        action = self.x_u[self.STATE_DIM:self.STATE_DIM + self.CONTROL_DIM]
        
        # Warm start for next iteration
        for i in range(self.N-1):
            self.x_u[i*self.DIM:(i+1)*self.DIM] = self.x_u[(i+1)*self.DIM:(i+2)*self.DIM]
        self.x_u[(self.N-1)*self.DIM:(self.N-1)*self.DIM + self.STATE_DIM] = \
            self.x_u[(self.N)*self.DIM:(self.N)*self.DIM + self.STATE_DIM]
        
        return torch.tensor(action, device=self.device, dtype=torch.float32)

    def explore_env(self, env, timesteps: int, random: bool = False, sample: bool = False):
        """Run MPC with Warp gradients in the environment."""
        # Similar to original DMS1 but using Warp-accelerated MPC
        return super().explore_env(env, timesteps, random, sample)
    
    def train(self):
        """Training loop using Warp-accelerated MPC."""
        return super().train()

    # Implement required abstract methods
    def eval(self):
        raise NotImplementedError("DMS1WithWarp doesnt have eval()")

    def set_train(self):
        pass

    def set_eval(self):
        raise NotImplementedError("DMS1WithWarp doesnt have set_eval()")

    def save(self, f):
        raise NotImplementedError("DMS1WithWarp doesnt have save()")

    def load(self, f, ckpt_keys=''):
        raise NotImplementedError("DMS1WithWarp doesnt have load()") 