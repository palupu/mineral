import numpy as np
import torch
from scipy.optimize import minimize
from mineral.agents.agent import Agent


class DMS1(Agent):
    r"""Template Agent."""

    def __init__(self, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.dms1_config = full_cfg.agent.dms1
        self.num_actors = self.dms1_config.num_actors
        self.max_agent_steps = int(self.dms1_config.max_agent_steps)

        self.N = self.dms1_config.get('N', 40)
        # self.DT = self.dms1_config.get('DT', 0.05)

        self.CONTROL_DIM = 1
        self.STATE_DIM = 2
        self.DIM = self.CONTROL_DIM + self.STATE_DIM
        
        self.COST_THETA = self.dms1_config.get('COST_THETA', 1.0)
        self.COST_THETA_DOT = self.dms1_config.get('COST_THETA_DOT', 0.1)
        self.COST_CONTROL = self.dms1_config.get('COST_CONTROL', 0.001)
        self.COST_TERMINAL = self.dms1_config.get('COST_TERMINAL', 3.0)
        
        # self.GRAVITY = self.dms1_config.get('GRAVITY', 10.0)
        # self.LENGTH = self.dms1_config.get('LENGTH', 1.0)
        # self.MASS = self.dms1_config.get('MASS', 1.0)

        self.MAX_ITER = self.dms1_config.get('MAX_ITER', 100)
        self.MAX_EPISODE_STEPS = self.dms1_config.get('MAX_EPISODE_STEPS', 100)

        # Call parent constructor - this sets up device, metrics, logging, etc.
        super().__init__(full_cfg, **kwargs)

        # Initialize optimization variables - these will be set up in get_actions()
        self.x_u = None  # Combined state and control vector for optimization
        self.bounds = None  # Bounds for optimization variables
        
    def rk4_step(self, x_u, f, dt):
        """Perform a single step of the Runge-Kutta 4th order integration method."""
        # Extract state and control from combined vector
        x = x_u[:self.STATE_DIM]  # state variables [theta, theta_dot]
        u = x_u[self.STATE_DIM:]  # control input [torque]

        # RK4 integration steps
        k1 = f(x, u)  # derivative at current point
        k2 = f(x + 0.5 * dt * k1, u)  # derivative at midpoint using k1
        k3 = f(x + 0.5 * dt * k2, u)  # derivative at midpoint using k2
        k4 = f(x + dt * k3, u)  # derivative at endpoint using k3

        # Return next state using RK4 formula
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    
    def dynamics(self, x, u):
        """Pendulum dynamics - computes the time derivative of the state."""
        theta = x[:1]   # current angle
        dtheta = x[1:2]  # current angular velocity

        # Pendulum equation of motion: I * ddot_theta = m*g*l*sin(theta) + u
        # For a point mass pendulum: I = m*l^2, so ddot_theta = (3*g/(2*l))*sin(theta) + (3/(m*l^2))*u
        ddtheta = (3 * self.GRAVITY / (2 * self.LENGTH)) * np.sin(theta) + \
                 (3 / (self.MASS * self.LENGTH**2)) * u

        # Return state derivative [dtheta, ddtheta]
        return np.concatenate([dtheta, ddtheta])
    
    def cost(self, x_u):
        """Cost function for the optimization problem - uses PyTorch for automatic differentiation."""
        # Convert numpy array to torch tensor with gradient tracking for automatic differentiation
        device = self.device  # use the framework's device (CPU/GPU)
        x_u_torch = torch.tensor(x_u, dtype=torch.float64, requires_grad=True, device=device)
        
        # Reshape the flat optimization vector into state and control sequences
        # x_u has structure: [x0, u0, x1, u1, ..., xN-1, uN-1, xN]
        last_x = x_u_torch[-self.STATE_DIM:]  # final state xN
        x_u_torch_short = x_u_torch[:-self.STATE_DIM].reshape(self.N, self.DIM)  # reshape [x0, u0, x1, u1, ...]

        # Extract state and control sequences
        x = torch.cat([x_u_torch_short[:, :self.STATE_DIM], last_x.unsqueeze(0)], dim=0)  # states [x0, x1, ..., xN]
        u = x_u_torch_short[:, self.STATE_DIM:]  # controls [u0, u1, ..., uN-1]
        
        # Normalize theta to [-pi, pi] to handle angle wrapping
        x[:, 0] = ((x[:, 0] + np.pi) % (2 * np.pi)) - np.pi
        
        # Define cost matrices for quadratic cost function
        Q = torch.diag(torch.tensor([self.COST_THETA, self.COST_THETA_DOT], device=device, dtype=torch.float64))  # state cost matrix
        R = torch.tensor([[self.COST_CONTROL]], device=device, dtype=torch.float64)  # control cost matrix
        QN = torch.diag(torch.tensor([self.COST_TERMINAL, self.COST_TERMINAL * self.COST_THETA_DOT], device=device, dtype=torch.float64))  # terminal cost matrix

        # Calculate stage costs: sum over all time steps of x^T Q x + u^T R u
        cost_value = sum(x[i].T @ Q @ x[i] + u[i].T @ R @ u[i] for i in range(self.N))
        
        # Add terminal cost: x_N^T Q_N x_N
        cost_value += x[self.N].T @ QN @ x[self.N]
        
        # Compute gradients automatically using PyTorch
        cost_value.backward()
        
        # Return cost value and gradient for scipy.minimize with jac=True
        return cost_value.item(), x_u_torch.grad.numpy()
    
    def eq_constraint(self, x_u, current_x):
        """Equality constraints for direct multiple shooting - enforces dynamics and initial condition."""
        # Extract states and controls from the flat optimization vector
        x = x_u[:-self.STATE_DIM].reshape(self.N, self.DIM)[:, :self.STATE_DIM]  # states [x0, x1, ..., xN-1]
        x = np.vstack([x, x_u[-self.STATE_DIM:]])  # add final state xN
        u = x_u[:-self.STATE_DIM].reshape(self.N, self.DIM)[:, self.STATE_DIM:]  # controls [u0, u1, ..., uN-1]
        
        constraints = []  # list to store all constraint violations

        # Initial constraint: x0 must equal current state
        constraints.extend(x[0] - current_x)
        
        # Continuity constraints: enforce that dynamics are satisfied between shooting nodes
        X = x[:-1]  # all states except last [x0, x1, ..., xN-1]
        U = u       # all controls [u0, u1, ..., uN-1]
        X_next = x[1:]  # all states except first [x1, x2, ..., xN]
        
        # Create combined state-control matrices for each step
        XU = np.hstack([X, U])  # [x0, u0], [x1, u1], ..., [xN-1, uN-1]
        
        # Apply RK4 step to all state-control pairs at once
        X_next_pred = np.array([self.rk4_step(xu_i, self.dynamics, self.DT) 
                               for xu_i in XU])  # predicted next states using dynamics
        
        # Add all continuity constraints: x_{k+1} - f(x_k, u_k) = 0
        constraints.extend((X_next_pred - X_next).flatten())
            
        return np.array(constraints)  # return constraint violations (should be zero)

    def get_actions(self, obs, sample: bool = True):
        """Generate actions using MPC optimization - this is the main controller."""
        # Convert observation to current state
        if isinstance(obs, dict):
            obs = obs['obs']  # extract observation from dict if needed
        
        # For pendulum environment, obs is [cos(theta), sin(theta), theta_dot]
        # Convert to [theta, theta_dot] state representation
        current_theta = np.arctan2(obs[1], obs[0])  # extract angle from cos/sin
        current_x = np.array([current_theta, obs[2]])  # current state [theta, theta_dot]
        
        # Initialize optimization variables if not done yet
        if self.x_u is None:
            # Create optimization vector: [x0, u0, x1, u1, ..., xN-1, uN-1, xN]
            self.x_u = np.zeros((self.STATE_DIM + self.CONTROL_DIM) * self.N + self.STATE_DIM, dtype=np.float64)
            
            # Initialize state trajectory with current state (warm start)
            for i in range(self.N):
                self.x_u[i*(self.STATE_DIM + self.CONTROL_DIM):
                    i*(self.STATE_DIM + self.CONTROL_DIM) + self.STATE_DIM] = current_x
            self.x_u[self.N*self.DIM:self.N*self.DIM + self.STATE_DIM] = current_x  # final state
            
            # Set bounds for optimization variables
            # States: no bounds, Controls: [-2, 2] (torque limits)
            self.bounds = [(None, None), (None, None), (-2, 2)] * self.N + [(None, None), (None, None)]
        
        # Update initial state constraint to current state
        self.x_u[0:self.STATE_DIM] = current_x
        
        # Create constraint wrapper for current state (scipy.minimize requires a function)
        def eq_constraint_wrapper(x_u):
            return self.eq_constraint(x_u, current_x)
        
        # Define constraints for scipy.minimize
        eq_cons = {'type': 'eq', 'fun': eq_constraint_wrapper}  # equality constraints
        
        # Run optimization using SLSQP (Sequential Least Squares Programming)
        result = minimize(
            self.cost,           # objective function
            self.x_u,            # initial guess
            method='SLSQP',      # optimization method
            jac=True,            # use automatic gradients from cost function
            bounds=self.bounds,  # variable bounds
            constraints=[eq_cons],  # equality constraints
            options={'disp': False, 'maxiter': self.MAX_ITER}  # optimization options
        )
        
        # Update optimization vector with the optimized result
        self.x_u = result.x
        
        # Extract the first control input (MPC principle: apply first control, then replan)
        action = self.x_u[self.STATE_DIM:self.STATE_DIM + self.CONTROL_DIM]
        
        # Shift the solution for the next iteration (warm start)
        # This improves convergence by using the previous solution as initial guess
        for i in range(self.N-1):
            self.x_u[i*self.DIM:(i+1)*self.DIM] = self.x_u[(i+1)*self.DIM:(i+2)*self.DIM]
        self.x_u[(self.N-1)*self.DIM:(self.N-1)*self.DIM + self.STATE_DIM] = \
            self.x_u[(self.N)*self.DIM:(self.N)*self.DIM + self.STATE_DIM]
        
        # Return action as torch tensor on the correct device
        return torch.tensor(action, device=self.device, dtype=torch.float32)
    
    def explore_env(self, env, timesteps: int, random: bool = False, sample: bool = False):
        """Collect experience from the environment - runs the MPC controller."""
        # For MPC, we don't need to collect experience in the traditional sense
        # We just run the environment with our MPC controller
        
        # Create trajectory storage arrays (following framework convention)
        traj_obs = {
            k: torch.empty((self.num_actors, timesteps) + v, dtype=torch.float32, device=self.device)
            for k, v in self.obs_space.items()
        }
        traj_actions = torch.empty((self.num_actors, timesteps) + (self.action_dim,), device=self.device)
        traj_rewards = torch.empty((self.num_actors, timesteps), device=self.device)
        traj_next_obs = {
            k: torch.empty((self.num_actors, timesteps) + v, dtype=torch.float32, device=self.device)
            for k, v in self.obs_space.items()
        }
        traj_dones = torch.empty((self.num_actors, timesteps), device=self.device)

        # Main environment interaction loop
        for i in range(timesteps):
            if not self.env_autoresets:
                raise NotImplementedError  # framework expects auto-resetting environments

            # Generate actions - either random exploration or MPC control
            if random:
                actions = torch.rand((self.num_actors, self.action_dim), device=self.device) * 2.0 - 1.0
            else:
                actions = self.get_actions(self.obs, sample=sample)  # MPC optimization

            # Step the environment with the actions
            next_obs, rewards, dones, infos = env.step(actions)
            next_obs = self._convert_obs(next_obs)  # convert to tensors on correct device

            # Update metrics for logging and visualization
            done_indices = torch.where(dones)[0].tolist()
            self.metrics.update(self.epoch, self.env, self.obs, rewards, done_indices, infos)

            # Store trajectory data (for compatibility with framework)
            for k, v in self.obs.items():
                traj_obs[k][:, i] = v
            traj_actions[:, i] = actions
            traj_dones[:, i] = dones
            traj_rewards[:, i] = rewards
            for k, v in next_obs.items():
                traj_next_obs[k][:, i] = v
            self.obs = next_obs  # update current observation

        # Flush video recording if enabled
        self.metrics.flush_video(self.epoch)

        # For MPC, we don't need to return trajectory data for training
        # Just return empty data structure to maintain framework compatibility
        data = None
        return data, timesteps * self.num_actors  # return data and step count
    
    def train(self):
        """Training loop for MPC agent - for MPC this just runs the controller."""
        # Initialize environment
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        
        # Main training loop
        while self.agent_steps < self.max_agent_steps:
            self.epoch += 1
            
            # For MPC, we don't need traditional training
            # Just run the environment with our controller
            self.set_train()
            trajectory, steps = self.explore_env(self.env, self.dms1_config.get('horizon_len', 100))
            self.agent_steps += steps
            
            # Log metrics for monitoring performance
            episode_metrics = {
                "train_scores/episode_rewards": self.metrics.episode_trackers["rewards"].mean(),
                "train_scores/episode_lengths": self.metrics.episode_trackers["lengths"].mean(),
                "train_scores/num_episodes": self.metrics.num_episodes,
                **self.metrics.result(prefix="train"),
            }
            
            # Write metrics to tensorboard/wandb
            self.writer.add(self.agent_steps, episode_metrics)
            self.writer.write()
            
            # Save checkpoints periodically
            if self.ckpt_every > 0 and (self.epoch + 1) % self.ckpt_every == 0:
                self._checkpoint_save(self.metrics.episode_trackers["rewards"].mean())
            
            

    def eval(self):
        raise NotImplementedError("DMS1 doesnt has eval()")

    def set_train(self):
        """Set to training mode - no-op for MPC since it has no trainable networks."""
        # MPC doesn't have trainable networks, so this is a no-op
        pass

    def set_eval(self):
        raise NotImplementedError("DMS1 doesnt has set_eval()")

    def save(self, f):
        raise NotImplementedError("DMS1 doesnt has save()")

    def load(self, f, ckpt_keys=''):
        raise NotImplementedError("DMS1 doesnt has load()")
