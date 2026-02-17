"""Single Shooting and Direct Multiple Shooting MPC with PyTorch Adam Optimizer.

This implementation provides two MPC methods with differentiable physics simulation:

1. SINGLE SHOOTING MPC (Default):
   - Optimizes only control sequence u = [u0, ..., u_{N-1}]
   - Dynamics satisfied implicitly by simulation rollout
   - Loss = -cumulative_reward (maximize reward)
   - Simpler and faster than DMS

2. TRUE Direct Multiple Shooting (DMS):
   - Optimizes both states and controls x = [x0, u0, ..., x_N]
   - Dynamics enforced via penalty method
   - Loss = task_cost + penalty_weight * constraint_violation

Single Shooting Algorithm:
    for timesteps:
        u = single_shooting(xk, u_next)  # optimize controls
        env.step(u[0])                   # execute first control
        u_next = [u1, ..., u_{N-1}, u_{N-1}]  # warm start shift

Key Features:
- PyTorch Adam: First-order gradient-based optimization
- Analytical Gradients: All gradients via PyTorch autograd through Rewarped
- Warm Starting: Control sequence shifted for next planning iteration
- Full State Representation: All MPM particle positions + rigid body configuration
"""

import json
import os
import time
import warnings
from typing import Any, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import warp as wp
from omegaconf import OmegaConf

from mineral.agents.agent import Agent
from mineral.agents.diffrl.shac import SHAC

# from scipy.optimize import minimize

# warnings.simplefilter('always')
warnings.filterwarnings('ignore', message='.*grad attribute of a Tensor that is not a leaf.*')


class DSSMPCAgent(Agent):
    r"""Single Shooting and Direct Multiple Shooting MPC Agent.

    This agent provides two MPC planning methods:

    1. SINGLE SHOOTING (Default - run_single_shooting_mpc):
        - Optimizes control sequence: u = [u0, ..., u_{N-1}]
        - Loss: -cumulative_reward from simulation rollout
        - Dynamics satisfied implicitly via simulation
        - Warm starting: shift controls between timesteps

    2. Direct Multiple Shooting (run_dms_mpc):
        - Optimizes states and controls: [x0, u0, ..., x_N]
        - Loss: task_cost + λ * ||x[i+1] - f(x[i], u[i])||^2
        - Dynamics enforced via penalty method

    Single Shooting Optimization:
        minimize_{u}  -Σ reward(x[i], u[i])
        subject to: x[i+1] = f(x[i], u[i])  (implicit via simulation)

    State Representation:
        - State x: ALL MPM particle positions + rigid body configuration
        - Control u: Action space from environment (e.g., 3D for RollingPin)

    Optimization:
        - PyTorch Adam optimizer with analytical gradients
        - Gradients via autograd through differentiable physics (Rewarped)
    """

    def __init__(self, full_cfg: Any, **kwargs: Any) -> None:
        self.network_config = full_cfg.agent.network
        self.params = full_cfg.agent.params
        self.num_actors = self.params.num_actors
        self.max_agent_steps = int(self.params.max_agent_steps)
        self.render_results = self.params.render_results

        # DMS MPC Parameters
        self.dss_mpc_params = full_cfg.agent.dss_mpc_params
        # self.dt = 0.1
        self.N = self.dss_mpc_params.N  # Number of shooting nodes
        # self.H = self.dss_mpc_params.N * self.dt  # Horizon length
        self.timesteps = self.dss_mpc_params.timesteps
        self.max_iter = self.dss_mpc_params.max_iter

        # Cost parameters
        self.cost_state = self.dss_mpc_params.cost_state
        self.cost_control = self.dss_mpc_params.cost_control
        self.cost_terminal = self.dss_mpc_params.cost_terminal
        self.penalty_weight = self.dss_mpc_params.get('penalty_weight', 1000.0)  # Penalty for constraint violations
        self.learning_rate = self.dss_mpc_params.get('learning_rate', 0.01)  # Adam learning rate

        # SHAC checkpoint for initial guess
        # self.shac_ckpt_path = self.dss_mpc_params.get('shac_ckpt_path', None)
        self.shac_agent = None

        super().__init__(full_cfg, **kwargs)

        # Get control dimension from parent Agent class
        self.control_dim = self.action_dim

        self.obs = None
        self.dones = None

        # Single shooting: warm start control sequence
        self._u_warm_start = None

        self.sapo_sample = self.dss_mpc_params.sapo_sample
        self.run_sapo_seperately = self.dss_mpc_params.run_sapo_seperately

        # Load SHAC agent (hardcoded paths)
        self._load_shac_agent()

    def _load_shac_agent(self) -> None:
        """Load SHAC agent from checkpoint for generating initial control sequences."""
        # Hardcoded paths
        shac_ckpt_path = "/app/workdir/RewarpedRollingPin4M-SAPO/20260101-230641.08/ckpt/best_rewards101.38.pth"
        config_path = "/app/workdir/RewarpedRollingPin4M-SAPO/20260101-230641.08/resolved_config.yaml"

        print(f"\n{'=' * 70}")
        print("Loading SHAC agent from checkpoint")
        print(f"  Checkpoint path: {shac_ckpt_path}")
        print(f"  Config path: {config_path}")
        print(f"{'=' * 70}\n")

        # Verify paths exist
        if not os.path.exists(shac_ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {shac_ckpt_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load config
        print(f"Loading config from: {config_path}")
        shac_cfg = OmegaConf.load(config_path)

        # Update config to use the same environment
        shac_cfg.task = self.full_cfg.task
        shac_cfg.rl_device = self.device
        shac_cfg.multi_gpu = False  # Disable multi-gpu for inference
        # Update num_actors to match the current environment
        shac_cfg.agent.shac.num_actors = self.env.num_envs

        # Initialize SHAC agent
        # Use the same logdir as the DSSMPC agent (set by super().__init__)
        self.shac_agent = SHAC(shac_cfg, logdir=self.logdir, env=self.env)

        # Load checkpoint
        print(f"Loading checkpoint: {shac_ckpt_path}")
        self.shac_agent.load(shac_ckpt_path, ckpt_keys='')
        self.shac_agent.set_eval()  # Set to eval mode for inference
        self.shac_agent.tanh_clamp = False

        print("✓ SHAC agent loaded successfully\n")

    def get_shac_initial_sequence(self, obs: dict, horizon: int, checkpoint=None) -> torch.Tensor:
        """Generate initial control sequence using SHAC policy rollout with full environment simulation.

        This method performs a full environment rollout using SHAC policy actions, providing
        a better initial guess for MPC optimization than just using policy mean actions.

        Args:
            obs: Current observation dictionary
            horizon: Horizon length N
            checkpoint: Optional checkpoint to use. If None, saves a new checkpoint.

        Returns:
            u_init: Initial control sequence, shape (N, control_dim)
        """
        if self.shac_agent is None:
            return None

        # Save current environment state (or use provided checkpoint)
        # checkpoint = self.env.get_checkpoint(detach=True)
        # checkpoint = self.clone_state(self.env.state_0)
        original_no_grad = getattr(self.env, 'no_grad', False)
        # Don't set no_grad=True here - it might prevent environment state updates
        # Instead, just use torch.no_grad() context for tensor operations

        # try:
        with torch.no_grad():
            # Normalize initial observation if needed (same as SHAC)
            current_obs = obs
            # Copied from shac.py line 265-266, because normalize_input was True (default) during training
            if self.shac_agent.obs_rms is not None:
                current_obs = {k: self.shac_agent.obs_rms[k].normalize(v) for k, v in obs.items()}

            # Roll out policy for horizon steps through actual environment
            u_init = []
            current_obs_dict = current_obs

            # prev_obs_dict = None
            for step in range(horizon):
                # Debug: print observation info and check if it changed
                # if step < 2:
                #     print(f"\nStep {step}: Observation keys: {list(current_obs_dict.keys())}")
                #     for k, v in current_obs_dict.items():
                #         if isinstance(v, torch.Tensor):
                #             print(f"  {k}: shape={v.shape}, mean={v.mean().item():.4f}, std={v.std().item():.4f}")
                #     if prev_obs_dict is not None:
                #         # Check if observations changed
                #         obs_changed = False
                #         for k in current_obs_dict.keys():
                #             if k in prev_obs_dict:
                #                 if isinstance(current_obs_dict[k], torch.Tensor) and isinstance(prev_obs_dict[k], torch.Tensor):
                #                     diff = (current_obs_dict[k] - prev_obs_dict[k]).abs().max().item()
                #                     if diff > 1e-6:
                #                         obs_changed = True
                #                         print(f"  {k} changed: max_diff={diff:.6f}")
                #         if not obs_changed:
                #             print("  WARNING: Observations did not change from previous step!")

                # Get action from SHAC policy (use mean, not sampled, for deterministic initial guess)
                # try:
                action = self.shac_agent.get_actions(current_obs_dict, sample=self.sapo_sample)


                action_single = action[0]  # Take first actor's action / Unwrap normally multiple agents
                # print(f"Step {step}: Action = [{action_single[0].item():.4f}, {action_single[1].item():.4f}, {action_single[2].item():.4f}]")
                u_init.append(action_single.clone())
                # except Exception as e:
                #     print(f"Error getting action at step {step}: {e}")
                #     # Use zero action as fallback
                #     action_single = torch.zeros(self.control_dim, device=self.device)
                #     u_init.append(action_single.clone())

                # Expand action for all actors and step environment
                actions = action_single.unsqueeze(0).repeat(self.num_actors, 1)
                next_obs, reward, done, _ = self.env.step(actions)

                # Debug: check if raw observation changed before normalization
                # if step < 2:
                #     print(f"Step {step}: After env.step, reward={reward[0].item():.4f}, done={done[0].item()}")
                #     next_obs_raw = self._convert_obs(next_obs)
                #     print(f"Step {step}: Raw observation (before normalization):")
                #     for k, v in next_obs_raw.items():
                #         if isinstance(v, torch.Tensor):
                #             print(f"  {k}: mean={v.mean().item():.6f}, std={v.std().item():.6f}, min={v.min().item():.6f}, max={v.max().item():.6f}")

                # Convert and normalize next observation (same as SHAC)
                next_obs = self._convert_obs(next_obs)
                if self.shac_agent.obs_rms is not None:
                    current_obs_dict = {k: self.shac_agent.obs_rms[k].normalize(v) for k, v in next_obs.items()}
                # else:
                #     current_obs_dict = next_obs

                # Save current observation for comparison in next iteration
                # prev_obs_dict = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in current_obs_dict.items()}

                # If episode terminates early, repeat last action for remaining steps
                if done[0]:
                    print(f"Simulation terminated early at step {step}")
                    # Fill remaining steps with last action
                    last_action = u_init[-1]
                    for _ in range(horizon - step - 1):
                        u_init.append(last_action.clone())
                    break

            u_init = torch.stack(u_init)  # Shape: (N, control_dim)

        # finally:
        # Restore environment state
        # self.env.clear_grad(checkpoint)
        # self.env.state_0 = self.clone_state(checkpoint)
        if hasattr(self.env, 'no_grad'):
            self.env.no_grad = original_no_grad

        return u_init

    def clone_state(self, state: Any) -> Any:
        """Clone state by copying all warp arrays and MPM structures."""
        s = type(state)()

        # Attributes to clone with wp.clone
        wp_clone_attrs = [
            "body_f", "body_q", "body_qd", "joint_q", "joint_qd",
            "mpm_C", "mpm_F", "mpm_F_trial", "mpm_grid_m", "mpm_grid_mv",
            "mpm_grid_v", "mpm_stress", "mpm_x", "mpm_v",
        ]

        for attr in wp_clone_attrs:
            if hasattr(state, attr):
                setattr(s, attr, wp.clone(getattr(state, attr)))

        return s

    # =========================================================================
    # SINGLE SHOOTING MPC IMPLEMENTATION
    # =========================================================================

    def single_shooting_loss(self, u_seq: torch.Tensor, checkpoint, debug: bool = False) -> torch.Tensor:
        """Single shooting loss: negative cumulative reward from rollout.

        Rolls out the simulation from checkpoint using control sequence u_seq,
        accumulates rewards, and returns negative total (for minimization).

        Args:
            u_seq: Control sequence, shape (N, control_dim)
            checkpoint: Environment checkpoint to restore from (from env.get_checkpoint())
            debug: If True, print gradient tracking debug info

        Returns:
            loss: Negative cumulative reward (scalar tensor)
        """
        if debug:
            print(f"[DEBUG] u_seq.requires_grad: {u_seq.requires_grad}")
            print(f"[DEBUG] env.requires_grad: {self.env.requires_grad}")
            print(f"[DEBUG] env.no_grad: {self.env.no_grad}")
            print(f"[DEBUG] env.control_tensors AFTER clear_grad: {[t.requires_grad for t in self.env.control_tensors]}")
            print(f"[DEBUG] env.control_tensors is_leaf: {[t.is_leaf for t in self.env.control_tensors]}")

        # Accumulate rewards in a list to maintain gradient graph
        rewards = []

        # Roll out simulation with control sequence
        for i in range(self.N):
            # Expand for all actors
            actions = u_seq[i].unsqueeze(0).repeat(self.num_actors, 1)

            if debug and i == 0:
                print(f"[DEBUG] actions.requires_grad: {actions.requires_grad}")
                print(f"[DEBUG] actions.grad_fn: {actions.grad_fn}")
                print(f"[DEBUG] control_tensors before step: {[t.requires_grad for t in self.env.control_tensors]}")
                print(f"[DEBUG] control_tensors[0].grad_fn before step: {self.env.control_tensors[0].grad_fn}")

            # Step environment (differentiable!)
            obs, reward, done, _ = self.env.step(actions)

            if debug and i == 0:
                print("env.step(actions)")
                print(f"[DEBUG] actions.requires_grad: {actions.requires_grad}")
                print(f"[DEBUG] actions.grad_fn: {actions.grad_fn}")
                print(f"[DEBUG] control_tensors before step: {[t.requires_grad for t in self.env.control_tensors]}")
                print(f"[DEBUG] control_tensors[0].grad_fn AFTER step: {self.env.control_tensors[0].grad_fn}")
                print(f"[DEBUG] reward.requires_grad: {reward.requires_grad}")
                print(f"[DEBUG] reward[0].requires_grad: {reward[0].requires_grad}")
                print(f"[DEBUG] reward.grad_fn: {reward.grad_fn}")
                print("[DEBUG] Tracing gradient path from reward back to actions...")
                # Try to manually compute gradient to see where it breaks
                test_grad = torch.autograd.grad(reward[0], actions, retain_graph=True, allow_unused=True)[0]
                print(f"[DEBUG]   torch.autograd.grad(reward[0], actions) = {test_grad}")

            # Collect reward (should be tensor with gradient from differentiable env)
            rewards.append(reward[0])

        # Sum rewards - this maintains gradient connections
        cumulative_reward = torch.stack(rewards).sum()

        if debug:
            print(f"[DEBUG] cumulative_reward.requires_grad: {cumulative_reward.requires_grad}")
            print(f"[DEBUG] cumulative_reward.grad_fn: {cumulative_reward.grad_fn}")

        # Return negative reward (minimize loss = maximize reward)
        return -cumulative_reward

    def single_shooting_plan(self, u_init: torch.Tensor = None, obs: dict = None) -> torch.Tensor:
        """Single shooting MPC planning with Adam optimizer.

        Optimizes control sequence u = [u0, ..., u_{N-1}] to maximize cumulative reward.

        Args:
            u_init: Initial control sequence for warm start, shape (N, control_dim)
            obs: Current observation dictionary (used for SHAC initial guess if u_init is None)

        Returns:
            u_opt: Optimized control sequence, shape (N, control_dim)
        """
        # Save original state for rollouts BEFORE any modifications
        # Enable gradient computation in environment
        original_no_grad = getattr(self.env, 'no_grad', False)
        self.env.no_grad = False

        # Save checkpoint for restoring state between optimization iterations
        # This checkpoint will be used by both get_shac_initial_sequence and optimization
        # False: keep gradients for control tensors
        # True: detach control tensors from gradient graph
        checkpoint = self.env.get_checkpoint(detach=True)
        # checkpoint = self.clone_state(self.env.state_0)
        print(f"\n{'=' * 70}")

        # Initialize control sequence
        # if u_init is not None:
        #     u_seq = u_init.clone().detach().requires_grad_(True)
        # elif self.shac_agent is not None and obs is not None:
            # Use SHAC policy to generate initial sequence
        print("Generating initial control sequence using SHAC policy...")
        # Pass checkpoint to ensure it uses the same starting state as optimization
        shac_u_init = self.get_shac_initial_sequence(obs, self.N, checkpoint=checkpoint)
        if shac_u_init is not None:
            u_seq = shac_u_init.clone().detach().requires_grad_(True)
            # Print the generated action sequence
            print(f"Generated SHAC initial action sequence (shape: {u_seq.shape}):")
            for i, action in enumerate(u_seq):
                print(f"  u[{i}]: [{action[0].item():.4f}, {action[1].item():.4f}, {action[2].item():.4f}]")
            print(f"  (mean norm: {u_seq.norm(dim=-1).mean().item():.4f})")
        #     else:
        #         u_seq = torch.zeros(
        #             (self.N, self.control_dim),
        #             dtype=torch.float32,
        #             device=self.device,
        #             requires_grad=True
        #         )
        # else:
        #     u_seq = torch.zeros(
        #         (self.N, self.control_dim),
        #         dtype=torch.float32,
        #         device=self.device,
        #         requires_grad=True
        #     )

        # Restore environment state from checkpoint and reinitialize gradient tape
        # self.env.clear_grad(checkpoint)
        # self.env.state_0 = self.clone_state(checkpoint)

        print("Starting Single Shooting Optimization (PyTorch Adam)")
        print(f"  Horizon: N={self.N} | Learning rate: {self.learning_rate} | Max iterations: {self.max_iter}")
        print(f"{'-' * 70}\n")

        # Create optimizer
        u_param = torch.nn.Parameter(u_seq)
        optimizer = torch.optim.Adam([u_param], lr=self.learning_rate)

        start_time = time.time()

        # Track optimization metrics
        losses_per_iteration = []
        grad_norms_per_iteration = []
        first_loss = None

        for iteration in range(self.max_iter):
            optimizer.zero_grad()

            # Compute loss (negative cumulative reward)
            # This restores from checkpoint and rolls out with gradient tracking
            # Restore environment state from checkpoint and reinitialize gradient tape
            self.env.clear_grad(checkpoint)
            # self.env.state_0 = self.clone_state(checkpoint)
            debug = False # (iteration == 0)  # Only print debug info on first iteration
            loss = self.single_shooting_loss(u_param, checkpoint, debug=debug)

            # Track loss
            loss_value = loss.item()
            losses_per_iteration.append(loss_value)

            if first_loss is None:
                first_loss = loss_value

            # Backward pass
            loss.backward()

            # Get gradient info
            if debug:
                print("[DEBUG] After backward:")
                print(f"[DEBUG]   u_param.grad is None: {u_param.grad is None}")
                if u_param.grad is not None:
                    print(f"[DEBUG]   u_param.grad.norm(): {u_param.grad.norm().item()}")
                    print(f"[DEBUG]   u_param.grad sample: {u_param.grad[0]}")
            grad_norm = u_param.grad.norm().item() if u_param.grad is not None else 0.0
            grad_norms_per_iteration.append(grad_norm)

            # Optimizer step
            optimizer.step()

            # Clip controls to valid range [-1, 1]
            # with torch.no_grad():
            #     u_param.data.clamp_(-1.0, 1.0)

            # Print progress
            if iteration % 10 == 0 or iteration == self.max_iter - 1:
                print(f"Iteration {iteration:4d}/{self.max_iter - 1} | Loss: {loss_value:.6f} | "
                        f"Grad: {grad_norm:.4e}", flush=True)

        elapsed_time = time.time() - start_time
        final_loss = losses_per_iteration[-1] if losses_per_iteration else None

        # Restore environment gradient setting
        self.env.no_grad = original_no_grad

        u_opt = u_param.data.clone().detach()

        # Restore original state
        # self.env.state_0 = original_state
        # Restore state from checkpoint for actual execution
        self.env.clear_grad(checkpoint)
        # self.env.state_0 = self.clone_state(checkpoint)

        print(f"\n{'-' * 70}")
        print("Single Shooting Optimization Complete")
        # print(f"  Final Loss: {best_loss:.6f} (Reward: {-best_loss:.6f})")
        print(f"  Final Loss: {final_loss:.6f}")
        print(f"  Loss difference: {final_loss - first_loss:.6f}")
        print(f"  Time: {elapsed_time:.2f}s")
        print("  Optimal controls:")
        for i, u in enumerate(u_opt):
            print(f"    u[{i}]: [{u[0].item():.4f}, {u[1].item():.4f}, {u[2].item():.4f}]")
        print(f"{'=' * 70}\n")

        # Compute action statistics
        action_mean = u_opt.mean(dim=0).cpu().numpy().tolist()
        action_std = u_opt.std(dim=0).cpu().numpy().tolist()
        action_norm_mean = u_opt.norm(dim=-1).mean().item()

        # Return optimization results with metrics
        opt_metrics = {
            "losses_per_iteration": losses_per_iteration,
            "grad_norms_per_iteration": grad_norms_per_iteration,
            "first_loss": first_loss,
            "final_loss": final_loss,
            "loss_improvement": first_loss - final_loss,  # Positive = improvement
            "optimization_time": elapsed_time,
            "action_mean": action_mean,
            "action_std": action_std,
            "action_norm_mean": action_norm_mean,
        }

        return u_opt, opt_metrics

    def run_single_shooting_mpc(self) -> None:
        """Main MPC loop using Single Shooting.

        Runs DSS MPC and SAPO in parallel on independent environments,
        starting from the same initial state.

        Following the pseudo code:
            for timesteps:
                u = single_shooting(xk, u_next)  # DSS MPC
                env_dss.step(u[0])               # execute DSS control
                env_sapo.step(sapo_action)        # execute SAPO control
                u_next = [u1, ..., u_{N-1}, u_{N-1}]  # warm start shift
        """
        # IMPORTANT: Enable gradient computation BEFORE reset
        # so that control_tensors are properly initialized for differentiation
        self.env.no_grad = False

        # Initialize main environment (DSS)
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)

        # Save initial checkpoint to clone state
        initial_checkpoint = self.env.get_checkpoint(detach=True)

        # Conditionally create and initialize SAPO environment with same initial state
        env_sapo = None
        obs_sapo_dict = None
        dones_sapo = None
        if self.run_sapo_seperately:
            # Use dynamic import based on task suite (same pattern as run.py)
            from ... import envs
            task_suite = self.full_cfg.task.get('suite', 'isaacgymenvs')
            TaskSuite = getattr(envs, task_suite)

            # Create SAPO environment config (disable rendering to avoid USD path conflicts)
            sapo_cfg = OmegaConf.create(OmegaConf.to_container(self.full_cfg, resolve=True))
            sapo_cfg.task.env.render = False  # Disable rendering for SAPO env

            env_sapo = TaskSuite.make_envs(sapo_cfg)
            env_sapo.no_grad = False
            obs_sapo = env_sapo.reset()
            env_sapo.clear_grad(initial_checkpoint)  # Restore to same initial state
            obs_sapo_dict = self._convert_obs(obs_sapo)
            dones_sapo = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)

        # Track rewards separately (per timestep and totals)
        total_dss_reward = 0.0
        total_sapo_reward = 0.0
        dss_rewards_per_timestep = []
        sapo_rewards_per_timestep = []


        # Let the dough fall to the ground and stabilize (25 steps)
        # print("\nLetting dough fall and stabilize for 25 steps...")
        # zero_action = torch.zeros((self.num_actors, self.control_dim), device=self.device)
        # move_down_action = torch.tensor([0.0, -1, 0.0], device=self.device)
        # for _ in range(5):
        #     obs, reward, done, _ = self.env.step(move_down_action)
        #     self.obs = self._convert_obs(obs)
        #     self.dones = done
        # obs, reward, done, _ = self.env.step(zero_action)
        # self.obs = self._convert_obs(obs)
        # self.dones = done
        # print("Dough stabilization complete.\n")

        # Initialize warm start control sequence
        u_next = None
        # u_next = torch.zeros(self.N, self.control_dim, device=self.device)
        # u_next[:, 1] = -1.0

        # Lists to save executed actions
        executed_actions_dss = []
        executed_actions_sapo = []
        # Track optimization metrics per timestep
        optimization_metrics_per_timestep = []

        print(f"\n{'#' * 70}")
        if self.run_sapo_seperately:
            print(f"{'SINGLE SHOOTING MPC vs SAPO':^70}")
        else:
            print(f"{'SINGLE SHOOTING MPC':^70}")
        print(f"{'#' * 70}")
        print(f"Horizon: N={self.N} steps")
        print(f"Control dim: {self.control_dim}")
        print(f"Optimizer: Adam (lr={self.learning_rate})")
        print(f"Timesteps: {self.timesteps}")
        print(f"Run SAPO separately: {self.run_sapo_seperately}")
        print(f"{'#' * 70}\n")

        for timestep in range(self.timesteps):
            # Check if either environment is done
            if self.dones[0]:
                if not self.run_sapo_seperately or (dones_sapo is not None and dones_sapo[0]):
                    break

            print(f"\n{'#' * 70}")
            print(f"Timestep {timestep + 1}/{self.timesteps}")
            print(f"{'#' * 70}")

            # ========== DSS MPC ==========
            if not self.dones[0]:
                # Disable renderer during optimization
                saved_renderer = self.env.renderer
                self.env.renderer = None

                # Plan using single shooting (returns u_opt and optimization metrics)
                u_opt, opt_metrics = self.single_shooting_plan(u_next, obs=self.obs)
                opt_metrics["timestep"] = timestep
                optimization_metrics_per_timestep.append(opt_metrics)

                # Re-enable renderer
                self.env.renderer = saved_renderer

                # Execute first control
                u_0_dss = u_opt[0]
                actions_dss = u_0_dss.unsqueeze(0).repeat(self.num_actors, 1)
                obs, reward_dss, done, _ = self.env.step(actions_dss)
                self.obs = self._convert_obs(obs)
                self.dones = done

                reward_dss_val = reward_dss[0].item() if isinstance(reward_dss, torch.Tensor) else reward_dss[0]
                total_dss_reward += reward_dss_val
                dss_rewards_per_timestep.append(reward_dss_val)
                executed_actions_dss.append(u_0_dss.clone())

                print(f"DSS - Action: [{u_0_dss[0].item():.4f}, {u_0_dss[1].item():.4f}, {u_0_dss[2].item():.4f}]")
                print(f"DSS - Reward: {reward_dss_val:.4f} | Cumulative: {total_dss_reward:.4f}")

                # Warm start: shift controls
                u_next = torch.zeros_like(u_opt)
                u_next[:-1] = u_opt[1:]
                u_next[-1] = u_opt[-1]

            # ========== SAPO Policy ==========
            if self.run_sapo_seperately and dones_sapo is not None and not dones_sapo[0]:
                # Normalize observation if needed
                if self.shac_agent.obs_rms is not None:
                    obs_sapo_normalized = {k: self.shac_agent.obs_rms[k].normalize(v)
                                         for k, v in obs_sapo_dict.items()}

                # Get SAPO action
                action_sapo = self.shac_agent.get_actions(obs_sapo_normalized, sample=self.sapo_sample)
                action_sapo_single = action_sapo[0]

                # Execute SAPO action
                actions_sapo = action_sapo_single.unsqueeze(0).repeat(self.num_actors, 1)
                obs_sapo, reward_sapo, done_sapo, _ = env_sapo.step(actions_sapo)
                obs_sapo_dict = self._convert_obs(obs_sapo)
                dones_sapo = done_sapo

                reward_sapo_val = reward_sapo[0].item() if isinstance(reward_sapo, torch.Tensor) else reward_sapo[0]
                total_sapo_reward += reward_sapo_val
                sapo_rewards_per_timestep.append(reward_sapo_val)
                executed_actions_sapo.append(action_sapo_single.clone())

                print(f"SAPO - Action: [{action_sapo_single[0].item():.4f}, {action_sapo_single[1].item():.4f}, {action_sapo_single[2].item():.4f}]")
                print(f"SAPO - Reward: {reward_sapo_val:.4f} | Cumulative: {total_sapo_reward:.4f}")

        # Print final results
        print(f"\n{'#' * 70}")
        if self.run_sapo_seperately:
            print(f"{'COMPARISON RESULTS':^70}")
        else:
            print(f"{'DSS MPC RESULTS':^70}")
        print(f"{'#' * 70}")
        print(f"DSS MPC Total Reward: {total_dss_reward:.4f}")
        if self.run_sapo_seperately:
            print(f"SAPO Total Reward: {total_sapo_reward:.4f}")
            print(f"Difference: {total_dss_reward - total_sapo_reward:.4f}")
        if len(executed_actions_dss) > 0:
            print(f"DSS Average Reward: {total_dss_reward / len(executed_actions_dss):.4f}")
        if self.run_sapo_seperately and len(executed_actions_sapo) > 0:
            print(f"SAPO Average Reward: {total_sapo_reward / len(executed_actions_sapo):.4f}")
        print(f"{'#' * 70}\n")

        # Save trajectories to logdir
        if executed_actions_dss:
            actions_tensor_dss = torch.stack(executed_actions_dss)
            trajectory_path_dss = os.path.join(self.logdir, 'trajectory_dss_mpc.pt')
            torch.save(actions_tensor_dss, trajectory_path_dss)
            print(f"✓ DSS trajectory saved to {trajectory_path_dss}")

        if executed_actions_sapo:
            actions_tensor_sapo = torch.stack(executed_actions_sapo)
            trajectory_path_sapo = os.path.join(self.logdir, 'trajectory_sapo.pt')
            torch.save(actions_tensor_sapo, trajectory_path_sapo)
            print(f"✓ SAPO trajectory saved to {trajectory_path_sapo}")

        # Compute aggregate optimization statistics
        total_opt_time = sum(m["optimization_time"] for m in optimization_metrics_per_timestep) if optimization_metrics_per_timestep else 0.0
        avg_opt_time = total_opt_time / len(optimization_metrics_per_timestep) if optimization_metrics_per_timestep else 0.0
        avg_loss_improvement = sum(m["loss_improvement"] for m in optimization_metrics_per_timestep) / len(optimization_metrics_per_timestep) if optimization_metrics_per_timestep else 0.0

        # Save comparison scores to my_scores.json
        scores = {
            "dss_mpc": {
                "total_reward": total_dss_reward,
                "num_timesteps": len(executed_actions_dss),
                "average_reward": total_dss_reward / len(executed_actions_dss) if len(executed_actions_dss) > 0 else 0.0,
                "rewards_per_timestep": dss_rewards_per_timestep,
            },
            "sapo": {
                "total_reward": total_sapo_reward,
                "num_timesteps": len(executed_actions_sapo),
                "average_reward": total_sapo_reward / len(executed_actions_sapo) if len(executed_actions_sapo) > 0 else 0.0,
                "rewards_per_timestep": sapo_rewards_per_timestep,
            },
            "optimization": {
                "total_optimization_time": total_opt_time,
                "average_optimization_time_per_timestep": avg_opt_time,
                "average_loss_improvement": avg_loss_improvement,
                "optimization_metrics_per_timestep": optimization_metrics_per_timestep,
            },
            "comparison": {
                "reward_difference": total_dss_reward - total_sapo_reward if self.run_sapo_seperately else None,
                "dss_advantage_percent": ((total_dss_reward - total_sapo_reward) / abs(total_sapo_reward) * 100) if self.run_sapo_seperately and total_sapo_reward != 0 else None,
            },
            "config": {
                "horizon_N": self.N,
                "control_dim": self.control_dim,
                "learning_rate": self.learning_rate,
                "max_iter": self.max_iter,
                "timesteps": self.timesteps,
            }
        }
        scores_path = os.path.join(self.logdir, "my_scores.json")
        json.dump(scores, open(scores_path, "w"), indent=4)
        print(f"✓ Scores saved to {scores_path}")

        # Create comparison plots from collected data
        self._create_comparison_plots(dss_rewards_per_timestep, sapo_rewards_per_timestep, total_dss_reward, total_sapo_reward)

    def _create_comparison_plots(self, rewards_dss, rewards_sapo, total_dss, total_sapo):
        """Create comparison plots from collected reward data.

        Handles both cases:
        - When SAPO is run: creates comparison plots between DSS MPC and SAPO
        - When SAPO is not run: creates DSS MPC-only plots
        """
        if not rewards_dss and not rewards_sapo:
            print("No reward data to plot")
            return

        has_sapo = rewards_sapo and len(rewards_sapo) > 0

        # Main comparison/performance plot
        plt.figure(figsize=(12, 6))
        if rewards_dss:
            plt.plot(rewards_dss, marker='x', color='blue', linewidth=1, markersize=3, label=f'DSS MPC (Total: {total_dss:.2f})', alpha=0.8)
        if has_sapo:
            plt.plot(rewards_sapo, marker='x', color='red', linewidth=1, markersize=3, label=f'SAPO (Total: {total_sapo:.2f})', alpha=0.8)
        plt.xlabel("Timestep", fontsize=12)
        plt.ylabel("Reward", fontsize=12)
        if has_sapo:
            plt.title("DSS MPC vs SAPO Performance Comparison", fontsize=14, fontweight='bold')
        else:
            plt.title("DSS MPC Performance", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(self.logdir, "reward_plot_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Comparison plot saved to {plot_path}")

        # Individual plots
        if has_sapo:
            # Side-by-side comparison plots (both DSS and SAPO)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Compute max length for x and y axis padding
            max_len = max(len(rewards_dss), len(rewards_sapo))
            # Compute global min/max for y limits
            all_rewards = list(rewards_dss) + list(rewards_sapo)
            min_y = 0
            max_y = max(all_rewards)

            # DSS plot
            ax1.plot(rewards_dss, marker='x', color='blue', linewidth=1, markersize=3, alpha=0.8)
            ax1.set_xlabel("Timestep", fontsize=12)
            ax1.set_ylabel("Reward", fontsize=12)
            ax1.set_title(f"DSS MPC\nTotal: {total_dss:.2f} | Avg: {total_dss/len(rewards_dss):.2f}", fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
            ax1.set_xlim(0, max_len - 1)
            ax1.set_ylim(min_y, max_y)

            # SAPO plot
            ax2.plot(rewards_sapo, marker='x', color='red', linewidth=1, markersize=3, alpha=0.8)
            ax2.set_xlabel("Timestep", fontsize=12)
            ax2.set_ylabel("Reward", fontsize=12)
            ax2.set_title(f"SAPO\nTotal: {total_sapo:.2f} | Avg: {total_sapo/len(rewards_sapo):.2f}", fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
            ax2.set_xlim(0, max_len - 1)
            ax2.set_ylim(min_y, max_y)

            plt.tight_layout()
            plot_path_individual = os.path.join(self.logdir, "reward_plot_individual.png")
            plt.savefig(plot_path_individual, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Individual plots saved to {plot_path_individual}")

            # Cumulative reward comparison
            fig, ax = plt.subplots(figsize=(12, 6))
            cumulative_dss = np.cumsum(rewards_dss)
            cumulative_sapo = np.cumsum(rewards_sapo)
            ax.plot(cumulative_dss, marker='x', color='blue', linewidth=1, markersize=3, label='DSS MPC (Cumulative)', alpha=0.8)
            ax.plot(cumulative_sapo, marker='x', color='red', linewidth=1, markersize=3, label='SAPO (Cumulative)', alpha=0.8)
            ax.set_xlabel("Timestep", fontsize=12)
            ax.set_ylabel("Cumulative Reward", fontsize=12)
            ax.set_title("Cumulative Reward Comparison", fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path_cumulative = os.path.join(self.logdir, "reward_plot_cumulative.png")
            plt.savefig(plot_path_cumulative, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Cumulative reward plot saved to {plot_path_cumulative}")

            # # Animated comparison plot
            # fig, ax = plt.subplots(figsize=(12, 6))
            # max_len = max(len(rewards_dss), len(rewards_sapo))
            # all_rewards = rewards_dss + rewards_sapo
            # ax.set_xlim(0, max_len)
            # ax.set_ylim(min(all_rewards) - 0.1, max(all_rewards) + 0.1)
            # ax.set_xlabel("Timestep", fontsize=12)
            # ax.set_ylabel("Reward", fontsize=12)
            # ax.set_title("DSS MPC vs SAPO Performance Comparison", fontsize=14, fontweight='bold')
            # ax.grid(True, alpha=0.3)

            # line_dss, = ax.plot([], [], marker='x', color='blue', linewidth=1, markersize=3, label="DSS MPC", alpha=0.8)
            # line_sapo, = ax.plot([], [], marker='x', color='red', linewidth=1, markersize=3, label="SAPO", alpha=0.8)
            # ax.legend(fontsize=11)

            # xdata_dss, ydata_dss = [], []
            # xdata_sapo, ydata_sapo = [], []

            # def init() -> Tuple[Any]:
            #     line_dss.set_data([], [])
            #     line_sapo.set_data([], [])
            #     return (line_dss, line_sapo)

            # def update(frame: int) -> Tuple[Any]:
            #     if frame < len(rewards_dss):
            #         xdata_dss.append(frame)
            #         ydata_dss.append(rewards_dss[frame])
            #         line_dss.set_data(xdata_dss, ydata_dss)
            #     if frame < len(rewards_sapo):
            #         xdata_sapo.append(frame)
            #         ydata_sapo.append(rewards_sapo[frame])
            #         line_sapo.set_data(xdata_sapo, ydata_sapo)
            #     return (line_dss, line_sapo)

            # ani = animation.FuncAnimation(fig, update, frames=max_len, init_func=init, blit=True, interval=100, repeat=False)
            # anim_path = os.path.join(self.logdir, "reward_animation_comparison.gif")
            # ani.save(anim_path, writer="pillow", fps=30)
            # plt.close()
            # print(f"✓ Comparison animation saved to {anim_path}")
        else:
            # DSS-only individual plot
            if rewards_dss:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(rewards_dss, marker='x', color='blue', linewidth=1, markersize=3, alpha=0.8)
                ax.set_xlabel("Timestep", fontsize=12)
                ax.set_ylabel("Reward", fontsize=12)
                ax.set_title(f"DSS MPC Performance\nTotal: {total_dss:.2f} | Avg: {total_dss/len(rewards_dss):.2f}", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
                ax.set_xlim(0, len(rewards_dss) - 1)
                if len(rewards_dss) > 0:
                    min_y = min(0, min(rewards_dss) - 0.1)
                    max_y = max(rewards_dss) + 0.1
                    ax.set_ylim(min_y, max_y)
                plt.tight_layout()
                plot_path_individual = os.path.join(self.logdir, "reward_plot_individual.png")
                plt.savefig(plot_path_individual, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Individual plot saved to {plot_path_individual}")

                # DSS-only cumulative reward plot
                fig, ax = plt.subplots(figsize=(12, 6))
                cumulative_dss = np.cumsum(rewards_dss)
                ax.plot(cumulative_dss, marker='x', color='blue', linewidth=1, markersize=3, label='DSS MPC (Cumulative)', alpha=0.8)
                ax.set_xlabel("Timestep", fontsize=12)
                ax.set_ylabel("Cumulative Reward", fontsize=12)
                ax.set_title("DSS MPC Cumulative Reward", fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plot_path_cumulative = os.path.join(self.logdir, "reward_plot_cumulative.png")
                plt.savefig(plot_path_cumulative, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Cumulative reward plot saved to {plot_path_cumulative}")

                # # DSS-only animated plot
                # fig, ax = plt.subplots(figsize=(12, 6))
                # max_len = len(rewards_dss)
                # if len(rewards_dss) > 0:
                #     ax.set_xlim(0, max_len)
                #     ax.set_ylim(min(rewards_dss) - 0.1, max(rewards_dss) + 0.1)
                # ax.set_xlabel("Timestep", fontsize=12)
                # ax.set_ylabel("Reward", fontsize=12)
                # ax.set_title("DSS MPC Performance", fontsize=14, fontweight='bold')
                # ax.grid(True, alpha=0.3)

                # line_dss, = ax.plot([], [], marker='x', color='blue', linewidth=1, markersize=3, label="DSS MPC", alpha=0.8)
                # ax.legend(fontsize=11)

                # xdata_dss, ydata_dss = [], []

                # def init() -> Tuple[Any]:
                #     line_dss.set_data([], [])
                #     return (line_dss,)

                # def update(frame: int) -> Tuple[Any]:
                #     if frame < len(rewards_dss):
                #         xdata_dss.append(frame)
                #         ydata_dss.append(rewards_dss[frame])
                #         line_dss.set_data(xdata_dss, ydata_dss)
                #     return (line_dss,)

                # ani = animation.FuncAnimation(fig, update, frames=max_len, init_func=init, blit=True, interval=100, repeat=False)
                # anim_path = os.path.join(self.logdir, "reward_animation_comparison.gif")
                # ani.save(anim_path, writer="pillow", fps=30)
                # plt.close()
                # print(f"✓ Animation saved to {anim_path}")

    def eval(self) -> None:
        if self.render_results:
            # Load data from logdir and create comparison plots
            scores_path = os.path.join(self.logdir, "my_scores.json")
            if not os.path.exists(scores_path):
                print(f"Error: Scores file not found at {scores_path}")
                print("Please run evaluation first to generate scores.json")
                return

            print(f"Loading scores from {scores_path}")
            with open(scores_path, 'r') as f:
                scores = json.load(f)

            # Extract reward data
            dss_rewards = scores.get("dss_mpc", {}).get("rewards_per_timestep", [])
            sapo_rewards = scores.get("sapo", {}).get("rewards_per_timestep", [])
            total_dss = scores.get("dss_mpc", {}).get("total_reward", 0.0)
            total_sapo = scores.get("sapo", {}).get("total_reward", 0.0)

            if not dss_rewards and not sapo_rewards:
                print("Error: No reward data found in scores.json")
                return

            print(f"Loaded data: DSS ({len(dss_rewards)} timesteps), SAPO ({len(sapo_rewards)} timesteps)")
            print(f"DSS Total Reward: {total_dss:.4f}, SAPO Total Reward: {total_sapo:.4f}")

            # Create comparison plots
            self._create_comparison_plots(dss_rewards, sapo_rewards, total_dss, total_sapo)
        else:
            start_time = time.time()
            # self.run_dms_mpc()
            # Use single shooting MPC (simpler, no constraint penalties)
            self.run_single_shooting_mpc()
            end_time = time.time()
            print(f"\nTotal time: {end_time - start_time:.2f}s")

    def replay_trajectory(self) -> None:
        """Replay saved trajectory."""
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)

        rewards = []
        actions_to_replay = torch.load('trajectory_single_shooting.pt')

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
