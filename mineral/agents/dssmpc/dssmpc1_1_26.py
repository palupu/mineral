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
        self.shac_ckpt_path = self.dss_mpc_params.get('shac_ckpt_path', None)
        self.shac_agent = None

        super().__init__(full_cfg, **kwargs)

        # Get control dimension from parent Agent class
        self.control_dim = self.action_dim

        self.obs = None
        self.dones = None

        # Single shooting: warm start control sequence
        self._u_warm_start = None

        # Load SHAC agent (hardcoded paths)
        self._load_shac_agent()

    def _load_shac_agent(self) -> None:
        """Load SHAC agent from checkpoint for generating initial control sequences."""
        # Hardcoded paths
        shac_ckpt_path = "/app/workdir/RewarpedRollingPin4M-SAPO/20251216-041123.97/ckpt/best_rewards15.15.pth"
        config_path = "/app/workdir/RewarpedRollingPin4M-SAPO/20251216-041123.97/resolved_config.yaml"

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
            if self.shac_agent.obs_rms is not None:
                current_obs = {k: self.shac_agent.obs_rms[k].normalize(v) for k, v in obs.items()}

            # Roll out policy for horizon steps through actual environment
            u_init = []
            current_obs_dict = current_obs

            prev_obs_dict = None
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
                action = self.shac_agent.get_actions(current_obs_dict, sample=False)


                action_single = action[0]  # Take first actor's action
                print(f"Step {step}: Action = [{action_single[0].item():.4f}, {action_single[1].item():.4f}, {action_single[2].item():.4f}]")
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
                else:
                    current_obs_dict = next_obs

                # Save current observation for comparison in next iteration
                prev_obs_dict = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in current_obs_dict.items()}

                # If episode terminates early, repeat last action for remaining steps
                if done[0]:
                    # Fill remaining steps with last action
                    last_action = u_init[-1]
                    for _ in range(horizon - step - 1):
                        u_init.append(last_action.clone())
                    break

            u_init = torch.stack(u_init)  # Shape: (N, control_dim)

            # Print the generated action sequence
            print(f"\nGenerated SHAC initial action sequence (shape: {u_init.shape}):")
            for i, action in enumerate(u_init):
                print(f"  u[{i}]: [{action[0].item():.4f}, {action[1].item():.4f}, {action[2].item():.4f}]")
            print()


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
        # Restore environment state from checkpoint and reinitialize gradient tape
        self.env.clear_grad(checkpoint)
        # self.env.state_0 = self.clone_state(checkpoint)

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
            print(f"  Using SHAC initial sequence (mean norm: {u_seq.norm(dim=-1).mean().item():.4f})")
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

        self.env.clear_grad(checkpoint)

        print(f"\n{'=' * 70}")
        print("Starting Single Shooting Optimization (PyTorch Adam)")
        print(f"  Horizon: N={self.N}")
        print(f"  Control dim: {self.control_dim}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Max iterations: {self.max_iter}")
        print(f"{'=' * 70}\n")

        # Create optimizer
        u_param = torch.nn.Parameter(u_seq)
        optimizer = torch.optim.Adam([u_param], lr=self.learning_rate)

        start_time = time.time()

        for iteration in range(self.max_iter):
            optimizer.zero_grad()

            # Compute loss (negative cumulative reward)
            # This restores from checkpoint and rolls out with gradient tracking
            debug = False # (iteration == 0)  # Only print debug info on first iteration
            loss = self.single_shooting_loss(u_param, checkpoint, debug=debug)

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

            # Optimizer step
            optimizer.step()

            # Clip controls to valid range [-1, 1]
            # with torch.no_grad():
            #     u_param.data.clamp_(-1.0, 1.0)

            # Print progress
            # if iteration % 10 == 0 or iteration == self.max_iter - 1:
            print(f"Iteration {iteration:4d}/{self.max_iter} | Loss: {loss.item():.6f} | "
                    f"Reward: {-loss.item():.6f} | Grad: {grad_norm:.4e}", flush=True)

        elapsed_time = time.time() - start_time

        # Restore environment gradient setting
        self.env.no_grad = original_no_grad

        u_opt = u_param.data.clone().detach()

        # Restore original state
        # self.env.state_0 = original_state
        # Restore state from checkpoint for actual execution
        self.env.clear_grad(checkpoint)
        # self.env.state_0 = self.clone_state(checkpoint)

        print(f"\n{'=' * 70}")
        print("Single Shooting Optimization Complete")
        # print(f"  Final Loss: {best_loss:.6f} (Reward: {-best_loss:.6f})")
        print(f"  Final Loss: {loss.item():.6f} (Reward: {-loss.item():.6f})")
        print(f"  Time: {elapsed_time:.2f}s")
        print("  Optimal controls:")
        for i, u in enumerate(u_opt):
            print(f"    u[{i}]: [{u[0].item():.4f}, {u[1].item():.4f}, {u[2].item():.4f}]")
        print(f"{'=' * 70}\n")

        return u_opt

    def run_single_shooting_mpc(self) -> None:
        """Main MPC loop using Single Shooting.

        Following the pseudo code:
            for timesteps:
                u = single_shooting(xk, u_next)
                env.step(u[0])  # execute first control
                u_next = [u1, ..., u_{N-1}, u_{N-1}]  # warm start shift
        """
        # IMPORTANT: Enable gradient computation BEFORE reset
        # so that control_tensors are properly initialized for differentiation
        self.env.no_grad = False

        # Initialize environment (with gradients enabled)
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)
        total_reward = 0.0

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

        # List to save executed actions
        executed_actions = []

        print(f"\n{'#' * 70}")
        print(f"{'SINGLE SHOOTING MPC (PyTorch Adam)':^70}")
        print(f"{'#' * 70}")
        print(f"Horizon: N={self.N} steps")
        print(f"Control dim: {self.control_dim}")
        print(f"Optimizer: Adam (lr={self.learning_rate})")
        print(f"Timesteps: {self.timesteps}")
        print(f"{'#' * 70}\n")

        for timestep in range(self.timesteps):
            if self.dones[0]:
                break

            print(f"\n{'─' * 70}")
            print(f"Timestep {timestep + 1}/{self.timesteps}")
            print(f"{'─' * 70}")

            # Disable renderer during optimization
            saved_renderer = self.env.renderer
            self.env.renderer = None
            # checkpoint = self.env.get_checkpoint(detach=True)

            # Plan using single shooting (uses checkpoints internally)
            # Pass current observation for SHAC initial guess
            u_opt = self.single_shooting_plan(u_next, obs=self.obs)

            # nur sapo laufen
            # total reward vergelichen. seed gleich setzen

            # self.env.clear_grad(checkpoint)
            # Re-enable renderer
            self.env.renderer = saved_renderer

            # Execute first control (state already restored by single_shooting_plan)
            u_0 = u_opt[0]
            actions = u_0.unsqueeze(0).repeat(self.num_actors, 1)
            obs, reward, done, _ = self.env.step(actions)
            self.obs = self._convert_obs(obs)
            self.dones = done

            reward_val = reward[0].item() if isinstance(reward, torch.Tensor) else reward[0]
            total_reward += reward_val

            # Save executed action
            executed_actions.append(u_0.clone())

            print(f"Executed action: [{u_0[0].item():.4f}, {u_0[1].item():.4f}, {u_0[2].item():.4f}]")
            print(f"Reward: {reward_val:.4f} | Cumulative: {total_reward:.4f}")

            # Warm start: shift controls [u1, ..., u_{N-1}, u_{N-1}]
            u_next = torch.zeros_like(u_opt)
            u_next[:-1] = u_opt[1:]  # Shift left
            u_next[-1] = u_opt[-1]   # Repeat last control

        print(f"\n{'#' * 70}")
        print(f"{'SINGLE SHOOTING MPC COMPLETE':^70}")
        print(f"{'#' * 70}")
        print(f"Total reward: {total_reward:.4f}")
        print(f"Average reward per step: {total_reward / max(len(executed_actions), 1):.4f}")
        print(f"{'#' * 70}\n")

        # Save trajectory
        if executed_actions:
            actions_tensor = torch.stack(executed_actions)
            torch.save(actions_tensor, 'trajectory_single_shooting.pt')
            print("✓ Trajectory saved to trajectory_single_shooting.pt")

    def eval(self) -> None:
        if self.render_results:
            self.replay_trajectory()
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
