import copy
import glob
import os
import time

import colorednoise
import torch
import warp as wp

from mineral.agents.agent import Agent


class CEMMPCAgent(Agent):
    r"""Cross Entropy Method Model Predictive Control using iCEM with colored noise and warm-starting of elite trajectories.

    MPC is performed by:
    1) Sampling action sequences (trajectories) using CEM + colored noise.
    2) Rolling out the trajectories in parallel in a batched simulation.
    3) Selecting elite trajectories and using the elites to update the mean/std of the sampling distribution.
    4) Executing only the first action of the best trajectory at each timestep (MPC).

    Attributes:
        H (int): MPC planning horizon.
        N (int): Number of sampled trajectories.
        K (int): Elite-set size.
        iterations (int): Number of CEM iterations per planning step.
        timesteps (int): Total number of MPC timesteps executed.
        beta (float): Exponent for colored-noise temporal correlation.
        keep_elite_fraction (float): Fraction of previous elites retained.
        alpha (float): Momentum coefficient for smoothing mean/std across iterations.
        initial_std (float): Initial standard deviation of sampled actions.
        previous_elite_actions (torch.Tensor | None): Stored elite trajectories.
        previous_mean (torch.Tensor | None): Mean action sequence from previous time step.
    """
    def __init__(self, full_cfg, **kwargs):
        """Load configuration parameters.

        Args:
            full_cfg: Hydra config containing agent, network, and CEM/MPC settings.
            **kwargs: Passed through to the parent Agent class.

        Outputs:
            - Initializes MPC parameters (H, N, K, iterations, ...).
            - Allocates buffers for warm-starting (previous_mean, previous_elite_actions).
            - Initializes internal environment state placeholders (`obs`, `dones`).
        """
        self.network_config = full_cfg.agent.network
        self.params = full_cfg.agent.params
        self.num_actors = self.params.num_actors
        self.max_agent_steps = int(self.params.max_agent_steps)
        self.render_results = self.params.render_results
        self.seed = self.params.seed

        # Collect all CEM MPC Parameters
        self.cem_mpc_params = full_cfg.agent.cem_mpc_params
        self.H = self.cem_mpc_params.H
        self.N = self.cem_mpc_params.N
        self.K = self.cem_mpc_params.K
        self.iterations = self.cem_mpc_params.iterations
        self.timesteps = self.cem_mpc_params.timesteps
        self.beta = self.cem_mpc_params.beta
        self.keep_elite_fraction = self.cem_mpc_params.keep_elite_fraction
        self.alpha = self.cem_mpc_params.alpha
        self.initial_std = self.cem_mpc_params.initial_std

        super().__init__(full_cfg, **kwargs)

        self.obs = None
        self.dones = None

        # Warm start
        self.previous_elite_actions = None
        self.previous_mean = None

    def clone_state(self, state):
        """Deep-copy a Warp simulation state.

        Args:
        state: Environment sim state containing Warp tensors and Python attributes.

        Returns:
        A new cloned simulation state where:
            - Warp tensors are cloned using wp.clone().
            - Python objects are deep-copied.
        """
        # Create a new empty State object of the same type
        s = type(state)()

        # Attributes to clone with wp.clone
        wp_clone_attrs = [
            "body_f", "body_q", "body_qd", "joint_q", "joint_qd", "mpm_C", "mpm_F", "mpm_F_trial",
            "mpm_grid_m", "mpm_grid_mv", "mpm_grid_v", "mpm_stress", "mpm_x", "mpm_v"
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
        """Entry point called by training/evaluation loop.

        Behavior:
            - If render_results is True -> replay stored trajectories and save as reward file.
            - Otherwise -> run CEM MPC evaluation over one or multiple N values.

        Output:
            - Runs evaluation or replay depending on config.
        """
        if self.render_results:
            self.replay_trajectory()

        else:
            # Specify N values to evaluate
            # Example: N_values = [64, 128, 256] or single N_values = [128]
            # self.eval_multiple([64, 128, 256])
            self.eval_multiple([self.N])

    def generate_action_sequences(self, mean, std, elite_actions, iteration=0):
        """Generate action trajectories using Gaussian sampling + colored noise (iCEM).

        Args:
            mean (torch.Tensor): Mean action trajectory of shape (H, action_dim).
            std (torch.Tensor): Standard deviation trajectory of shape (H, action_dim).
            elite_actions (torch.Tensor | None): Previously elite trajectories (warm start).
            iteration (int): Current CEM iteration index.

        Returns:
            torch.Tensor: Sampled action sequences with shape (N, H, action_dim).
        """
        mean_exp = mean.unsqueeze(0).expand(self.N, -1, -1)     # Shape: (N, H, action_dim)
        std_exp = std.unsqueeze(0).expand(self.N, -1, -1)       # Shape: (N, H, action_dim)

        # Generate colored noise (N, action_dim, H) --> transpose to (N, H, action_dim)
        # Colored noise generated should be temporally correlated along the axis of H
        noise = colorednoise.powerlaw_psd_gaussian(
            self.beta, size=(self.N, self.action_dim, self.H)
        ).transpose(0, 2, 1)
        noise = torch.from_numpy(noise).cuda().float()          # Shape: (N, H, action_dim)

        # Scale the distribution by noise * std
        action_sequences = mean_exp + noise * std_exp           # Shape: (N, H, action_dim)

        # [iCEM 3.3] - Clipping at the action boundaries (clip)
        # Clip actions to range of [-1, 1]
        env_action_low = torch.from_numpy(self.env.action_space.low).float().to(self.device)    # low:  -1
        env_action_high = torch.from_numpy(self.env.action_space.high).float().to(self.device)  # high:  1
        action_sequences = torch.clip(action_sequences, env_action_low, env_action_high)

        # [iCEM 3.2] - CEM with memory
        # Keep a fraction of the elites and shift elites every iteration
        if (iteration == 0) and elite_actions is not None:
            num_elite_to_add = int(self.keep_elite_fraction * self.K)
            shifted_elite = torch.roll(elite_actions[:num_elite_to_add], -1, dims=1)
            action_sequences[:num_elite_to_add] = shifted_elite

        elif elite_actions is not None:
            num_elite_to_add = int(self.keep_elite_fraction * self.K)
            action_sequences[:num_elite_to_add] = elite_actions[:num_elite_to_add]

        if iteration == self.iterations - 1:
            action_sequences[-1] = mean

        return action_sequences

    def evaluate_action_sequence_batch(self, init_state, action_sequences):
        """Roll out sampled action trajectories in a batched simulation  in parallel by copying the current state.

        Args:
            init_state: Initial environment state (cloned before rollouts).
            action_sequences (torch.Tensor): Action sequences with shape (N, H, action_dim).

        Returns:
            torch.Tensor: Total reward for each trajectory of shape (N,).
        """
        # Initialise rewards tensor with shape (N,)
        batch_size = self.num_actors
        rewards = torch.zeros(self.N, device=self.device)

        # Batch the rewards based on the amount of samples
        for start_idx in range(0, self.N, batch_size):
            end_idx = min(start_idx + batch_size, self.N)
            batch_sequences = action_sequences[start_idx:end_idx]
            batch_N = batch_sequences.shape[0]

            # Copy the current state of the env to perform actions
            self.env.state_0 = self.clone_state(init_state)
            batch_rewards = torch.zeros(batch_N, device=self.device)

            for h in range(self.H):
                actions_h = batch_sequences[:, h, :]   # Shape: (batch_N, action_dim)

                # Pad actions to match num_actors if needed
                if batch_N < self.num_actors:
                    # Repeat the last action or pad with zeros
                    padded_actions = torch.zeros(self.num_actors, self.action_dim, device=self.device)
                    padded_actions[:batch_N] = actions_h
                    actions_h = padded_actions

                _, r ,_, _ = self.env.step(actions_h)
                batch_rewards += r[:batch_N]

            rewards[start_idx:end_idx] = batch_rewards

        return rewards

    def cem_plan(self, init_state, previous_elite_actions=None):
        """Perform one full CEM optimization loop to find the best action sequence.

        Args:
            init_state: Simulation state at beginning of planning step.
            previous_elite_actions (torch.Tensor | None): Warm-start elite trajectories.

        Returns:
            (tuple):
                torch.Tensor: First action of best sequence (shape: action_dim).
                torch.Tensor: Elite action trajectories (shape: K, H, action_dim).
        """
        # # TESTING: No warm start of mean
        # mean = torch.zeros((self.H, self.action_dim), device=self.device)                       # Mean
        # std = torch.ones((self.H, self.action_dim), device=self.device) * self.initial_std      # Standard deviation

        # Warm start of mean
        if self.previous_mean is not None:
            # Shift mean one step forward (discard the executed action)
            shifted_mean = torch.zeros_like(self.previous_mean)
            shifted_mean[:-1] = self.previous_mean[1:]               # Shift forward
            shifted_mean[-1] = torch.zeros_like(shifted_mean[-1])    # Fresh last step
            mean = shifted_mean.clone()
        else:
            # Initialize from scratch if no previous mean
            mean = torch.zeros((self.H, self.action_dim), device=self.device)

        std = torch.ones((self.H, self.action_dim), device=self.device) * self.initial_std

        prev_mean = mean.clone()
        prev_std = std.clone()

        # Use previous elite actions from last timestep for first iteration
        elite_actions = previous_elite_actions

        for iteration in range(self.iterations):
            # Generation action sequences and allocate rewards for each action sequence
            action_sequences = self.generate_action_sequences(mean, std, elite_actions, iteration)
            rewards = self.evaluate_action_sequence_batch(init_state, action_sequences)

            # Sort the rewards array and pick only the top K elements
            _, elite_idxs = torch.topk(rewards, self.K)
            elite_actions = action_sequences[elite_idxs]

            # [iCEM 3.3] - Executing the best action (best-a)
            # Identify the best trajectory among elites (first element in sorted elites)
            best_action_sequence = action_sequences[elite_idxs[0]]

            # Update the values of mu and sigma after iteration
            new_mean = elite_actions.mean(dim=0)
            new_std = elite_actions.std(dim=0) + 1e-3   # Stability

            # Momentum update
            mean = (self.alpha * prev_mean) + ((1 - self.alpha) * new_mean)
            std = (self.alpha * prev_std) + ((1 - self.alpha) * new_std)

            # Update values for next iteration
            prev_mean = mean.clone()
            prev_std = std.clone()

        # Store the mean of the best action sequence for warm starting next timestep
        self.previous_mean = mean.detach().clone()

        # Return the first action of the best sequence
        return best_action_sequence[0], elite_actions

    def run_cem_mpc(self, save_name="trajectory.pt"):
        """Run MPC loop: at each time step, perform CEM planning and execute the best action.

        Args:
            save_name (str): Filename to save executed best action trajectory.

        Output:
            - Writes trajectory to disk.
            - Prints time step, best action and step reward.
        """
        # Initialise environment
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)
        total_reward = 0.0

        # Create list to save actions
        best_actions_list = []

        # Main loop
        print(f"\n===== Running evaluation with N={self.N} =====")
        for timestep in range(self.timesteps):
            if self.dones[0]:
                break  # Stop if real environment is done

            # Save the initial state
            init_state = self.clone_state(self.env.state_0)

            # Evaluate new best action
            best_action, elite_actions = self.cem_plan(init_state, self.previous_elite_actions)
            actions = best_action.unsqueeze(0).repeat(self.num_actors, 1)   # Shape: (64, action_dim)

            # Store elite actions for next timestep
            self.previous_elite_actions = elite_actions

            # Reset state of all environments
            self.env.state_0 = init_state

            # Step the environment forward
            obs, reward, done, _ = self.env.step(actions)
            self.obs = self._convert_obs(obs)
            self.dones = done
            total_reward += reward[0]

            # Append actions to best action list
            best_actions_list.append(best_action.clone())

            print(f"Timestep {timestep + 1} | Action: {best_action} | Reward: {reward[0]:.3f}")
            # tqdm.write(f"Timestep {timestep + 1} | Action: {best_action} | Reward: {reward[0]:.3f}")

        print("Evaluation complete")
        print(f"Total reward: {total_reward:.3f}")

        # Save trajectory as PyTorch file
        best_actions_tensor = torch.stack(best_actions_list)

        torch.save(best_actions_tensor, save_name)
        print(f"Trajectory {self.N} saved")

    def eval_multiple(self, N_values):
        """Run multiple evaluations by sweeping different sample sizes N.

        Args:
            N_values (list[int]): Values of N to test and save.
        """
        for N in N_values:
            # Override N and reset previous_elite_actions
            self.N = N
            self.previous_elite_actions = None
            self.previous_mean = None

            start_time = time.time()
            self.run_cem_mpc(save_name=f"trajectory_{N}_seed_{self.seed}_std_{self.initial_std}.pt")
            end_time = time.time()
            print(f"Time taken: {end_time - start_time}")


    def replay_trajectory(self):
        """Replay previously saved action trajectories on disk and log rewards.

        Searches and loads matching files in the following format:
            trajectory_*_seed_{seed}_std_{initial_std}.pt

        Outputs:
            - Runs env in real time.
            - Saves reward logs to disk.
        """
        # Initialise environment
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)

        # Collect all trajectory files
        trajectory_files = glob.glob(f"trajectory_*_seed_{self.seed}_std_{self.initial_std}.pt")
        for file in trajectory_files:
            actions_to_replay = torch.load(file)

            rewards = []
            timestep = 0

            for action in actions_to_replay:
                obs, reward, done, _ = self.env.step(action)
                self.obs = self._convert_obs(obs)
                self.dones = done

                print(f"Time step {timestep + 1} | Action: {action} | Reward: {reward[0]:.3f}")
                timestep += 1
                rewards.append(reward[0].item())
            print("Trajectory completed")

            # Save rewards tensor as pytorch file
            base_name = os.path.basename(file)
            idx = base_name.replace("trajectory_", "").replace(".pt", "")
            reward_filename = f"reward_{idx}.pt"

            torch.save(rewards, reward_filename)