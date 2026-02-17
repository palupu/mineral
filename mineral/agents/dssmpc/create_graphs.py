#!/usr/bin/env python3
"""Create graphs from DSSMPC hyperparameter sweep results.

This script generates plots showing how total_reward varies across different
hyperparameter values for each swept parameter.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Always export plots as SVG (vector graphics).
def _as_svg_path(output_path) -> Path:
    """Normalize output path to an .svg Path."""
    p = output_path if isinstance(output_path, Path) else Path(output_path)
    return p.with_suffix(".svg")

# Flag to control whether to use confidence intervals or standard deviation
# Set to True for 95% confidence intervals, False for standard deviation
USE_CONFIDENCE_INTERVALS = True

# Base configuration (same as in run_multiple_dssmpc.py)
BASE_CONFIG = {
    "n": 5,
    "max_iter": 30,
    "lr": 0.01,
    "sample": False,
}
# BASE_CONFIG = {
#     "n": 4,
#     "max_iter": 20,
#     "lr": 0.01,
#     "sample": False,
# }

# Parameter values to test (same as in run_multiple_dssmpc.py)
# N_VALUES = [1, 3, 5, 9]
# MAX_ITER_VALUES = [10, 30, 50]
# LR_VALUES = [0.05, 0.01, 0.005, 0.001]
# SAMPLE_VALUES = [False, True]
# SEED_VALUES = [1, 2, 3]
N_VALUES = [1,2,3,4, 5, 7, 9 ,12, 15]
MAX_ITER_VALUES = [30, 50, 70]
LR_VALUES = [0.01, 0.02, 0.03]
SAMPLE_VALUES = [False]
SEED_VALUES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13] # [1, 2, 3]

# Hardcoded SAPO data paths
# Dictionary mapping seed to SAPO path
SAPO_PATHS_BY_SEED = {
    # 1: Path("workdir/DSSMPC/2026_01_09-18_56_00/N1_iter1_lr0.01_samplefalse_seed1/my_scores.json"),
    # 2: Path("workdir/DSSMPC/2026_01_09-18_56_36/N1_iter1_lr0.01_samplefalse_seed2/my_scores.json"),
    # 3: Path("workdir/DSSMPC/2026_01_09-18_57_07/N1_iter1_lr0.01_samplefalse_seed3/my_scores.json"),
    4: Path("workdir/DSSMPC/2026_01_09-20_19_15/N4_iter20_lr0.001_samplefalse_seed4/my_scores.json"),
    5: Path("workdir/DSSMPC/2026_01_09-20_19_15/N4_iter20_lr0.001_samplefalse_seed5/my_scores.json"),
    6: Path("workdir/DSSMPC/2026_01_09-20_19_15/N4_iter20_lr0.001_samplefalse_seed6/my_scores.json"),
    7: Path("workdir/DSSMPC/2026_01_10-17_07_50/N4_iter30_lr0.01_samplefalse_seed7/my_scores.json"),
    8: Path("workdir/DSSMPC/2026_01_10-17_07_50/N4_iter30_lr0.01_samplefalse_seed8/my_scores.json"),
    9: Path("workdir/DSSMPC/2026_01_11-18_11_55/N4_iter30_lr0.01_samplefalse_seed9/my_scores.json"),
    10: Path("workdir/DSSMPC/2026_01_11-18_11_55/N4_iter30_lr0.01_samplefalse_seed10/my_scores.json"),
    11: Path("workdir/DSSMPC/2026_01_11-18_11_55/N4_iter30_lr0.01_samplefalse_seed11/my_scores.json"),
    12: Path("workdir/DSSMPC/2026_01_11-18_11_55/N4_iter30_lr0.01_samplefalse_seed12/my_scores.json"),
    13: Path("workdir/DSSMPC/2026_01_11-18_11_55/N4_iter30_lr0.01_samplefalse_seed13/my_scores.json"),

}

# List of all SAPO paths (for load_sapo_reference)
SAPO_PATHS = list(SAPO_PATHS_BY_SEED.values())

# Consistent color and style definitions for parameters
# Colors are visually distinct and consistent across all plots
PARAM_COLORS = {
    "N": "blue",  # Blue
    "max_iter": "green",  # Green
    "lr": "red",  # Orange
    "sample": "yellow",  # Purple
}

# Line styles for different parameter values (to make them more distinct)
LINE_STYLES = ['-', '--', '-.', ':']

# Color palette for different config values (when multiple values of same param)
CONFIG_COLORS = [
    "blue",
    "red",
    "green",
    "yellow",
    "purple",
    "orange",
    "cyan",
    "magenta",
    "brown",
    "gray",
    "olive",
    "pink",
    "teal",
    "gold",
]


def generate_configurations():
    """Generate all configurations using the same logic as run_multiple_dssmpc.py."""
    configurations = []

    # Vary N (keep others at base)
    for n in N_VALUES:
        for seed in SEED_VALUES:
            configurations.append({
                "n": n,
                "max_iter": BASE_CONFIG["max_iter"],
                "lr": BASE_CONFIG["lr"],
                "sample": BASE_CONFIG["sample"],
                "seed": seed,
                "varied_param": "N",
            })

    # Vary max_iter (keep others at base)
    for max_iter in MAX_ITER_VALUES:
        for seed in SEED_VALUES:
            configurations.append({
                "n": BASE_CONFIG["n"],
                "max_iter": max_iter,
                "lr": BASE_CONFIG["lr"],
                "sample": BASE_CONFIG["sample"],
                "seed": seed,
                "varied_param": "max_iter",
            })

    # Vary lr (keep others at base)
    for lr in LR_VALUES:
        for seed in SEED_VALUES:
            configurations.append({
                "n": BASE_CONFIG["n"],
                "max_iter": BASE_CONFIG["max_iter"],
                "lr": lr,
                "sample": BASE_CONFIG["sample"],
                "seed": seed,
                "varied_param": "lr",
            })

    # # Vary sample (keep others at base)
    # for sample in SAMPLE_VALUES:
    #     for seed in SEED_VALUES:
    #         configurations.append({
    #             "n": BASE_CONFIG["n"],
    #             "max_iter": BASE_CONFIG["max_iter"],
    #             "lr": BASE_CONFIG["lr"],
    #             "sample": sample,
    #             "seed": seed,
    #             "varied_param": "sample",
    #         })

    return configurations


def get_folder_name(n, max_iter, lr, sample, seed):
    """Generate folder name from configuration parameters."""
    sample_str = str(sample).lower()
    return f"N{n}_iter{max_iter}_lr{lr}_sample{sample_str}_seed{seed}"


def get_param_value(config, varied_param):
    """Get parameter value from config given varied_param name (handles N->n mapping)."""
    param_key_map = {"N": "n", "max_iter": "max_iter", "lr": "lr", "sample": "sample"}
    config_key = param_key_map.get(varied_param, varied_param)
    return config[config_key]


def calculate_center_and_bounds(values, axis=None):
    """Return a center line and asymmetric bounds for plotting.

    - If `USE_CONFIDENCE_INTERVALS` is True:
      - center: median (50th percentile)
      - lower/upper: 2.5th and 97.5th percentiles (95% CI)
    - Else:
      - center: mean
      - lower/upper: mean ± std

    Args:
        values: Array-like or ndarray.
        axis: Axis along which to aggregate (e.g., axis=0 for multiple time series).

    Returns:
        (center, lower, upper) with shapes consistent with the reduction over `axis`.
    """
    if USE_CONFIDENCE_INTERVALS:
        lower = np.percentile(values, 2.5, axis=axis)
        center = np.percentile(values, 50, axis=axis)
        upper = np.percentile(values, 97.5, axis=axis)
        return center, lower, upper

    # Standard deviation band around the mean
    center = np.mean(values, axis=axis)
    if axis is None:
        n = np.size(values)
        ddof = 1 if n > 1 else 0
    else:
        ddof = 1
    std = np.std(values, axis=axis, ddof=ddof)
    return center, center - std, center + std


def find_scores_file(base_dirs, folder_name):
    """Find my_scores.json file in one of the base directories."""
    if isinstance(base_dirs, (str, Path)):
        base_dirs = [base_dirs]
    
    for base_dir in base_dirs:
        base_dir = Path(base_dir)
        scores_path = base_dir / folder_name / "my_scores.json"
        if scores_path.exists():
            return scores_path
    
    return None


def load_total_reward(base_dirs, config):
    """Load total_reward from my_scores.json for a given configuration.
    
    Args:
        base_dirs: Single base directory (Path/str) or list of base directories
        config: Configuration dictionary
    """
    folder_name = get_folder_name(
        config["n"], config["max_iter"], config["lr"],
        config["sample"], config["seed"]
    )
    
    scores_path = find_scores_file(base_dirs, folder_name)
    if scores_path is None:
        print(f"Warning: {folder_name}/my_scores.json not found in any base directory")
        return None

    try:
        with open(scores_path, "r") as f:
            data = json.load(f)
            return data["dss_mpc"]["total_reward"]
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Warning: Error reading {scores_path}: {e}")
        return None


def load_sapo_reference():
    """Load SAPO total_reward from hardcoded paths and calculate mean/std."""
    sapo_rewards = []
    for sapo_path in SAPO_PATHS:
        if not sapo_path.exists():
            print(f"Warning: SAPO path {sapo_path} not found")
            continue

        try:
            with open(sapo_path, "r") as f:
                data = json.load(f)
                if "sapo" in data and "total_reward" in data["sapo"]:
                    sapo_rewards.append(data["sapo"]["total_reward"])
                else:
                    print(f"Warning: SAPO data not found in {sapo_path}")
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Warning: Error reading SAPO data from {sapo_path}: {e}")

    if not sapo_rewards:
        return None, None, None

    sapo_center, sapo_lower, sapo_upper = calculate_center_and_bounds(sapo_rewards)
    return sapo_center, sapo_lower, sapo_upper


def load_rewards_per_timestep(base_dirs, config):
    """Load rewards_per_timestep from my_scores.json for a given configuration.
    
    Args:
        base_dirs: Single base directory (Path/str) or list of base directories
        config: Configuration dictionary
    """
    folder_name = get_folder_name(
        config["n"], config["max_iter"], config["lr"],
        config["sample"], config["seed"]
    )
    
    scores_path = find_scores_file(base_dirs, folder_name)
    if scores_path is None:
        return None

    try:
        with open(scores_path, "r") as f:
            data = json.load(f)
            if "dss_mpc" in data and "rewards_per_timestep" in data["dss_mpc"]:
                return data["dss_mpc"]["rewards_per_timestep"]
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Warning: Error reading {scores_path}: {e}")

    return None


def load_sapo_rewards_per_timestep(seed):
    """Load SAPO rewards_per_timestep from hardcoded path for a specific seed."""
    sapo_path = SAPO_PATHS_BY_SEED.get(seed)
    if not sapo_path or not sapo_path.exists():
        print(f"Warning: SAPO path for seed {seed} not found")
        return None

    try:
        with open(sapo_path, "r") as f:
            data = json.load(f)
            if "sapo" in data and "rewards_per_timestep" in data["sapo"]:
                return data["sapo"]["rewards_per_timestep"]
            else:
                print(f"Warning: SAPO rewards_per_timestep not found in {sapo_path}")
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Warning: Error reading SAPO data from {sapo_path}: {e}")

    return None


def plot_parameter_sweep(base_dir, varied_param, param_values, param_label, output_path,
                         sapo_center=None, sapo_lower=None, sapo_upper=None):
    """Create a plot for a single parameter sweep."""
    output_path = _as_svg_path(output_path)
    # Group configurations by parameter value
    data_by_param = defaultdict(list)

    configs = generate_configurations()
    for config in configs:
        if config["varied_param"] != varied_param:
            continue

        param_value = get_param_value(config, varied_param)
        reward = load_total_reward(base_dir, config)

        if reward is not None:
            data_by_param[param_value].append(reward)

    # Prepare data for plotting
    param_values_sorted = sorted(data_by_param.keys())
    centers = []
    lowers = []
    uppers = []
    all_points = []

    for param_val in param_values_sorted:
        rewards = data_by_param[param_val]
        if rewards:
            center, lower, upper = calculate_center_and_bounds(rewards)
            centers.append(center)
            lowers.append(lower)
            uppers.append(upper)
            all_points.append((param_val, rewards))

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot individual points (one per seed)
    for param_val, rewards in all_points:
        x_pos = [param_val] * len(rewards)
        print(x_pos, rewards)
        ax.scatter(x_pos, rewards, alpha=1, s=50, facecolors='none', edgecolors='gray', linewidths=0.5, zorder=1)

    # Plot center line with asymmetric bounds (CI or std band)
    if centers:
        if USE_CONFIDENCE_INTERVALS:
            error_label = 'Median with 95% CI'
        else:
            error_label = 'Mean ± Std'

        centers_arr = np.asarray(centers, dtype=float)
        lowers_arr = np.asarray(lowers, dtype=float)
        uppers_arr = np.asarray(uppers, dtype=float)
        yerr = np.vstack([centers_arr - lowers_arr, uppers_arr - centers_arr])

        # Shaded CI/std band (like plot_rewards_over_time_averaged)
        ax.fill_between(
            param_values_sorted,
            lowers_arr,
            uppers_arr,
            color='blue',
            alpha=0.2,
            zorder=0,
            edgecolor='none'
        )

        ax.errorbar(
            param_values_sorted, centers_arr, yerr=yerr,
            marker='x', markersize=6, linewidth=1, capsize=0,
            capthick=1, color='blue', zorder=2, label=error_label, alpha=0.7
        )

    # Add SAPO reference line
    if sapo_center is not None and sapo_lower is not None and sapo_upper is not None:
        if USE_CONFIDENCE_INTERVALS:
            sapo_error_label = f'SAPO (median={sapo_center:.2f}, 95% CI [{sapo_lower:.2f}, {sapo_upper:.2f}])'
        else:
            sapo_std = float(sapo_upper - sapo_center)
            sapo_error_label = f'SAPO (mean={sapo_center:.2f}±{sapo_std:.2f})'
        ax.axhline(
            y=sapo_center, color='black', linestyle='-', linewidth=1.5,
            label=sapo_error_label, zorder=3
        )
        # Add error shading
        ax.axhspan(
            sapo_lower, sapo_upper,
            alpha=0.2, color='black', zorder=1
        )

    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    # ax.set_title(f'Total Reward vs {param_label}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_all_runs(base_dirs, output_path, sapo_center=None, sapo_lower=None, sapo_upper=None):
    """Create a single plot showing all runs across all parameters.
    
    Args:
        base_dirs: Single base directory (Path/str) or list of base directories
    """
    output_path = _as_svg_path(output_path)
    configs = generate_configurations()

    # Collect all data
    all_data = []
    for config in configs:
        reward = load_total_reward(base_dirs, config)
        if reward is not None:
            varied_param = config["varied_param"]
            all_data.append({
                "varied_param": varied_param,
                "param_value": get_param_value(config, varied_param),
                "reward": reward,
                "seed": config["seed"],
            })

    # Create single plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors for each parameter
    colors = {
        "N": "blue",
        "max_iter": "green",
        "lr": "orange",
        "sample": "purple",
    }

    param_info = [
        ("N", "N (Horizon)", N_VALUES),
        ("max_iter", "Max Iterations", MAX_ITER_VALUES),
        ("lr", "Learning Rate", LR_VALUES),
        ("sample", "Sample", SAMPLE_VALUES),
    ]

    # Each parameter gets its own x-axis range (offset by 1 unit each)
    x_offset = 0
    all_ticks = []
    all_tick_labels = []

    for param_key, param_label, param_values_list in param_info:
        # Filter data for this parameter
        param_data = [d for d in all_data if d["varied_param"] == param_key]

        # Group by parameter value
        data_by_param = defaultdict(list)
        for d in param_data:
            data_by_param[d["param_value"]].append(d["reward"])

        # Prepare for plotting
        param_values_sorted = sorted(data_by_param.keys())
        centers = []
        lowers = []
        uppers = []
        all_points = []

        for param_val in param_values_sorted:
            rewards = data_by_param[param_val]
            if rewards:
                center, lower, upper = calculate_center_and_bounds(rewards)
                centers.append(center)
                lowers.append(lower)
                uppers.append(upper)
                all_points.append((param_val, rewards))

        # Normalize x-values to span 0-1, then offset
        if param_values_sorted:
            min_val = min(param_values_sorted)
            max_val = max(param_values_sorted)
            if max_val == min_val:
                # Handle case where all values are the same
                normalized_x = [x_offset + 0.5] * len(param_values_sorted)
            else:
                # Normalize to 0-1 range
                normalized_x = [
                    x_offset + (val - min_val) / (max_val - min_val)
                    for val in param_values_sorted
                ]

            # Plot individual points (one per seed)
            color = colors.get(param_key, "black")
            for (param_val, rewards), x_pos in zip(all_points, normalized_x):
                x_positions = [x_pos] * len(rewards)
                ax.scatter(x_positions, rewards, alpha=0.3, s=30, color=color, zorder=1)

            # Plot mean line with error bars
            if centers:
                centers_arr = np.asarray(centers, dtype=float)
                lowers_arr = np.asarray(lowers, dtype=float)
                uppers_arr = np.asarray(uppers, dtype=float)
                yerr = np.vstack([centers_arr - lowers_arr, uppers_arr - centers_arr])
                ax.errorbar(
                    normalized_x, centers_arr, yerr=yerr,
                    marker='x', markersize=6, linewidth=1, capsize=3,
                    capthick=1, color=color, zorder=2, label=param_label
                )

            # Add ticks and labels for this parameter
            for val, x_pos in zip(param_values_sorted, normalized_x):
                all_ticks.append(x_pos)
                # Format label based on parameter type
                if param_key == "lr":
                    all_tick_labels.append(f"{val:.3f}")
                elif param_key == "sample":
                    all_tick_labels.append(str(val))
                else:
                    all_tick_labels.append(str(val))

        # Add vertical separator line between parameters
        if x_offset > 0:
            ax.axvline(x=x_offset - 0.1, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

        # Move to next x-axis range
        x_offset += 1.2  # 1 unit for data + 0.2 spacing

    # Set custom x-axis ticks and labels
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_tick_labels, rotation=45, ha='right', fontsize=9)

    # Add parameter name labels below x-axis
    x_positions = [0.5, 1.7, 2.9, 4.1]  # Approximate centers of each parameter section
    param_names = ["N", "max_iter", "lr", "sample"]
    for x_pos, name in zip(x_positions, param_names):
        ax.text(x_pos, -0.08, name, transform=ax.get_xaxis_transform(),
                ha='center', fontsize=10, fontweight='bold', color=colors.get(name, "black"))

    # Add SAPO reference line
    if sapo_center is not None and sapo_lower is not None and sapo_upper is not None:
        if USE_CONFIDENCE_INTERVALS:
            sapo_error_label = f'SAPO (median={sapo_center:.2f}, 95% CI [{sapo_lower:.2f}, {sapo_upper:.2f}])'
        else:
            sapo_std = float(sapo_upper - sapo_center)
            sapo_error_label = f'SAPO (mean={sapo_center:.2f}±{sapo_std:.2f})'
        ax.axhline(
            y=sapo_center, color='black', linestyle='-', linewidth=1.5,
            label=sapo_error_label, zorder=3
        )
        # Add error shading
        ax.axhspan(
            sapo_lower, sapo_upper,
            alpha=0.2, color='black', zorder=1
        )

    # ax.set_xlabel('Parameter Values (Normalized)', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    # ax.set_title('Hyperparameter Sweep Results - All Parameters',
    #              fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot: {output_path}")


def plot_rewards_over_time(base_dirs, seed, output_path, varied_param_filter=None):
    """Plot rewards_per_timestep over time for all DSS configs and SAPO for a given seed.
    
    Args:
        base_dirs: Single base directory (Path/str) or list of base directories
        seed: Seed to plot
        output_path: Output file path
        varied_param_filter: If provided, only plot configs with this varied_param (e.g., "N", "max_iter")
    """
    output_path = _as_svg_path(output_path)
    configs = generate_configurations()

    # Filter configs for the specified seed
    seed_configs = [c for c in configs if c["seed"] == seed]

    # Filter by varied_param if specified
    if varied_param_filter:
        seed_configs = [c for c in seed_configs if c["varied_param"] == varied_param_filter]

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Use consistent colors from module-level definition
    param_colors = PARAM_COLORS

    # Plot each DSS configuration
    plotted_configs = []
    for config in seed_configs:
        rewards = load_rewards_per_timestep(base_dirs, config)
        if rewards is None or len(rewards) == 0:
            continue

        # Create label for this config
        varied_param = config["varied_param"]
        param_value = get_param_value(config, varied_param)
        if varied_param == "lr":
            label = f"{varied_param}={param_value:.3f}"
        else:
            label = f"{varied_param}={param_value}"

        # Add other fixed params to label if not base values AND not the varied param
        if config["n"] != BASE_CONFIG["n"] and varied_param != "N":
            label += f", n={config['n']}"
        if config["max_iter"] != BASE_CONFIG["max_iter"] and varied_param != "max_iter":
            label += f", iter={config['max_iter']}"
        if config["lr"] != BASE_CONFIG["lr"] and varied_param != "lr":
            label += f", lr={config['lr']}"
        if config["sample"] != BASE_CONFIG["sample"] and varied_param != "sample":
            label += f", sample={config['sample']}"

        # Get base color for this parameter
        base_color = param_colors.get(varied_param, "#7f7f7f")
        
        # Get distinct color/style based on param value index
        param_values_list = {
            "N": N_VALUES,
            "max_iter": MAX_ITER_VALUES,
            "lr": LR_VALUES,
            "sample": SAMPLE_VALUES,
        }.get(varied_param, [])
        
        try:
            value_idx = param_values_list.index(param_value)
            color = CONFIG_COLORS[value_idx % len(CONFIG_COLORS)]
            linestyle = LINE_STYLES[value_idx % len(LINE_STYLES)]
        except (ValueError, KeyError):
            color = base_color
            linestyle = '-'
        
        timesteps = list(range(1, len(rewards) + 1))

        ax.plot(
            timesteps, rewards,
            linewidth=1, color=color, linestyle=linestyle, alpha=0.8,
            label=label, zorder=2
        )

        plotted_configs.append(label)

    # Plot SAPO
    sapo_rewards = load_sapo_rewards_per_timestep(seed)
    if sapo_rewards is not None and len(sapo_rewards) > 0:
        sapo_timesteps = list(range(1, len(sapo_rewards) + 1))
        ax.plot(
            sapo_timesteps, sapo_rewards,
            linewidth=1, color='black', linestyle='-',
            label='SAPO', zorder=3
        )

    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Reward per Timestep', fontsize=12)
    if varied_param_filter:
        param_labels = {"N": "N (Horizon)", "max_iter": "Max Iterations", 
                       "lr": "Learning Rate", "sample": "Sample"}
        title = f'Rewards Over Time - {param_labels.get(varied_param_filter, varied_param_filter)} Sweep - Seed {seed}'
    else:
        title = f'Rewards Over Time - All Runs - Seed {seed}'
    # ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best', ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved rewards over time plot: {output_path}")


def plot_rewards_over_time_averaged(base_dirs, output_path, varied_param_filter=None):
    """Plot averaged rewards_per_timestep over time across all seeds for all DSS configs and SAPO.
    
    Args:
        base_dirs: Single base directory (Path/str) or list of base directories
        output_path: Output file path
        varied_param_filter: If provided, only plot configs with this varied_param (e.g., "N", "max_iter")
    """
    output_path = _as_svg_path(output_path)
    configs = generate_configurations()

    # Filter by varied_param if specified
    if varied_param_filter:
        configs = [c for c in configs if c["varied_param"] == varied_param_filter]

    # Group configs by their parameter values (excluding seed)
    config_groups = defaultdict(list)
    for config in configs:
        # Create a key that identifies the config without seed
        key = (
            config["varied_param"],
            get_param_value(config, config["varied_param"]),
            config["n"],
            config["max_iter"],
            config["lr"],
            config["sample"],
        )
        config_groups[key].append(config)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each configuration group (averaged over seeds)
    for (varied_param, param_value, n, max_iter, lr, sample), group_configs in config_groups.items():
        # Collect rewards for all seeds
        all_rewards = []
        for config in group_configs:
            rewards = load_rewards_per_timestep(base_dirs, config)
            if rewards is not None and len(rewards) > 0:
                all_rewards.append(rewards)

        if not all_rewards:
            continue

        # Average over seeds
        min_length = min(len(r) for r in all_rewards)
        all_rewards = [r[:min_length] for r in all_rewards]
        all_rewards_array = np.array(all_rewards)
        center_rewards, lower_rewards, upper_rewards = calculate_center_and_bounds(all_rewards_array, axis=0)

        # Create label
        if varied_param == "lr":
            label = f"{varied_param}={param_value:.3f}"
        else:
            label = f"{varied_param}={param_value}"

        # Add other fixed params to label if not base values AND not the varied param
        if n != BASE_CONFIG["n"] and varied_param != "N":
            label += f", n={n}"
        if max_iter != BASE_CONFIG["max_iter"] and varied_param != "max_iter":
            label += f", iter={max_iter}"
        if lr != BASE_CONFIG["lr"] and varied_param != "lr":
            label += f", lr={lr}"
        if sample != BASE_CONFIG["sample"] and varied_param != "sample":
            label += f", sample={sample}"

        # Get distinct color/style based on param value index
        param_values_list = {
            "N": N_VALUES,
            "max_iter": MAX_ITER_VALUES,
            "lr": LR_VALUES,
            "sample": SAMPLE_VALUES,
        }.get(varied_param, [])

        try:
            value_idx = param_values_list.index(param_value)
            color = CONFIG_COLORS[value_idx % len(CONFIG_COLORS)]
            linestyle = LINE_STYLES[value_idx % len(LINE_STYLES)]
        except (ValueError, KeyError):
            color = PARAM_COLORS.get(varied_param, "#7f7f7f")
            linestyle = '-'

        timesteps = list(range(1, len(center_rewards) + 1))

        # Plot mean with shaded error region
        ax.plot(
            timesteps, center_rewards,
            linewidth=1, color=color, linestyle=linestyle, alpha=0.9,
            label=label, zorder=2
        )
        # ax.fill_between(
        #     timesteps,
        #     lower_rewards,
        #     upper_rewards,
        #     color=color, alpha=0.1, zorder=1, edgecolor='none'
        # )

    # Plot averaged SAPO across available seeds
    all_sapo_rewards = []
    for seed in SEED_VALUES:
        sapo_rewards = load_sapo_rewards_per_timestep(seed)
        if sapo_rewards is not None and len(sapo_rewards) > 0:
            all_sapo_rewards.append(sapo_rewards)

    if all_sapo_rewards:
        min_length = min(len(r) for r in all_sapo_rewards)
        all_sapo_rewards = [r[:min_length] for r in all_sapo_rewards]
        all_sapo_rewards_array = np.array(all_sapo_rewards)
        center_sapo, lower_sapo, upper_sapo = calculate_center_and_bounds(all_sapo_rewards_array, axis=0)
        sapo_timesteps = list(range(1, len(center_sapo) + 1))

        ax.plot(
            sapo_timesteps, center_sapo,
            linewidth=1, color='black', linestyle='-',
            label='SAPO', zorder=3
        )
        # ax.fill_between(
        #     sapo_timesteps,
        #     lower_sapo,
        #     upper_sapo,
        #     color='black', alpha=0.1, zorder=1, edgecolor='none'
        # )

    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Reward per Timestep', fontsize=12)
    if varied_param_filter:
        param_labels = {"N": "N (Horizon)", "max_iter": "Max Iterations",
                       "lr": "Learning Rate", "sample": "Sample"}
        title = f'Rewards Over Time (Averaged) - {param_labels.get(varied_param_filter, varied_param_filter)} Sweep'
    else:
        title = 'Rewards Over Time (Averaged) - All Runs'
    # ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best', ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved averaged rewards over time plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate graphs from DSSMPC hyperparameter sweep results"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sweep", "rewards"],
        default="sweep",
        help="Plotting mode: 'sweep' for parameter sweep plots, 'rewards' for rewards over time"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to use for rewards over time plot (default: all seeds)"
    )
    # parser.add_argument(
    #     "--base-dir",
    #     type=str,
    #     default="workdir/DSSMPC/2026_01_09-01_10_49",
    #     help="Base directory containing sweep results (default: workdir/DSSMPC/2026_01_09-01_10_49)"
    # )

    args = parser.parse_args()

    # Use multiple base directories
    base_dirs = [
        Path("workdir/DSSMPC/2026_01_10-17_07_50"),
        Path("workdir/DSSMPC/2026_01_11-18_11_55"),
        Path("workdir/DSSMPC/2026_01_18-02_59_42"),
    ]

    # Check if directories exist
    existing_dirs = [d for d in base_dirs if d.exists()]
    if not existing_dirs:
        print(f"Error: None of the base directories exist: {base_dirs}")
        return

    # Use last existing directory for output
    base_dir = existing_dirs[-1]
    output_dir = base_dir / "plots"
    output_dir.mkdir(exist_ok=True)

    print(f"Searching in {len(existing_dirs)} base directories: {existing_dirs}")

    if args.mode == "sweep":
        print("Generating plots from hyperparameter sweep results...")
        print(f"Output directory: {output_dir}")
        print()

        # Load SAPO reference data once
        sapo_center, sapo_lower, sapo_upper = load_sapo_reference()
        if sapo_center is not None:
            if USE_CONFIDENCE_INTERVALS:
                print(f"SAPO reference: median={sapo_center:.2f}, 95% CI=[{sapo_lower:.2f}, {sapo_upper:.2f}]")
            else:
                sapo_std = float(sapo_upper - sapo_center)
                print(f"SAPO reference: mean={sapo_center:.2f}, std={sapo_std:.2f}")
        print()

        # Generate individual plots for each parameter
        plot_parameter_sweep(
            existing_dirs, "N", N_VALUES, "N (Horizon)",
            output_dir / "sweep_N.svg", sapo_center, sapo_lower, sapo_upper
        )

        plot_parameter_sweep(
            existing_dirs, "max_iter", MAX_ITER_VALUES, "Max Iterations",
            output_dir / "sweep_max_iter.svg", sapo_center, sapo_lower, sapo_upper
        )

        plot_parameter_sweep(
            existing_dirs, "lr", LR_VALUES, "Learning Rate",
            output_dir / "sweep_lr.svg", sapo_center, sapo_lower, sapo_upper
        )

        # plot_parameter_sweep(
        #     existing_dirs, "sample", SAMPLE_VALUES, "Sample",
        #     output_dir / "sweep_sample.svg", sapo_mean, sapo_error
        # )

        # Generate combined plot
        plot_all_runs(existing_dirs, output_dir / "sweep_all_parameters.svg", sapo_center, sapo_lower, sapo_upper)

        print()
        print("All sweep plots generated successfully!")

    elif args.mode == "rewards":
        print(f"Generating rewards over time plots...")
        print(f"Output directory: {output_dir}")
        print()

        # Generate rewards over time plot for specified seed(s)
        if args.seed:
            seeds_to_plot = [args.seed]
        else:
            seeds_to_plot = SEED_VALUES

        # Generate plots for each seed
        for seed in seeds_to_plot:
            # Plot for each sweep type
            for varied_param in ["N", "max_iter", "lr", "sample"]:
                plot_rewards_over_time(
                    existing_dirs, seed,
                    output_dir / f"rewards_over_time_{varied_param}_seed{seed}.svg",
                    varied_param_filter=varied_param
                )

            # Plot with all runs
            plot_rewards_over_time(
                existing_dirs, seed,
                output_dir / f"rewards_over_time_all_seed{seed}.svg",
                varied_param_filter=None
            )

        # Generate averaged plots (over seeds) for each parameter
        for varied_param in ["N", "max_iter", "lr", "sample"]:
            plot_rewards_over_time_averaged(
                existing_dirs,
                output_dir / f"rewards_over_time_{varied_param}_averaged.svg",
                varied_param_filter=varied_param
            )

        # Generate averaged plot with all runs
        plot_rewards_over_time_averaged(
            existing_dirs,
            output_dir / f"rewards_over_time_all_averaged.svg",
            varied_param_filter=None
        )

        print()
        print("All rewards over time plots generated successfully!")


if __name__ == "__main__":
    main()

