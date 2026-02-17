#!/usr/bin/env python3
"""Hyperparameter sweep script for DSSMPC.

This script runs multiple configurations and organizes results in timestamped folders.
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Get the directory of this script
SCRIPT_DIR = Path(__file__).parent.absolute()
RUN_DSSMPC_SCRIPT = SCRIPT_DIR / "run_dssmpc.sh"


def run_configuration(base_dir, n, max_iter, lr, sample, seed):
    """Run a single DSSMPC configuration with real-time output to log file."""
    cmd = [
        str(RUN_DSSMPC_SCRIPT),
        str(base_dir),
        str(n),
        str(max_iter),
        str(lr),
        str(sample),
        str(seed),
    ]

    # Create output log file path (same as bash script creates)
    run_folder = base_dir / f"N{n}_iter{max_iter}_lr{lr}_sample{sample}_seed{seed}"
    run_folder.mkdir(parents=True, exist_ok=True)
    log_file = run_folder / "output.log"

    start_time = time.time()

    # Open log file in text mode with line buffering for real-time output
    # Line buffering (1) ensures each line is written immediately
    with open(log_file, "w", encoding="utf-8", buffering=1) as log:
        # Use Popen for real-time streaming
        # text=True ensures stdout/stderr are text streams
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            text=True,  # Text mode (not binary)
            bufsize=1,  # Line buffered
        )
        # Wait for process to complete
        returncode = process.wait()

    elapsed_time = time.time() - start_time
    return returncode == 0, elapsed_time


def main():
    # Create main timestamp folder
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    base_dir = Path("workdir/DSSMPC") / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("DSSMPC Hyperparameter Sweep")
    print(f"Main output directory: {base_dir}")
    print("=" * 50)

    # Base configuration (middle/default values)
    # All other parameters are kept at these values when varying one parameter
    BASE_CONFIG = {
        "n": 5,
        "max_iter": 30,
        "lr": 0.01,
        "sample": False,
    }

    # Parameter values to test (one at a time)
    # N_VALUES = [1, 3, 5, 9]
    # MAX_ITER_VALUES = [10, 30, 50]
    # LR_VALUES = [0.05, 0.01, 0.005, 0.001]
    # SAMPLE_VALUES = [False, True]

    # N_VALUES = [4, 5, 7, 9]
    # MAX_ITER_VALUES = [20, 30, 40]
    # LR_VALUES = [0.01, 0.001]
    # SAMPLE_VALUES = [False]

    # N_VALUES = [4, 5, 7, 9]
    # MAX_ITER_VALUES = [30, 50, 70]
    # LR_VALUES = [0.01, 0.02, 0.03]
    # SAMPLE_VALUES = [False]

    N_VALUES = [1,2,3,12, 15]
    MAX_ITER_VALUES = [30]
    LR_VALUES = [0.01]
    SAMPLE_VALUES = [False]

    # Random seeds for reproducibility (3 seeds per config)
    SEED_VALUES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13] # [4, 5, 6, 7, 8] # [4, 5, 6] # [1, 2, 3]

    # Generate configurations using one-at-a-time approach
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

    # # Vary max_iter (keep others at base)
    # for max_iter in MAX_ITER_VALUES:
    #     for seed in SEED_VALUES:
    #         configurations.append({
    #             "n": BASE_CONFIG["n"],
    #             "max_iter": max_iter,
    #             "lr": BASE_CONFIG["lr"],
    #             "sample": BASE_CONFIG["sample"],
    #             "seed": seed,
    #             "varied_param": "max_iter",
    #         })

    # # Vary lr (keep others at base)
    # for lr in LR_VALUES:
    #     for seed in SEED_VALUES:
    #         configurations.append({
    #             "n": BASE_CONFIG["n"],
    #             "max_iter": BASE_CONFIG["max_iter"],
    #             "lr": lr,
    #             "sample": BASE_CONFIG["sample"],
    #             "seed": seed,
    #             "varied_param": "lr",
    #         })

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

    total = len(configurations)
    print(f"Base configuration: N={BASE_CONFIG['n']}, max_iter={BASE_CONFIG['max_iter']}, "
          f"lr={BASE_CONFIG['lr']}, sample={BASE_CONFIG['sample']}")
    print(f"Total configurations to run: {total}")
    print(f"  - N sweep: {len(N_VALUES)} values × {len(SEED_VALUES)} seeds = {len(N_VALUES) * len(SEED_VALUES)} runs")
    print(f"  - max_iter sweep: {len(MAX_ITER_VALUES)} values × {len(SEED_VALUES)} seeds = {len(MAX_ITER_VALUES) * len(SEED_VALUES)} runs")
    print(f"  - lr sweep: {len(LR_VALUES)} values × {len(SEED_VALUES)} seeds = {len(LR_VALUES) * len(SEED_VALUES)} runs")
    print(f"  - sample sweep: {len(SAMPLE_VALUES)} values × {len(SEED_VALUES)} seeds = {len(SAMPLE_VALUES) * len(SEED_VALUES)} runs")
    print()

    # Track results
    results = []
    total_start_time = time.time()

    # Run all configurations
    for idx, config in enumerate(configurations, 1):
        n = config["n"]
        max_iter = config["max_iter"]
        lr = config["lr"]
        sample = str(config["sample"]).lower()
        seed = config["seed"]

        # Get which parameter is being varied
        varied_param = config.get("varied_param", "unknown")

        print(f"[{idx}/{total}] {varied_param} sweep: N={n} max_iter={max_iter} lr={lr} sample={sample} seed={seed}", end="", flush=True)

        # Run the configuration
        success, elapsed_time = run_configuration(
            base_dir, n, max_iter, lr, sample, seed
        )

        # Record result
        result = {
            "config": config,
            "success": success,
            "elapsed_time": elapsed_time,
            "timestamp": datetime.now().isoformat(),
        }
        results.append(result)

        if success:
            print(f" ✓ ({elapsed_time:.2f}s)", flush=True)
        else:
            print(f" ✗ ({elapsed_time:.2f}s)", flush=True)

    total_elapsed_time = time.time() - total_start_time

    # Save summary
    summary = {
        "timestamp": timestamp,
        "base_dir": str(base_dir),
        "total_configurations": total,
        "total_time_seconds": total_elapsed_time,
        "total_time_hours": total_elapsed_time / 3600,
        "total_time_formatted": f"{int(total_elapsed_time // 3600)}h {int((total_elapsed_time % 3600) // 60)}m {int(total_elapsed_time % 60)}s",
        "successful_runs": sum(1 for r in results if r["success"]),
        "failed_runs": sum(1 for r in results if not r["success"]),
        "average_time_per_config": total_elapsed_time / total if total > 0 else 0,
        "results": results,
    }

    summary_path = base_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 50)
    print("Hyperparameter sweep complete!")
    print(f"Results saved in: {base_dir}")
    print(f"Total configurations: {total}")
    print(f"Successful: {summary['successful_runs']}")
    print(f"Failed: {summary['failed_runs']}")
    print(f"Total time: {summary['total_time_formatted']}")
    print(f"Average time per config: {summary['average_time_per_config']:.2f}s")
    print(f"Summary saved to: {summary_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()

