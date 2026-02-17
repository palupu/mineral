#!/bin/bash
# Hyperparameter sweep script for DSSMPC
# This script runs multiple configurations and organizes results in timestamped folders

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create main timestamp folder
TIMESTAMP=$(date +%Y_%m_%d-%H_%M_%S_%2N)
BASE_DIR="workdir/DSSMPC/${TIMESTAMP}"
mkdir -p "${BASE_DIR}"

echo "=========================================="
echo "DSSMPC Hyperparameter Sweep"
echo "Main output directory: ${BASE_DIR}"
echo "=========================================="

# Define hyperparameter arrays
# Modify these arrays to change the sweep configuration

# Horizon lengths to test
N_VALUES=(1 3 5 9)

# Maximum Adam iterations per MPC step
MAX_ITER_VALUES=(10 30 50)

# Learning rates for Adam optimizer
LR_VALUES=(0.01 0.005)

# Whether to sample from SHAC policy (true) or use mean (false)
SAMPLE_VALUES=(false)

# Random seeds for reproducibility
SEED_VALUES=(1 2 3)

# Additional configurations with sample=true (for promising configs)
# These will be added separately
SAMPLE_TRUE_CONFIGS=(
    # Format: "N max_iter lr"
    "3 30 0.01"
    "5 30 0.01"
    "5 50 0.01"
)

# Calculate total configurations
TOTAL=0
for N in "${N_VALUES[@]}"; do
    for MAX_ITER in "${MAX_ITER_VALUES[@]}"; do
        for LR in "${LR_VALUES[@]}"; do
            for SAMPLE in "${SAMPLE_VALUES[@]}"; do
                for SEED in "${SEED_VALUES[@]}"; do
                    TOTAL=$((TOTAL + 1))
                done
            done
        done
    done
done

# Add sample=true configurations
for config in "${SAMPLE_TRUE_CONFIGS[@]}"; do
    read -r N MAX_ITER LR <<< "${config}"
    for SEED in "${SEED_VALUES[@]}"; do
        TOTAL=$((TOTAL + 1))
    done
done

echo "Total configurations to run: ${TOTAL}"
echo ""

CURRENT=0

# Run main grid search (sample=false)
for N in "${N_VALUES[@]}"; do
    for MAX_ITER in "${MAX_ITER_VALUES[@]}"; do
        for LR in "${LR_VALUES[@]}"; do
            for SAMPLE in "${SAMPLE_VALUES[@]}"; do
                for SEED in "${SEED_VALUES[@]}"; do
                    CURRENT=$((CURRENT + 1))
                    echo ""
                    echo "[${CURRENT}/${TOTAL}] N=${N} max_iter=${MAX_ITER} lr=${LR} sample=${SAMPLE} seed=${SEED}"
                    echo "----------------------------------------"
                    
                    # Run the single configuration script
                    "${SCRIPT_DIR}/run_dssmpc.sh" "${BASE_DIR}" "${N}" "${MAX_ITER}" "${LR}" "${SAMPLE}" "${SEED}"
                    
                    # Check exit status
                    if [ $? -eq 0 ]; then
                        echo "✓ Configuration ${CURRENT}/${TOTAL} completed successfully"
                    else
                        echo "✗ Configuration ${CURRENT}/${TOTAL} failed"
                    fi
                done
            done
        done
    done
done

# Run additional sample=true configurations
for config in "${SAMPLE_TRUE_CONFIGS[@]}"; do
    read -r N MAX_ITER LR <<< "${config}"
    for SEED in "${SEED_VALUES[@]}"; do
        CURRENT=$((CURRENT + 1))
        SAMPLE="true"
        echo ""
        echo "[${CURRENT}/${TOTAL}] N=${N} max_iter=${MAX_ITER} lr=${LR} sample=${SAMPLE} seed=${SEED}"
        echo "----------------------------------------"
        
        # Run the single configuration script
        "${SCRIPT_DIR}/run_dssmpc.sh" "${BASE_DIR}" "${N}" "${MAX_ITER}" "${LR}" "${SAMPLE}" "${SEED}"
        
        # Check exit status
        if [ $? -eq 0 ]; then
            echo "✓ Configuration ${CURRENT}/${TOTAL} completed successfully"
        else
            echo "✗ Configuration ${CURRENT}/${TOTAL} failed"
        fi
    done
done

echo ""
echo "=========================================="
echo "Hyperparameter sweep complete!"
echo "Results saved in: ${BASE_DIR}"
echo "Total configurations: ${TOTAL}"
echo "=========================================="
