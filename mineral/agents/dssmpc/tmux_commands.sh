#!/bin/bash
# Tmux Commands Reference for DSSMPC Hyperparameter Sweep
# 
# This file contains helpful tmux commands for running long-running experiments
# that need to survive SSH disconnections (e.g., when laptop goes to sleep).

# ============================================================================
# BASIC SESSION MANAGEMENT
# ============================================================================

# Create a new tmux session named "dssmpc_sweep"
# tmux new -s dssmpc_sweep

# List all tmux sessions
# tmux ls

# Attach to an existing session
# tmux attach -t dssmpc_sweep
# OR shorter:
# tmux a -t dssmpc_sweep

# Kill a specific session
# tmux kill-session -t dssmpc_sweep

# Kill all tmux sessions
# tmux kill-server

# ============================================================================
# DETACHING AND REATTACHING
# ============================================================================

# Detach from current session (keeps it running in background):
# Press: Ctrl+B, then D
# OR from command line:
# tmux detach

# Reattach to a session (after SSH reconnection):
# tmux attach -t dssmpc_sweep

# ============================================================================
# WORKFLOW FOR OVERNIGHT RUNS
# ============================================================================

# 1. SSH into remote PC
#    ssh user@remote-pc

# 2. Start tmux session
#    tmux new -s dssmpc_sweep

# 3. Inside tmux, navigate and run script
#    cd /app
#    python3 ./mineral/mineral/agents/dssmpc/run_multiple_dssmpc.py

# 4. Detach from tmux (Ctrl+B, then D)
#    Now you can close SSH, laptop can sleep, etc.

# 5. Later, SSH back and reattach
#    ssh user@remote-pc
#    tmux attach -t dssmpc_sweep

# ============================================================================
# USEFUL KEYBINDINGS (while inside tmux)
# ============================================================================

# Ctrl+B is the default prefix key. Press it, then:
#   D  - Detach from session
#   C  - Create new window
#   N  - Next window
#   P  - Previous window
#   %  - Split pane vertically
#   "  - Split pane horizontally
#   Arrow keys - Navigate between panes
#   X  - Close current pane
#   [  - Enter copy mode (scroll up)
#   ]  - Paste
#   ?  - Show all keybindings

# ============================================================================
# MONITORING OUTPUT
# ============================================================================

# View output in real-time while attached:
#   - Just watch the terminal output
#   - Use Ctrl+B then [ to scroll up and see history

# View output from outside tmux (if script writes to file):
#   tail -f workdir/DSSMPC/*/output.log

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# If you get "session not found":
#   tmux ls  # Check what sessions exist

# If you get "session already exists":
#   tmux attach -t dssmpc_sweep  # Attach instead of creating new

# To see all tmux sessions and their status:
#   tmux ls

# To rename a session:
#   tmux rename-session -t old_name new_name

