import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

from mineral.envs.rewarped import make_envs
from mineral.examples.agents.unified_mpc.unified_mpc import UnifiedMPC

if __name__ == "__main__":
    print("Hello, World!")