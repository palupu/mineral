"""TRUE Direct Multiple Shooting MPC Implementation.

This module contains a proper implementation of Direct Multiple Shooting (DMS)
with independent shooting nodes, as opposed to the pseudo-DMS in the parent directory.

Key Features:
- True independent shooting nodes
- State setting from reduced state vectors  
- Two strategies: 'joint_only' (recommended) and 'joint_com' (experimental)
- Comprehensive documentation and examples

Usage:
    from mineral.agents.dmsmpc.true_dms.dmsmpc_true import TrueDMSMPCAgent
    
    agent = TrueDMSMPCAgent(cfg)
    agent.eval()

Files:
- dmsmpc_true.py: Main implementation
- TRUE_DMS_EXPLAINED.md: Technical documentation
- IMPLEMENTATION_SUMMARY.md: What was implemented
- example_config_true_dms.yaml: Configuration template
- test_true_dms.py: Test and demonstration scripts

Documentation:
For a comparison with other methods, see ../COMPARISON_GUIDE.md
"""

from .dmsmpc_true import TrueDMSMPCAgent

__all__ = ['TrueDMSMPCAgent']

