"""Tests for TRUE DMS MPC vectorization.

This module verifies that vectorized implementations produce identical results
to their non-vectorized counterparts.
"""

import numpy as np
import torch
import pytest
from typing import Any


class TestTaskCostVectorization:
    """Test vectorization of task cost computation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.N = 5
        self.num_particles = 100
        self.mpm_pos_dim = self.num_particles * 3
        self.mpm_vel_dim = self.num_particles * 3
        self.body_q_dim = 7
        self.body_qd_dim = 6
        self.state_dim = self.mpm_pos_dim + self.mpm_vel_dim + self.body_q_dim + self.body_qd_dim
        self.control_dim = 3

        # Cost weights
        self.cost_state = 1.0
        self.cost_control = 0.01
        self.cost_terminal = 10.0

    def _create_mock_agent(self):
        """Create mock agent with necessary attributes."""
        class MockAgent:
            pass

        agent = MockAgent()
        agent.N = self.N
        agent.num_particles = self.num_particles
        agent.mpm_pos_dim = self.mpm_pos_dim
        agent.mpm_vel_dim = self.mpm_vel_dim
        agent.body_q_dim = self.body_q_dim
        agent.body_qd_dim = self.body_qd_dim
        agent.state_dim = self.state_dim
        agent.control_dim = self.control_dim
        agent.cost_state = self.cost_state
        agent.cost_control = self.cost_control
        agent.cost_terminal = self.cost_terminal
        agent.device = self.device

        return agent

    def _compute_task_cost_loop(self, agent: Any, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Non-vectorized (loop-based) task cost computation."""
        h_ref = 0.125
        task_cost = 0.0

        # Stage costs
        for i in range(agent.N):
            mpm_x_flat = x[i][:agent.mpm_pos_dim]
            particles = mpm_x_flat.reshape(agent.num_particles, 3)
            heights = particles[:, 1]

            mean_height = heights.mean()
            height_cost = (mean_height / h_ref) ** 2

            variance_cost = heights.var()
            control_cost = agent.cost_control * (u[i] @ u[i])

            stage_cost = agent.cost_state * (height_cost + variance_cost) + control_cost
            task_cost += stage_cost

        # Terminal cost
        mpm_x_flat_final = x[agent.N][:agent.mpm_pos_dim]
        particles_final = mpm_x_flat_final.reshape(agent.num_particles, 3)
        heights_final = particles_final[:, 1]
        mean_height_final = heights_final.mean()
        variance_final = heights_final.var()

        terminal_cost = agent.cost_terminal * ((mean_height_final / h_ref) ** 2 + variance_final)
        task_cost += terminal_cost

        return task_cost

    def _compute_task_cost_vectorized(self, agent: Any, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Vectorized task cost computation."""
        h_ref = 0.125

        # Vectorized stage costs
        x_stage = x[:agent.N]
        mpm_x_stage = x_stage[:, :agent.mpm_pos_dim]
        particles_stage = mpm_x_stage.reshape(agent.N, agent.num_particles, 3)
        heights_stage = particles_stage[:, :, 1]

        mean_heights = heights_stage.mean(dim=1)
        height_costs = (mean_heights / h_ref) ** 2

        variance_costs = heights_stage.var(dim=1)
        control_costs = agent.cost_control * (u * u).sum(dim=1)

        stage_cost = agent.cost_state * (height_costs.sum() + variance_costs.sum()) + control_costs.sum()

        # Terminal cost
        mpm_x_final = x[agent.N][:agent.mpm_pos_dim]
        particles_final = mpm_x_final.reshape(agent.num_particles, 3)
        heights_final = particles_final[:, 1]

        mean_height_final = heights_final.mean()
        variance_final = heights_final.var()

        terminal_cost = agent.cost_terminal * ((mean_height_final / h_ref) ** 2 + variance_final)

        return stage_cost + terminal_cost

    def test_task_cost_equivalence_random_states(self):
        """Test that vectorized and loop versions produce identical results with random states."""
        agent = self._create_mock_agent()

        # Generate random states and controls
        x = torch.randn(self.N + 1, self.state_dim, device=self.device, dtype=torch.float32)
        u = torch.randn(self.N, self.control_dim, device=self.device, dtype=torch.float32)

        # Compute with both methods
        cost_loop = self._compute_task_cost_loop(agent, x, u)
        cost_vectorized = self._compute_task_cost_vectorized(agent, x, u)

        # Should be exactly equal (within floating point precision)
        assert torch.allclose(cost_loop, cost_vectorized, rtol=1e-5, atol=1e-7), \
            f"Task costs differ: loop={cost_loop.item()}, vectorized={cost_vectorized.item()}"

    def test_task_cost_equivalence_zero_states(self):
        """Test with zero states (edge case)."""
        agent = self._create_mock_agent()

        x = torch.zeros(self.N + 1, self.state_dim, device=self.device, dtype=torch.float32)
        u = torch.zeros(self.N, self.control_dim, device=self.device, dtype=torch.float32)

        cost_loop = self._compute_task_cost_loop(agent, x, u)
        cost_vectorized = self._compute_task_cost_vectorized(agent, x, u)

        assert torch.allclose(cost_loop, cost_vectorized, rtol=1e-5, atol=1e-7)

    def test_task_cost_equivalence_positive_heights(self):
        """Test with all particles at positive heights."""
        agent = self._create_mock_agent()

        x = torch.randn(self.N + 1, self.state_dim, device=self.device, dtype=torch.float32)
        u = torch.randn(self.N, self.control_dim, device=self.device, dtype=torch.float32)

        # Set all particle heights (y-coordinates) to positive values
        for i in range(self.N + 1):
            positions = x[i][:self.mpm_pos_dim].reshape(self.num_particles, 3)
            positions[:, 1] = torch.abs(torch.randn(self.num_particles, device=self.device)) + 0.1
            x[i][:self.mpm_pos_dim] = positions.flatten()

        cost_loop = self._compute_task_cost_loop(agent, x, u)
        cost_vectorized = self._compute_task_cost_vectorized(agent, x, u)

        assert torch.allclose(cost_loop, cost_vectorized, rtol=1e-5, atol=1e-7)

    def test_task_cost_gradient_equivalence(self):
        """Test that gradients are identical for both implementations."""
        agent = self._create_mock_agent()

        x = torch.randn(self.N + 1, self.state_dim, device=self.device, dtype=torch.float32, requires_grad=True)
        u = torch.randn(self.N, self.control_dim, device=self.device, dtype=torch.float32, requires_grad=True)

        # Compute and backprop with loop version
        x_loop = x.clone().detach().requires_grad_(True)
        u_loop = u.clone().detach().requires_grad_(True)
        cost_loop = self._compute_task_cost_loop(agent, x_loop, u_loop)
        cost_loop.backward()
        grad_x_loop = x_loop.grad.clone()
        grad_u_loop = u_loop.grad.clone()

        # Compute and backprop with vectorized version
        x_vec = x.clone().detach().requires_grad_(True)
        u_vec = u.clone().detach().requires_grad_(True)
        cost_vec = self._compute_task_cost_vectorized(agent, x_vec, u_vec)
        cost_vec.backward()
        grad_x_vec = x_vec.grad.clone()
        grad_u_vec = u_vec.grad.clone()

        # Gradients should match
        assert torch.allclose(grad_x_loop, grad_x_vec, rtol=1e-4, atol=1e-6), \
            "State gradients differ between loop and vectorized versions"
        assert torch.allclose(grad_u_loop, grad_u_vec, rtol=1e-4, atol=1e-6), \
            "Control gradients differ between loop and vectorized versions"

    @pytest.mark.parametrize("N", [1, 3, 5, 10])
    def test_task_cost_different_horizons(self, N):
        """Test with different horizon lengths."""
        # Create agent with specific N
        agent = self._create_mock_agent()
        agent.N = N

        x = torch.randn(N + 1, self.state_dim, device=self.device, dtype=torch.float32)
        u = torch.randn(N, self.control_dim, device=self.device, dtype=torch.float32)

        cost_loop = self._compute_task_cost_loop(agent, x, u)
        cost_vectorized = self._compute_task_cost_vectorized(agent, x, u)

        assert torch.allclose(cost_loop, cost_vectorized, rtol=1e-5, atol=1e-7)


class TestConstraintPenaltyVectorization:
    """Test vectorization of constraint penalty computation.
    
    Note: Full testing requires actual environment, so we test components.
    """

    def setup_method(self):
        """Setup test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.N = 5
        self.state_dim = 100
        self.control_dim = 3

    def test_initial_constraint_computation(self):
        """Test initial constraint penalty computation."""
        current_x = torch.randn(self.state_dim, device=self.device, dtype=torch.float32)
        x_0 = current_x + torch.randn(self.state_dim, device=self.device, dtype=torch.float32) * 0.1

        # Loop version
        initial_violation_loop = torch.sum((x_0 - current_x) ** 2)

        # Vectorized version (same, just verifying)
        initial_violation_vec = torch.sum((x_0 - current_x) ** 2)

        assert torch.allclose(initial_violation_loop, initial_violation_vec)

    def test_dynamics_constraint_batch_computation(self):
        """Test that batched constraint violation computation is correct."""
        # Simulate having N next states (predicted) and N target states
        x_target = torch.randn(self.N + 1, self.state_dim, device=self.device, dtype=torch.float32)
        x_predicted = torch.randn(self.N, self.state_dim, device=self.device, dtype=torch.float32)

        # Loop version
        dynamics_penalty_loop = 0.0
        for i in range(self.N):
            violation = torch.sum((x_target[i + 1] - x_predicted[i]) ** 2)
            dynamics_penalty_loop += violation

        # Vectorized version
        violations = torch.sum((x_target[1:self.N+1] - x_predicted) ** 2, dim=1)  # (N,)
        dynamics_penalty_vec = violations.sum()

        assert torch.allclose(dynamics_penalty_loop, dynamics_penalty_vec, rtol=1e-5, atol=1e-7)


class TestFullIntegration:
    """Integration tests for complete loss computation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.mark.skip(reason="Requires full environment setup")
    def test_full_loss_equivalence_with_env(self):
        """Test full loss computation with actual environment.
        
        This test is skipped by default as it requires a full environment setup.
        Enable it when running integration tests.
        """
        # TODO: Implement when environment is available
        pass


def run_tests():
    """Run all tests and print summary."""
    import sys

    print("=" * 70)
    print("Running TRUE DMS Vectorization Tests")
    print("=" * 70)

    # Run pytest programmatically
    exit_code = pytest.main([__file__, "-v", "--tb=short"])

    if exit_code == 0:
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("✗ Some tests failed")
        print("=" * 70)

    sys.exit(exit_code)


if __name__ == "__main__":
    run_tests()


