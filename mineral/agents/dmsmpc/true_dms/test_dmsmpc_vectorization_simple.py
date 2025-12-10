"""Simple tests for TRUE DMS MPC vectorization (no pytest required).

This module verifies that vectorized implementations produce identical results
to their non-vectorized counterparts.
"""

import torch


class TestTaskCostVectorization:
    """Test vectorization of task cost computation."""

    def __init__(self):
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

    def _compute_task_cost_loop(self, agent, x, u):
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

    def _compute_task_cost_vectorized(self, agent, x, u):
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

        # Check if equal within tolerance
        diff = torch.abs(cost_loop - cost_vectorized).item()
        rel_error = diff / (torch.abs(cost_loop).item() + 1e-8)

        print(f"  Loop cost: {cost_loop.item():.6e}")
        print(f"  Vectorized cost: {cost_vectorized.item():.6e}")
        print(f"  Absolute difference: {diff:.6e}")
        print(f"  Relative error: {rel_error:.6e}")

        assert rel_error < 1e-5, f"Costs differ too much: relative error = {rel_error}"
        return True

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

        # Check gradient differences
        grad_x_diff = torch.max(torch.abs(grad_x_loop - grad_x_vec)).item()
        grad_u_diff = torch.max(torch.abs(grad_u_loop - grad_u_vec)).item()

        print(f"  Max state gradient difference: {grad_x_diff:.6e}")
        print(f"  Max control gradient difference: {grad_u_diff:.6e}")

        assert grad_x_diff < 1e-4, f"State gradients differ: {grad_x_diff}"
        assert grad_u_diff < 1e-4, f"Control gradients differ: {grad_u_diff}"
        return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 70)
    print("TRUE DMS Vectorization Tests")
    print("=" * 70)
    print()

    tester = TestTaskCostVectorization()

    tests = [
        ("Task Cost Equivalence (Random States)", tester.test_task_cost_equivalence_random_states),
        ("Task Cost Gradient Equivalence", tester.test_task_cost_gradient_equivalence),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        try:
            test_func()
            print(f"  ✓ PASSED\n")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}\n")
            failed += 1

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)


