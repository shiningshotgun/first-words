"""Tests for author-direction KL divergence probe."""

import unittest
import numpy as np
import torch
import torch.nn.functional as F

from src.probes import compute_kl_pi_b_to_model


class TestKLProbe(unittest.TestCase):
    """Test suite for author-direction KL(pi_b || model) probe."""

    def test_handcrafted_closed_form_with_zeros(self):
        """Test KL against manual closed-form calculation with zero-target entries contributing 0."""
        # 3-symbol vocab, dim 2
        w_E = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
        w_U = torch.tensor([[1.0, -1.0], [0.5, 0.5], [-1.0, 1.0]], dtype=torch.float64)
        W_F = torch.tensor([[0.5, -0.5], [1.0, 0.0]], dtype=torch.float64)

        # Target distribution containing exact zeros
        pi_b = np.array([
            [0.7, 0.3, 0.0],
            [0.0, 0.5, 0.5],
            [0.2, 0.8, 0.0],
        ], dtype=np.float64)

        # Closed-form model logits: w_E @ W_F^T @ w_U^T
        logits = w_E @ W_F.t() @ w_U.t()
        log_probs = F.log_softmax(logits, dim=-1).numpy()

        # Closed-form KL(pi_b || model) = sum_j pi_b[k, j] * (log(pi_b[k, j]) - log_probs[k, j])
        expected_rows = []
        for k in range(3):
            row_sum = 0.0
            for j in range(3):
                p_val = pi_b[k, j]
                if p_val > 0.0:
                    row_sum += p_val * (np.log(p_val) - log_probs[k, j])
            expected_rows.append(row_sum)
        expected_kl = float(np.mean(expected_rows))

        computed_kl = compute_kl_pi_b_to_model(W_F, w_E, w_U, pi_b)
        self.assertTrue(np.isfinite(computed_kl))
        self.assertAlmostEqual(computed_kl, expected_kl, delta=1e-12)

    def test_asymmetric_direction_pin(self):
        """Asymmetric fixture ensuring KL(pi_b || model) differs from reverse KL(model || pi_b) by >= 0.3."""
        w_E = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
        w_U = torch.tensor([[3.0, 0.0], [0.0, 3.0], [-3.0, -3.0]], dtype=torch.float64)
        W_F = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)

        pi_b_pos = np.array([
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
        ], dtype=np.float64)

        logits = w_E @ W_F.t() @ w_U.t()
        probs = F.softmax(logits, dim=-1).numpy()

        # Forward direction: KL(pi_b || model)
        kl_fwd = compute_kl_pi_b_to_model(W_F, w_E, w_U, pi_b_pos)

        # Reverse direction: KL(model || pi_b)
        kl_rev = 0.0
        for k in range(3):
            for j in range(3):
                kl_rev += probs[k, j] * (np.log(probs[k, j]) - np.log(pi_b_pos[k, j]))
        kl_rev /= 3.0

        gap = abs(kl_fwd - kl_rev)
        self.assertGreaterEqual(gap, 0.3)
        self.assertAlmostEqual(kl_fwd, 2.626001610551262, delta=1e-10)


if __name__ == "__main__":
    unittest.main()
