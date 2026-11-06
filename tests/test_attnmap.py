"""Tests for attention map properties and determinism."""

import unittest
import numpy as np
import torch

from src.model import SimplifiedTransformer
from src.data import SyntheticBigramDataset


class TestAttentionMap(unittest.TestCase):
    """Test suite for attention map mathematical properties and determinism."""

    def test_post_softmax_properties(self):
        """Values in [0, 1], strict upper triangle exactly 0.0, rows sum to 1 at float32 (atol 1e-6)."""
        d = 16
        N = 10
        T = 32
        model = SimplifiedTransformer(d=d, N=N, T=T, seed=42)

        z = torch.randint(0, N, (2, T))
        logits, (A1, A2) = model(z, return_attention_weights=True)

        for name, A in [("Layer 1", A1), ("Layer 2", A2)]:
            self.assertTrue((A >= 0.0).all().item(), f"{name} attention has negative values")
            self.assertTrue((A <= 1.0).all().item(), f"{name} attention exceeds 1.0")

            B = A.shape[0]
            for b in range(B):
                # Strict upper triangle must be exactly 0.0 (causal mask)
                upper = torch.triu(A[b], diagonal=1)
                self.assertTrue((upper == 0.0).all().item(), f"{name} upper triangle is non-zero")

                # Rows sum to 1.0
                row_sums = A[b].sum(dim=-1)
                self.assertTrue(
                    torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6),
                    f"{name} rows do not sum to 1.0",
                )

    def test_attention_determinism(self):
        """Same model seed and dataset seed produce bitwise-identical attention matrices and sequences."""
        d = 16
        N = 8
        T = 24
        K = 2

        pi_u = np.ones(N) / N
        pi_b = np.ones((N, N)) / N

        ds1 = SyntheticBigramDataset(pi_u, pi_b, T=T, K=K, seed=123)
        b1 = ds1.sample_batch(4)

        ds2 = SyntheticBigramDataset(pi_u, pi_b, T=T, K=K, seed=123)
        b2 = ds2.sample_batch(4)

        self.assertTrue(torch.equal(b1["seqs"], b2["seqs"]))

        m1 = SimplifiedTransformer(d=d, N=N, T=T, seed=999)
        m2 = SimplifiedTransformer(d=d, N=N, T=T, seed=999)

        _, (a1_1, a2_1) = m1(b1["seqs"], return_attention_weights=True)
        _, (a1_2, a2_2) = m2(b2["seqs"], return_attention_weights=True)

        self.assertTrue(torch.equal(a1_1, a1_2))
        self.assertTrue(torch.equal(a2_1, a2_2))


if __name__ == "__main__":
    unittest.main()
