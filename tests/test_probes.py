"""Unit tests for probe functions and probe extensions."""

import unittest
import torch
from src.data import load_corpus, get_vocab, estimate_distributions
from src.model import SimplifiedTransformer, build_constructed_solution
from src.probes import (
    compute_w_k1_recall,
    compute_w_k2_recall,
    compute_w_o2_recall,
    compute_w_f_kl,
    evaluate_all_probes,
)


class TestProbes(unittest.TestCase):
    """Test suite verifying probe behavior, regression, and subset_Q extension."""

    @classmethod
    def setUpClass(cls):
        text = load_corpus("input.txt")
        chars, _, _ = get_vocab(text)
        cls.pi_u, cls.pi_b = estimate_distributions(text, chars)
        cls.model = SimplifiedTransformer(d=128, N=65, T=256, seed=0)
        cls.sol = build_constructed_solution(cls.model, cls.pi_b)

    def test_probe_regression_on_construction(self):
        """Probe regression: outputs on session-1 construction match exact reference values."""
        r_k1 = compute_w_k1_recall(self.sol["W_K1"], self.model.p)
        r_k1_early = compute_w_k1_recall(self.sol["W_K1"], self.model.p, max_t=64)
        r_k2 = compute_w_k2_recall(self.sol["W_K2"], self.model.w_E, self.model.W_O1, self.model.W_V1)
        r_o2 = compute_w_o2_recall(self.sol["W_O2"], self.model.w_E, self.model.w_U, self.model.W_V2)
        kl_f = compute_w_f_kl(self.sol["W_F"], self.model.w_E, self.model.w_U, self.pi_b)

        self.assertAlmostEqual(r_k1, 1.0, places=5)
        self.assertAlmostEqual(r_k1_early, 1.0, places=5)
        self.assertAlmostEqual(r_k2, 1.0, places=5)
        self.assertAlmostEqual(r_o2, 1.0, places=5)
        self.assertAlmostEqual(kl_f, 2.6113181114196777, places=4)

    def test_w_k2_recall_subset_q(self):
        """W_K2 recall subset_Q: evaluates recall restricted to a trigger subset Q."""
        # On constructed W_K2*, recall on any subset Q must be 1.0
        subset_Q = [1, 46, 57]
        r_q = compute_w_k2_recall(
            self.sol["W_K2"],
            self.model.w_E,
            self.model.W_O1,
            self.model.W_V1,
            subset_Q=subset_Q,
        )
        self.assertAlmostEqual(r_q, 1.0, places=5)

        # On untrained random weights, recall is near chance (1/65)
        r_untrained = compute_w_k2_recall(
            self.model.W_K2,
            self.model.w_E,
            self.model.W_O1,
            self.model.W_V1,
            subset_Q=subset_Q,
        )
        self.assertLess(r_untrained, 0.5)

        # evaluate_all_probes with fixed_triggers
        probes = evaluate_all_probes(self.model, self.pi_b, fixed_triggers=subset_Q)
        self.assertIn("recall_W_K2_Q", probes)
        self.assertEqual(probes["recall_W_K2_Q"], r_untrained)


if __name__ == "__main__":
    unittest.main()
