"""Tests for raw-MLE distribution estimation and dataset properties."""

import hashlib
import unittest
import numpy as np
import torch

from src.data import (
    load_corpus,
    get_vocab,
    estimate_distributions_raw,
    SyntheticBigramDataset,
)


class TestRawMLE(unittest.TestCase):
    """Test suite for raw maximum likelihood estimation (Change 1 & Tests T1-T6)."""

    def test_t1_exact_mle_ratios_synthetic(self):
        """T1: Exact MLE ratios on a hardcoded synthetic corpus, unobserved transitions are 0.0."""
        text = "abacab"
        chars = ["a", "b", "c"]
        pi_u, pi_b = estimate_distributions_raw(text, chars)

        # n = 6 tokens: a:3, b:2, c:1
        self.assertAlmostEqual(pi_u[0], 3.0 / 6.0, delta=1e-15)
        self.assertAlmostEqual(pi_u[1], 2.0 / 6.0, delta=1e-15)
        self.assertAlmostEqual(pi_u[2], 1.0 / 6.0, delta=1e-15)

        # Transitions in "abacab": (a,b), (b,a), (a,c), (c,a), (a,b)
        # From a (3 transitions): b:2, c:1 -> pi_b[0] = [0.0, 2/3, 1/3]
        # From b (1 transition): a:1 -> pi_b[1] = [1.0, 0.0, 0.0]
        # From c (1 transition): a:1 -> pi_b[2] = [1.0, 0.0, 0.0]
        self.assertEqual(pi_b[0, 0], 0.0)
        self.assertAlmostEqual(pi_b[0, 1], 2.0 / 3.0, delta=1e-15)
        self.assertAlmostEqual(pi_b[0, 2], 1.0 / 3.0, delta=1e-15)

        self.assertAlmostEqual(pi_b[1, 0], 1.0, delta=1e-15)
        self.assertEqual(pi_b[1, 1], 0.0)
        self.assertEqual(pi_b[1, 2], 0.0)

        self.assertAlmostEqual(pi_b[2, 0], 1.0, delta=1e-15)
        self.assertEqual(pi_b[2, 1], 0.0)
        self.assertEqual(pi_b[2, 2], 0.0)

        for i in range(3):
            self.assertAlmostEqual(float(np.sum(pi_b[i])), 1.0, delta=1e-15)

    def test_t3_in_context_mask_vs_independent_recompute(self):
        """T3: In-context mask matches independent recomputation exactly."""
        text = load_corpus("input.txt")
        chars, _, _ = get_vocab(text)
        pi_u, pi_b = estimate_distributions_raw(text, chars)

        ds = SyntheticBigramDataset(pi_u, pi_b, T=64, K=3, seed=42)
        batch = ds.sample_batch(16)
        seqs = batch["seqs"].numpy()
        trigs = batch["triggers"].numpy()
        mask = batch["in_context_mask"].numpy()

        B, T = seqs.shape
        expected_mask = np.zeros((B, T - 1), dtype=bool)
        for b in range(B):
            trig_set = set(trigs[b])
            counts = {trig: 0 for trig in trig_set}
            for t in range(T - 1):
                tok = seqs[b, t]
                if tok in trig_set:
                    counts[tok] += 1
                    if counts[tok] >= 2:
                        expected_mask[b, t] = True

        np.testing.assert_array_equal(mask, expected_mask)

    def test_t4_real_corpus_properties(self):
        """T4: On real corpus, pi_u sums to 1.0, all entries > 0, pi_b rows sum to 1.0 with positive entries."""
        text = load_corpus("input.txt")
        chars, _, _ = get_vocab(text)
        pi_u, pi_b = estimate_distributions_raw(text, chars)
        N = len(chars)

        self.assertAlmostEqual(float(np.sum(pi_u)), 1.0, delta=1e-12)
        self.assertTrue(np.all(pi_u > 0.0))

        for i in range(N):
            self.assertAlmostEqual(float(np.sum(pi_b[i])), 1.0, delta=1e-12)
            self.assertTrue(np.any(pi_b[i] > 0.0))

    def test_t5_regression_pin_raw_dataset(self):
        """T5: Raw-path regression pin for 512 sequences with seed 0."""
        text = load_corpus("input.txt")
        chars, _, _ = get_vocab(text)
        pi_u, pi_b = estimate_distributions_raw(text, chars)

        ds = SyntheticBigramDataset(pi_u, pi_b, T=256, K=3, seed=0)
        batch = ds.sample_batch(512)
        seq_bytes = batch["seqs"].numpy().tobytes()
        h = hashlib.sha256(seq_bytes).hexdigest()

        expected = "9f50fad57ed91ee8d39bf8fc00dc1b1316bfdabbd2afa9ea227ce486a26f9b8f"
        self.assertEqual(h, expected)

    def test_t6_raw_transition_support_synthetic(self):
        """T6: Each pi_b row's positive set equals the observed successor set exactly (no smoothing mass)."""
        text = "abacab"
        chars = ["a", "b", "c"]
        _, pi_b = estimate_distributions_raw(text, chars)

        # Observed successors for a (index 0): {b (1), c (2)}
        pos_0 = set(np.where(pi_b[0] > 0.0)[0])
        self.assertEqual(pos_0, {1, 2})
        self.assertEqual(pi_b[0, 0], 0.0)

        # Observed successors for b (index 1): {a (0)}
        pos_1 = set(np.where(pi_b[1] > 0.0)[0])
        self.assertEqual(pos_1, {0})
        self.assertEqual(pi_b[1, 1], 0.0)
        self.assertEqual(pi_b[1, 2], 0.0)

        # Observed successors for c (index 2): {a (0)}
        pos_2 = set(np.where(pi_b[2] > 0.0)[0])
        self.assertEqual(pos_2, {0})
        self.assertEqual(pi_b[2, 1], 0.0)
        self.assertEqual(pi_b[2, 2], 0.0)


if __name__ == "__main__":
    unittest.main()
