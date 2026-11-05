"""Tests for synthetic data generator encoding Gate A criteria."""

import unittest
import numpy as np
import torch
from src.data import load_corpus, get_vocab, estimate_distributions, SyntheticBigramDataset


class TestGeneratorGateA(unittest.TestCase):
    """Test suite encoding Gate A exact and statistical criteria."""

    @classmethod
    def setUpClass(cls):
        text = load_corpus("input.txt")
        cls.chars, cls.char_to_idx, cls.idx_to_char = get_vocab(text)
        cls.pi_u, cls.pi_b = estimate_distributions(text, cls.chars)
        cls.N = len(cls.chars)
        cls.T = 256
        cls.K = 3

        # Sample 8192 sequences with seed 0
        dataset = SyntheticBigramDataset(cls.pi_u, cls.pi_b, T=cls.T, K=cls.K, seed=0)
        batch = dataset.sample_batch(8192)
        cls.seqs = batch["seqs"].numpy()
        cls.triggers = batch["triggers"].numpy()
        cls.outputs = batch["outputs"].numpy()

    def test_a1_exact_trigger_output_transition(self):
        """A1: For every transition where previous token is a trigger, next token equals assigned output."""
        num_seqs = len(self.seqs)
        for s in range(num_seqs):
            trig_to_out = {self.triggers[s, k]: self.outputs[s, k] for k in range(self.K)}
            for t in range(1, self.T):
                prev = self.seqs[s, t - 1]
                if prev in trig_to_out:
                    self.assertEqual(
                        self.seqs[s, t],
                        trig_to_out[prev],
                        f"Mismatch in seq {s} at t={t}: prev={prev}, expected={trig_to_out[prev]}, got={self.seqs[s, t]}",
                    )

    def test_a2_statistical_non_trigger_and_marginal(self):
        """A2: Non-trigger transitions follow pi_b and first token follows pi_u within Bonferroni z=4.9 bound."""
        num_seqs = len(self.seqs)
        # First-token marginal
        first_tokens = self.seqs[:, 0]
        ft_counts = np.bincount(first_tokens, minlength=self.N)
        n_first = num_seqs

        max_dev = 0.0
        for j in range(self.N):
            p = self.pi_u[j]
            if n_first * p >= 25 and n_first * (1 - p) >= 25:
                p_hat = ft_counts[j] / n_first
                dev = abs(p_hat - p) / np.sqrt(p * (1 - p) / n_first)
                max_dev = max(max_dev, dev)
                self.assertLessEqual(
                    dev,
                    4.9,
                    f"First token marginal failed for token {j}: dev={dev:.3f} > 4.9",
                )

        # Transitions where previous token is not a trigger
        non_trig_transitions = []
        for s in range(num_seqs):
            trig_set = set(self.triggers[s])
            for t in range(1, self.T):
                prev = self.seqs[s, t - 1]
                if prev not in trig_set:
                    non_trig_transitions.append((prev, self.seqs[s, t]))

        nt_arr = np.array(non_trig_transitions, dtype=np.int64)
        trans_counts = np.zeros((self.N, self.N), dtype=np.float64)
        for i, j in nt_arr:
            trans_counts[i, j] += 1

        n_i_emp = trans_counts.sum(axis=1)
        for i in range(self.N):
            n_i_val = n_i_emp[i]
            for j in range(self.N):
                p = self.pi_b[i, j]
                if n_i_val * p >= 25 and n_i_val * (1 - p) >= 25:
                    p_hat = trans_counts[i, j] / n_i_val
                    dev = abs(p_hat - p) / np.sqrt(p * (1 - p) / n_i_val)
                    max_dev = max(max_dev, dev)
                    self.assertLessEqual(
                        dev,
                        4.9,
                        f"Transition ({i}, {j}) failed: dev={dev:.3f} > 4.9",
                    )

    def test_a3_statistical_trigger_output_uniformity(self):
        """A3: Assigned outputs for each trigger token follow uniform distribution within z=4.9 bound."""
        num_seqs = len(self.seqs)
        trig_assigned = [[] for _ in range(self.N)]
        for s in range(num_seqs):
            for k in range(self.K):
                q = self.triggers[s, k]
                o = self.outputs[s, k]
                trig_assigned[q].append(o)

        p_unif = 1.0 / self.N
        for q in range(self.N):
            outs = trig_assigned[q]
            n_q = len(outs)
            if n_q == 0:
                continue
            o_counts = np.bincount(outs, minlength=self.N)
            for j in range(self.N):
                if n_q * p_unif >= 25 and n_q * (1 - p_unif) >= 25:
                    p_hat = o_counts[j] / n_q
                    dev = abs(p_hat - p_unif) / np.sqrt(p_unif * (1 - p_unif) / n_q)
                    self.assertLessEqual(
                        dev,
                        4.9,
                        f"Trigger {q} output {j} failed: dev={dev:.3f} > 4.9",
                    )


if __name__ == "__main__":
    unittest.main()
