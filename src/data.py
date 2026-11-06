"""Synthetic data generator and distribution estimation for Birth of a Transformer reproduction."""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


def load_corpus(path: str = "input.txt") -> str:
    """Load text corpus from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_vocab(text: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Extract sorted character vocabulary of size N."""
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return chars, char_to_idx, idx_to_char


def estimate_distributions(text: str, chars: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate unigram (pi_u) and bigram (pi_b) distributions with Laplace (add-one) smoothing.
    
    Formula:
        pi_b(j|i) = (count(i, j) + 1) / (n_i + N)
        pi_u(j) = (count(j) + 1) / (n + N)
    where N = len(chars) = 65.
    """
    N = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    indices = np.array([char_to_idx[c] for c in text], dtype=np.int64)
    n = len(indices)

    # Unigram with Laplace smoothing
    u_counts = np.bincount(indices, minlength=N)
    pi_u = (u_counts + 1.0) / (n + N)

    # Bigram with Laplace smoothing
    b_counts = np.zeros((N, N), dtype=np.float64)
    for i, j in zip(indices[:-1], indices[1:]):
        b_counts[i, j] += 1
    n_i = b_counts.sum(axis=1, keepdims=True)
    pi_b = (b_counts + 1.0) / (n_i + N)

    return pi_u, pi_b


def estimate_distributions_raw(text: str, chars: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate empirical unigram (pi_u) and bigram (pi_b) distributions using raw MLE counts.
    
    Formula:
        pi_u(j) = count(j) / n
        pi_b(j|i) = count(i, j) / n_i
    Returns float64 arrays with exact row normalization (no smoothing).
    """
    N = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    indices = np.array([char_to_idx[c] for c in text], dtype=np.int64)
    n = len(indices)

    # Raw unigram counts and MLE
    u_counts = np.bincount(indices, minlength=N).astype(np.float64)
    pi_u = u_counts / float(n)
    assert np.all(pi_u > 0.0), "All vocabulary tokens must appear at least once in corpus"
    assert np.isclose(np.sum(pi_u), 1.0, atol=1e-12), "pi_u must sum to 1.0"

    # Raw bigram counts and MLE
    pair_counts = np.zeros((N, N), dtype=np.float64)
    np.add.at(pair_counts, (indices[:-1], indices[1:]), 1.0)
    row_sums = pair_counts.sum(axis=1)

    for i in range(N):
        assert row_sums[i] > 0.0, f"Token {chars[i]!r} has zero row sum"
        assert np.any(pair_counts[i] > 0.0), f"Token {chars[i]!r} has no positive transition"

    pi_b = pair_counts / row_sums[:, np.newaxis]
    for i in range(N):
        assert np.isclose(np.sum(pi_b[i]), 1.0, atol=1e-12), f"Row {i} must sum to 1.0"

    return pi_u, pi_b


class SyntheticBigramDataset:
    """Synthetic dataset generating sequences with random triggers and deterministic in-context outputs."""

    def __init__(
        self,
        pi_u: np.ndarray,
        pi_b: np.ndarray,
        T: int = 256,
        K: int = 3,
        fixed_triggers: Optional[List[int]] = None,
        pi_o: str = "uniform",
        seed: Optional[int] = None,
    ):
        self.pi_u = pi_u
        self.pi_b = pi_b
        self.N = len(pi_u)
        self.T = T
        self.fixed_triggers = list(fixed_triggers) if fixed_triggers is not None else None
        self.K = len(self.fixed_triggers) if self.fixed_triggers is not None else K
        self.pi_o = pi_o
        self.rng = np.random.default_rng(seed)
        self.pi_u_cdf = np.cumsum(pi_u)
        self.pi_b_cdf = np.cumsum(pi_b, axis=1)

    def sample_batch(
        self,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample a batch of sequences.
        
        Returns:
            dict containing:
                - 'seqs': (B, T) LongTensor of token IDs
                - 'triggers': (B, K) LongTensor of trigger tokens
                - 'outputs': (B, K) LongTensor of assigned output tokens
                - 'in_context_mask': (B, T - 1) BoolTensor (triggers at >= 2nd occurrence)
                - 'global_mask': (B, T - 1) BoolTensor (non-trigger tokens)
        """
        r = self.rng if rng is None else rng
        B = batch_size
        T = self.T
        K = self.K
        N = self.N

        trigs = np.zeros((B, K), dtype=np.int64)
        outs = np.zeros((B, K), dtype=np.int64)

        if self.fixed_triggers is not None:
            trigs[:] = self.fixed_triggers
            if self.pi_o == "bigram":
                for b in range(B):
                    for k in range(K):
                        outs[b, k] = r.choice(N, p=self.pi_b[trigs[b, k]])
            else:
                for b in range(B):
                    outs[b] = r.choice(N, size=K, replace=True)
        else:
            if self.pi_o == "bigram":
                for b in range(B):
                    trigs[b] = r.choice(N, size=K, replace=False, p=self.pi_u)
                    for k in range(K):
                        outs[b, k] = r.choice(N, p=self.pi_b[trigs[b, k]])
            else:
                for b in range(B):
                    trigs[b] = r.choice(N, size=K, replace=False, p=self.pi_u)
                    outs[b] = r.choice(N, size=K, replace=True)

        rand_u = r.random((B, T))
        seqs = np.zeros((B, T), dtype=np.int64)
        seqs[:, 0] = np.searchsorted(self.pi_u_cdf, rand_u[:, 0])

        for t in range(1, T):
            prev = seqs[:, t - 1]
            is_trig = np.zeros(B, dtype=bool)
            for k in range(K):
                mask = (prev == trigs[:, k])
                seqs[mask, t] = outs[mask, k]
                is_trig |= mask

            non_trig_idx = np.where(~is_trig)[0]
            if len(non_trig_idx) > 0:
                prev_non = prev[non_trig_idx]
                rand_val = rand_u[non_trig_idx, t]
                cdfs = self.pi_b_cdf[prev_non]
                seqs[non_trig_idx, t] = (cdfs < rand_val[:, None]).sum(axis=1)

        # Build in-context and global masks for transitions t -> t+1 (positions 0 to T-2)
        in_context_mask = np.zeros((B, T - 1), dtype=bool)
        global_mask = np.zeros((B, T - 1), dtype=bool)

        for b in range(B):
            trig_set = set(trigs[b])
            counts = {trig: 0 for trig in trig_set}
            for t in range(T - 1):
                tok = seqs[b, t]
                if tok in counts:
                    counts[tok] += 1
                    if counts[tok] >= 2:
                        in_context_mask[b, t] = True
                else:
                    global_mask[b, t] = True

        return {
            "seqs": torch.from_numpy(seqs),
            "triggers": torch.from_numpy(trigs),
            "outputs": torch.from_numpy(outs),
            "in_context_mask": torch.from_numpy(in_context_mask),
            "global_mask": torch.from_numpy(global_mask),
        }
