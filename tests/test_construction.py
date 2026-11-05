"""Tests for model construction and structural properties encoding Gate B1 criteria."""

import unittest
import numpy as np
import torch
import torch.nn.functional as F
from src.data import load_corpus, get_vocab, estimate_distributions, SyntheticBigramDataset
from src.model import SimplifiedTransformer, build_constructed_solution


class TestConstructionGateB1(unittest.TestCase):
    """Test suite encoding Gate B1 structural checks on CPU."""

    @classmethod
    def setUpClass(cls):
        text = load_corpus("input.txt")
        cls.chars, cls.char_to_idx, cls.idx_to_char = get_vocab(text)
        cls.pi_u, cls.pi_b = estimate_distributions(text, cls.chars)
        cls.d = 128
        cls.N = len(cls.chars)
        cls.T = 256
        cls.K = 3

    def test_b1_i_causal_masking(self):
        """B1 (i): Causal masking: attention weights exactly zero for key positions after query position."""
        model = SimplifiedTransformer(d=self.d, N=self.N, T=self.T, seed=0)
        z = torch.randint(0, self.N, (4, self.T))
        _, (A1, A2) = model(z, return_attention_weights=True)

        for name, A in [("A1", A1), ("A2", A2)]:
            # Upper triangle strictly above diagonal must be exactly 0.0
            for t in range(self.T - 1):
                future_weights = A[:, t, t + 1 :]
                self.assertTrue(
                    torch.all(future_weights == 0.0),
                    f"{name} has non-zero weights in future positions at query position {t}: max={future_weights.max().item()}",
                )

    def test_b1_ii_frozen_parameters(self):
        """B1 (ii): Frozen parameters bit-identical before and after one training step, trained matrices changed."""
        model = SimplifiedTransformer(d=self.d, N=self.N, T=self.T, seed=0)
        optimizer = torch.optim.SGD(
            [model.W_K1, model.W_K2, model.W_O2, model.W_F],
            lr=1.0,
            momentum=0.9,
            weight_decay=1e-4,
        )

        # Snapshot frozen and trained parameters
        frozen_before = {
            "w_E": model.w_E.clone(),
            "p": model.p.clone(),
            "w_U": model.w_U.clone(),
            "W_V1": model.W_V1.clone(),
            "W_O1": model.W_O1.clone(),
            "W_V2": model.W_V2.clone(),
        }
        trained_before = {
            "W_K1": model.W_K1.clone(),
            "W_K2": model.W_K2.clone(),
            "W_O2": model.W_O2.clone(),
            "W_F": model.W_F.clone(),
        }

        # One real training step
        dataset = SyntheticBigramDataset(self.pi_u, self.pi_b, T=self.T, K=self.K, seed=0)
        batch = dataset.sample_batch(16)
        z = batch["seqs"]
        optimizer.zero_grad()
        logits, _ = model(z)
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, self.N), z[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

        # Check frozen are bit-identical
        for name, tensor_before in frozen_before.items():
            tensor_after = getattr(model, name)
            self.assertTrue(
                torch.equal(tensor_before, tensor_after),
                f"Frozen parameter {name} changed during training step!",
            )

        # Check trained matrices changed
        for name, tensor_before in trained_before.items():
            tensor_after = getattr(model, name)
            self.assertFalse(
                torch.equal(tensor_before, tensor_after),
                f"Trained parameter {name} did not change during training step!",
            )

    def test_b1_iii_determinism_on_cpu(self):
        """B1 (iii): Determinism: two 5-iteration training runs from seed 0 on CPU produce bit-identical losses."""
        def run_5_steps():
            torch.manual_seed(0)
            model = SimplifiedTransformer(d=self.d, N=self.N, T=self.T, seed=0)
            optimizer = torch.optim.SGD(
                [model.W_K1, model.W_K2, model.W_O2, model.W_F],
                lr=1.0,
                momentum=0.9,
                weight_decay=1e-4,
            )
            dataset = SyntheticBigramDataset(self.pi_u, self.pi_b, T=self.T, K=self.K, seed=0)
            losses = []
            for _ in range(5):
                batch = dataset.sample_batch(32)
                z = batch["seqs"]
                optimizer.zero_grad()
                logits, _ = model(z)
                loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, self.N), z[:, 1:].reshape(-1))
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            return losses

        run1 = run_5_steps()
        run2 = run_5_steps()
        self.assertEqual(run1, run2, f"Losses not bit-identical across runs:\nRun1: {run1}\nRun2: {run2}")

    def test_b1_iv_finiteness(self):
        """B1 (iv): Finiteness: every constructed matrix and parameter/loss is finite (no NaN or Inf)."""
        model = SimplifiedTransformer(d=self.d, N=self.N, T=self.T, seed=0)
        sol = build_constructed_solution(model, self.pi_b)

        # Constructed matrices
        for name, matrix in sol.items():
            self.assertTrue(
                torch.all(torch.isfinite(matrix)),
                f"Constructed matrix {name} has non-finite values!",
            )

        # Initial parameters
        for name, param in model.named_parameters():
            self.assertTrue(
                torch.all(torch.isfinite(param)),
                f"Parameter {name} has non-finite values at init!",
            )
        for name, buf in model.named_buffers():
            if name != "causal_mask":
                self.assertTrue(
                    torch.all(torch.isfinite(buf)),
                    f"Buffer {name} has non-finite values at init!",
                )

        # After one training step
        optimizer = torch.optim.SGD(
            [model.W_K1, model.W_K2, model.W_O2, model.W_F],
            lr=1.0,
            momentum=0.9,
            weight_decay=1e-4,
        )
        dataset = SyntheticBigramDataset(self.pi_u, self.pi_b, T=self.T, K=self.K, seed=0)
        batch = dataset.sample_batch(32)
        z = batch["seqs"]
        optimizer.zero_grad()
        logits, _ = model(z)
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, self.N), z[:, 1:].reshape(-1))
        self.assertTrue(torch.isfinite(loss), f"Loss is non-finite: {loss.item()}")
        loss.backward()
        optimizer.step()

        for name, param in model.named_parameters():
            self.assertTrue(
                torch.all(torch.isfinite(param)),
                f"Parameter {name} has non-finite values after training step!",
            )


if __name__ == "__main__":
    unittest.main()
