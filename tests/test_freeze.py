"""Tests for freeze experiments and SGD grad=None semantics."""

import unittest
import torch
import torch.nn.functional as F

from src.freeze_train import parse_freeze_targets
from src.model import SimplifiedTransformer


class TestFreeze(unittest.TestCase):
    def test_freeze_parameter_resolution(self):
        """Test that --freeze targets resolve to exact model parameters."""
        model = SimplifiedTransformer(d=16, N=10, T=20, seed=42, final_ffn=False)
        targets = parse_freeze_targets(model, "wo2,wk1")
        self.assertEqual(len(targets), 2)
        self.assertIs(targets[0], model.W_O2)
        self.assertIs(targets[1], model.W_K1)

        single = parse_freeze_targets(model, "wk2")
        self.assertEqual(len(single), 1)
        self.assertIs(single[0], model.W_K2)

        with self.assertRaises(ValueError):
            parse_freeze_targets(model, "invalid_param")

    def test_masked_loss_manual_cross_entropy(self):
        """Test masked loss equals manual double precision cross-entropy (1e-12)."""
        torch.manual_seed(123)
        B, T_minus_1, N = 4, 10, 8
        logits = torch.randn(B, T_minus_1, N, dtype=torch.float64)
        targets = torch.randint(0, N, (B, T_minus_1))
        mask = torch.rand(B, T_minus_1) > 0.5
        # Ensure at least some elements are selected
        mask[0, 0] = True

        # PyTorch cross entropy
        pt_loss = F.cross_entropy(logits[mask], targets[mask])

        # Manual float64 cross entropy
        selected_logits = logits[mask]
        selected_targets = targets[mask]

        # Log-sum-exp
        max_val = selected_logits.max(dim=-1, keepdim=True).values
        exp_logits = torch.exp(selected_logits - max_val)
        sum_exp = exp_logits.sum(dim=-1)
        log_prob = (selected_logits - max_val) - torch.log(sum_exp).unsqueeze(-1)

        target_log_probs = log_prob.gather(1, selected_targets.unsqueeze(1)).squeeze(1)
        manual_loss = -target_log_probs.mean()

        diff = (pt_loss - manual_loss).abs().item()
        self.assertLess(diff, 1e-12)

    def test_freeze_unfreeze_dynamics(self):
        """Verify frozen parameter stays bit-identical until boundary, then updates."""
        d, N, T = 16, 10, 20
        unfreeze_at = 10
        total_iters = 30

        torch.manual_seed(42)
        model = SimplifiedTransformer(d=d, N=N, T=T, seed=42, final_ffn=False)
        model.W_F.requires_grad_(False)

        frozen_params = parse_freeze_targets(model, "wo2")
        for p in frozen_params:
            p.requires_grad_(False)

        wo2_init = model.W_O2.clone().detach()
        wk2_init = model.W_K2.clone().detach()

        trainable_params = [model.W_K1, model.W_K2, model.W_O2]
        optimizer = torch.optim.SGD(trainable_params, lr=0.1, momentum=0.9, weight_decay=1e-4)

        for step in range(total_iters):
            if step == unfreeze_at:
                for p in frozen_params:
                    p.requires_grad_(True)

            seqs = torch.randint(0, N, (4, T))
            logits, _ = model(seqs)
            pred_logits = logits[:, :-1, :]
            targets = seqs[:, 1:]

            # Mask every 2nd position
            mask = torch.zeros((4, T - 1), dtype=torch.bool)
            mask[:, 1::2] = True

            loss = F.cross_entropy(pred_logits[mask], targets[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step < unfreeze_at:
                # W_O2 must remain bit-identical to init
                self.assertTrue(torch.equal(model.W_O2, wo2_init), f"W_O2 changed at step {step} before unfreeze!")
                # W_K2 is never frozen, must change
                self.assertFalse(torch.equal(model.W_K2, wk2_init), f"W_K2 did not update at step {step}!")
            else:
                # Post boundary, W_O2 must update
                pass

        # At the end of total_iters (step 30 > 10), W_O2 must have updated
        self.assertFalse(torch.equal(model.W_O2, wo2_init), "W_O2 did not update after unfreeze!")

    def test_masked_in_context_loss_matches_indexing(self):
        import torch
        import torch.nn.functional as F
        from src.freeze_train import masked_in_context_loss
        torch.manual_seed(0)
        B, T, N = 8, 12, 65
        for _ in range(20):
            logits = torch.randn(B, T, N, dtype=torch.float64, requires_grad=True)
            targets = torch.randint(0, N, (B, T))
            mask = torch.rand(B, T) > 0.6
            if mask.sum() == 0:
                continue
            ref = F.cross_entropy(logits[mask], targets[mask])
            fixed = masked_in_context_loss(logits, targets, mask)
            self.assertTrue(torch.allclose(ref, fixed, rtol=0, atol=1e-12))
            g_ref, = torch.autograd.grad(ref, logits, retain_graph=True)
            g_fix, = torch.autograd.grad(fixed, logits)
            self.assertTrue(torch.allclose(g_ref, g_fix, rtol=0, atol=1e-12))
        # masked-out non-finite values must not propagate
        logits = torch.randn(B, T, N, dtype=torch.float64, requires_grad=True)
        targets = torch.randint(0, N, (B, T))
        mask = torch.rand(B, T) > 0.6
        logits.data[~mask] = float("inf")
        ref = F.cross_entropy(logits[mask], targets[mask])
        fixed = masked_in_context_loss(logits, targets, mask)
        self.assertTrue(torch.isfinite(fixed))
        self.assertTrue(torch.allclose(ref, fixed, rtol=0, atol=1e-12))
        g_ref, = torch.autograd.grad(ref, logits, retain_graph=True)
        g_fix, = torch.autograd.grad(fixed, logits)
        self.assertTrue(torch.isfinite(g_fix).all().item())
        self.assertTrue(torch.allclose(g_ref, g_fix, rtol=0, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
