"""Probes for memory recall (eq 9) and KL divergence (eq 10)."""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from src.model import SimplifiedTransformer


def compute_w_k1_recall(
    W_K1: torch.Tensor,
    p: torch.Tensor,
    max_t: Optional[int] = None,
) -> float:
    """Compute recall probe (eq 9) for previous token head W_K1 against target sum_{t=2}^T p_t p_{t-1}^T.
    
    Target memory: W* = sum_{t=2}^T v_t u_t^T where u_t = p_{t-1}, v_t = p_t.
    Probe: 1/|M| sum_{(i,j)} 1{argmax_{j'} v_{j'}^T W u_i = j}.
    If max_t is specified (e.g. 64), only evaluate for t < max_t (positions 2 to max_t - 1 in 1-based,
    or 1 to max_t - 1 in 0-based).
    """
    T = p.shape[0]
    # S[j', t-1] = p[j']^T W_K1 p[t-1]
    # shape: (T, T)
    S = torch.matmul(torch.matmul(p, W_K1), p.t())

    end_t = T if max_t is None else min(T, max_t)
    # Queries correspond to p_{t-1} for t in 1..end_t-1
    queries = torch.arange(0, end_t - 1, device=p.device)
    targets = torch.arange(1, end_t, device=p.device)

    # argmax over all candidate positions j' in 0..T-1
    preds = S[:, queries].argmax(dim=0)
    correct = (preds == targets).float().sum().item()
    total = len(targets)
    return correct / total if total > 0 else 0.0


def compute_w_k2_recall(
    W_K2: torch.Tensor,
    w_E: torch.Tensor,
    W_O1: torch.Tensor,
    W_V1: torch.Tensor,
    subset_Q: Optional[List[int]] = None,
) -> float:
    """Compute recall probe (eq 9) for induction key-query matrix W_K2.
    
    Target memory: W* = sum_{k in Q} v_k u_k^T where u_k = W_O1 W_V1 w_E(k), v_k = w_E(k).
    Probe: 1/N sum_k 1{argmax_{k'} w_E(k')^T W u_k = k}.
    If subset_Q is provided, recall is computed only over keys k in Q (the paper notes
    this restriction is an easier task for fixed triggers).
    """
    N = w_E.shape[0]
    # u_k = W_O1 @ W_V1 @ w_E[k]  shape: (N, d)
    M1 = torch.matmul(W_O1, W_V1)
    u_k = torch.matmul(w_E, M1.t())

    # S[k', k] = w_E[k']^T W_K2 u_k
    S = torch.matmul(torch.matmul(w_E, W_K2), u_k.t())
    targets = torch.arange(N, device=w_E.device)
    preds = S.argmax(dim=0)

    if subset_Q is not None:
        Q_tensor = torch.tensor(subset_Q, dtype=torch.long, device=w_E.device)
        return (preds[Q_tensor] == targets[Q_tensor]).float().mean().item()
    return (preds == targets).float().mean().item()


def compute_w_o2_recall(
    W_O2: torch.Tensor,
    w_E: torch.Tensor,
    w_U: torch.Tensor,
    W_V2: torch.Tensor,
) -> float:
    """Compute recall probe (eq 9) for induction output matrix W_O2.
    
    Target memory: W* = sum_{k=1}^N v_k u_k^T where u_k = W_V2 w_E(k), v_k = w_U(k).
    Probe: 1/N sum_k 1{argmax_{k'} w_U(k')^T W u_k = k}.
    """
    N = w_E.shape[0]
    # u_k = W_V2 @ w_E[k]  shape: (N, d)
    u_k = torch.matmul(w_E, W_V2.t())

    # S[k', k] = w_U[k']^T W_O2 u_k
    S = torch.matmul(torch.matmul(w_U, W_O2), u_k.t())
    targets = torch.arange(N, device=w_E.device)
    preds = S.argmax(dim=0)
    return (preds == targets).float().mean().item()


def compute_w_f_kl(
    W_F: torch.Tensor,
    w_E: torch.Tensor,
    w_U: torch.Tensor,
    pi_b: np.ndarray,
) -> float:
    """Compute KL divergence probe (eq 10) for feed-forward layer W_F.
    
    d_KL(W_F, pi_b) = 1/N sum_k d_KL( softmax(W_U W_F w_E(k)), pi_b(.|k) ).
    Note: This is the paper-notation direction KL(model || pi_b), used only by
    our construction diagnostic instrument and legacy smoothed evaluations.
    """
    if W_F is None:
        return float("nan")
    device = W_F.device
    # logits[k, j] = w_U[j]^T W_F w_E[k]
    # w_E @ W_F^T @ w_U^T has shape (N, N)
    logits = torch.matmul(torch.matmul(w_E, W_F.t()), w_U.t())
    P = F.softmax(logits, dim=-1)
    log_P = F.log_softmax(logits, dim=-1)

    pi_b_t = torch.tensor(pi_b, dtype=torch.float32, device=device)
    log_pi_b = torch.log(pi_b_t)

    # d_KL(P, Q) = sum P * (log P - log Q)
    kl_per_token = (P * (log_P - log_pi_b)).sum(dim=-1)
    return kl_per_token.mean().item()


def compute_kl_pi_b_to_model(
    W_F: Optional[torch.Tensor],
    w_E: torch.Tensor,
    w_U: torch.Tensor,
    pi_b: np.ndarray,
) -> float:
    """Compute KL(pi_b(.|k) || softmax(W_U W_F w_E(k))) averaged over conditioning tokens k.
    
    Direction: KL(pi_b || model).
    Discrepancy: The paper text (Section 5, Eq 10) writes d_KL(softmax, pi_b);
    the authors' code (ihead_basic_main.py:105-115) computes F.kl_div(log_softmax(model), pi_b_rows),
    which is KL(pi_b || softmax). This repo follows the authors' code for raw-MLE training metrics.
    All computations are performed in float64.
    Entries where pi_b[k, j] == 0 contribute exactly 0.0 to the divergence.
    """
    if W_F is None:
        return float("nan")
    # Evaluate probe on CPU to support exact float64 arithmetic across backends (e.g. MPS float64 limits)
    cpu_dev = torch.device("cpu")
    W_F_d = W_F.detach().cpu().to(dtype=torch.float64)
    w_E_d = w_E.detach().cpu().to(dtype=torch.float64)
    w_U_d = w_U.detach().cpu().to(dtype=torch.float64)

    # Logits: w_E[k] @ W_F^T @ w_U^T has shape (N, N) where row k gives logits over next tokens j
    logits = torch.matmul(torch.matmul(w_E_d, W_F_d.t()), w_U_d.t())
    log_probs = F.log_softmax(logits, dim=-1)

    pi_b_t = torch.tensor(pi_b, dtype=torch.float64, device=cpu_dev)
    nonzero = pi_b_t > 0.0
    safe_target = torch.where(nonzero, pi_b_t, torch.ones_like(pi_b_t))
    kl_terms = torch.where(
        nonzero,
        pi_b_t * (torch.log(safe_target) - log_probs),
        torch.zeros_like(pi_b_t),
    )
    kl_per_k = kl_terms.sum(dim=-1)
    return float(kl_per_k.mean().item())


def evaluate_all_probes(
    model: SimplifiedTransformer,
    pi_b: np.ndarray,
    fixed_triggers: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Evaluate all memory recall and KL probes on the current model parameters."""
    with torch.no_grad():
        w_k1_all = compute_w_k1_recall(model.W_K1, model.p)
        w_k1_early = compute_w_k1_recall(model.W_K1, model.p, max_t=64)
        w_k2 = compute_w_k2_recall(model.W_K2, model.w_E, model.W_O1, model.W_V1)
        w_o2 = compute_w_o2_recall(model.W_O2, model.w_E, model.w_U, model.W_V2)
        if getattr(model, "final_ffn", True) and model.W_F is not None:
            w_f_kl = compute_w_f_kl(model.W_F, model.w_E, model.w_U, pi_b)
        else:
            w_f_kl = None
        w_k2_q = (
            compute_w_k2_recall(model.W_K2, model.w_E, model.W_O1, model.W_V1, subset_Q=fixed_triggers)
            if fixed_triggers is not None
            else None
        )

    res = {
        "recall_W_K1": float(w_k1_all),
        "recall_W_K1_early": float(w_k1_early),
        "recall_W_K2": float(w_k2),
        "recall_W_O2": float(w_o2),
    }
    if w_f_kl is not None:
        res["kl_W_F"] = float(w_f_kl)
    if w_k2_q is not None:
        res["recall_W_K2_Q"] = float(w_k2_q)
    return res
