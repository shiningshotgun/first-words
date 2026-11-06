"""Simplified two-layer Transformer model as specified in Bietti et al. (NeurIPS 2023)."""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedTransformer(nn.Module):
    """Two-layer transformer with W_Q = Id, linear FF layer, and frozen embeddings/value matrices."""

    def __init__(
        self,
        d: int = 128,
        N: int = 65,
        T: int = 256,
        seed: Optional[int] = None,
        final_ffn: bool = True,
    ):
        super().__init__()
        self.d = d
        self.N = N
        self.T = T
        self.final_ffn = final_ffn

        if seed is not None:
            torch.manual_seed(seed)

        # Variance 1/d -> std 1/sqrt(d)
        scale = 1.0 / np.sqrt(d)

        # Frozen embeddings and projection matrices
        # w_E: (N, d) where row j is w_E(j)^T
        self.register_buffer("w_E", torch.randn(N, d) * scale)
        # p: (T, d) where row t is p_t^T
        self.register_buffer("p", torch.randn(T, d) * scale)
        # w_U: (N, d) where row j is w_U(j)^T
        self.register_buffer("w_U", torch.randn(N, d) * scale)

        # Frozen attention projection matrices
        self.register_buffer("W_V1", torch.randn(d, d) * scale)
        self.register_buffer("W_O1", torch.randn(d, d) * scale)
        self.register_buffer("W_V2", torch.randn(d, d) * scale)

        # Trained matrices: initialized Gaussian with variance 1/d (stated choice)
        self.W_K1 = nn.Parameter(torch.randn(d, d) * scale)
        self.W_K2 = nn.Parameter(torch.randn(d, d) * scale)
        self.W_O2 = nn.Parameter(torch.randn(d, d) * scale)
        self.W_F = nn.Parameter(torch.randn(d, d) * scale)

        # Causal mask: upper triangular positions s > t get -inf
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(
        self,
        z: torch.Tensor,
        disable_attn: bool = False,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass.
        
        Args:
            z: (B, T) LongTensor of token IDs.
            disable_attn: if True, disables attention output (residual baseline).
            return_attention_weights: whether to return attention matrices (A1, A2).
            
        Returns:
            logits: (B, T, N) predictions for next token.
            (A1, A2): optional tuple of attention weights if requested.
        """
        B, T = z.shape
        # Input composition (eq 1): x_t = w_E(z_t) + p_t
        X0 = F.embedding(z, self.w_E) + self.p[:T].unsqueeze(0)

        if disable_attn:
            X2 = X0
            A1, A2 = None, None
        else:
            # Layer 1 Attention (eq 2 with W_Q = Id)
            # scores: x_t^T W_K1 x_s -> (X0 @ W_K1) @ X0^T
            scores1 = torch.matmul(torch.matmul(X0, self.W_K1), X0.transpose(-1, -2))
            A1 = F.softmax(scores1 + self.causal_mask[:T, :T], dim=-1)
            V1 = torch.matmul(X0, self.W_V1.t())
            attn1_out = torch.matmul(torch.matmul(A1, V1), self.W_O1.t())
            X1 = X0 + attn1_out

            # Layer 2 Attention (eq 2 with W_Q = Id)
            scores2 = torch.matmul(torch.matmul(X1, self.W_K2), X1.transpose(-1, -2))
            A2 = F.softmax(scores2 + self.causal_mask[:T, :T], dim=-1)
            V2 = torch.matmul(X1, self.W_V2.t())
            attn2_out = torch.matmul(torch.matmul(A2, V2), self.W_O2.t())
            X2 = X1 + attn2_out

        # Linear feedforward layer (eq 2 / Section 4.2)
        if self.final_ffn:
            ff_out = torch.matmul(X2, self.W_F.t())
            X3 = X2 + ff_out
        else:
            X3 = X2

        # Unembedding: logits = X3 @ w_U^T
        logits = torch.matmul(X3, self.w_U.t())

        if return_attention_weights:
            return logits, (A1, A2)
        return logits, None


def build_constructed_solution(
    model: SimplifiedTransformer,
    pi_b: np.ndarray,
    scale_k1: float = 1.0,
    scale_k2: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Construct analytic solution from eq 7 and eq 8 using the model's frozen weights.
    
    W_K1* = sum_{t=2}^T p_t p_{t-1}^T
    W_K2* = sum_{k in Q} w_E(k) (W_O1 W_V1 w_E(k))^T  (Q = all N tokens for random triggers)
    W_O2* = sum_{k=1}^N w_U(k) (W_V2 w_E(k))^T
    W_F*  = sum_{i=1}^N sum_{j=1}^N log pi_b(j|i) w_U(j) w_E(i)^T
    """
    d = model.d
    N = model.N
    T = model.T
    p = model.p
    w_E = model.w_E
    w_U = model.w_U
    W_V1 = model.W_V1
    W_O1 = model.W_O1
    W_V2 = model.W_V2

    device = p.device

    # W_K1*: sum_{t=1}^{T-1} p[t] p[t-1]^T (0-indexed)
    W_K1_star = torch.zeros(d, d, device=device)
    for t in range(1, T):
        W_K1_star += torch.outer(p[t], p[t - 1])
    W_K1_star = W_K1_star * scale_k1

    # W_K2*: sum_{k=0}^{N-1} w_E[k] (W_O1 W_V1 w_E[k])^T
    W_K2_star = torch.zeros(d, d, device=device)
    M1 = torch.matmul(W_O1, W_V1)
    for k in range(N):
        remapped = torch.matmul(M1, w_E[k])
        W_K2_star += torch.outer(w_E[k], remapped)
    W_K2_star = W_K2_star * scale_k2

    # W_O2*: sum_{k=0}^{N-1} w_U[k] (W_V2 w_E[k])^T
    W_O2_star = torch.zeros(d, d, device=device)
    for k in range(N):
        remapped = torch.matmul(W_V2, w_E[k])
        W_O2_star += torch.outer(w_U[k], remapped)

    # W_F*: sum_{i, j} log pi_b[i, j] w_U[j] w_E[i]^T
    log_pi_b = torch.tensor(np.log(pi_b), dtype=torch.float32, device=device)
    # w_U: (N, d), w_E: (N, d). outer(w_U[j], w_E[i]) entry (a, b) is w_U[j, a] w_E[i, b].
    # W_F* = w_U.t() @ log_pi_b.t() @ w_E
    W_F_star = torch.matmul(torch.matmul(w_U.t(), log_pi_b.t()), w_E)

    return {
        "W_K1": W_K1_star,
        "W_K2": W_K2_star,
        "W_O2": W_O2_star,
        "W_F": W_F_star,
    }
