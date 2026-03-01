from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .c2fab_math import causal_iir_filter_parallel
except ImportError:
    from c2fab_math import causal_iir_filter_parallel


class C2FAB_Heads(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 4096,
        D: int = 8,
        num_layers: int = 4,
        num_q_heads: int = 32,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.D = D
        self.num_layers = num_layers
        self.num_q_heads = num_q_heads

        # charge_mlp: [..., hidden_dim] -> [..., D]
        self.charge_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, D),
            nn.ReLU(),  # Critical: enforces non-negative sparse charge activations.
        )
        # receptor_mlp: [..., hidden_dim] -> [..., 2*D]
        self.receptor_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2 * D),
        )

        self.lambdas_fast = nn.Parameter(torch.ones(D) * 0.7)
        self.lambdas_slow = nn.Parameter(torch.ones(D) * 0.99)
        # Zero-init keeps injected bias as a no-op at step zero.
        self.alphas = nn.Parameter(torch.zeros(num_layers, num_q_heads))

    def forward(
        self,
        x_u: torch.Tensor,
        x_q: torch.Tensor,
        use_bidirectional: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_u: [batch_size, seq_len, hidden_dim]
            x_q: [batch_size, query_len, hidden_dim]
            use_bidirectional: if True, add backward IIR fields.
        Returns:
            Phi_total: [batch_size, seq_len, 2*D]
            R_q: [batch_size, query_len, 2*D]
            C_u: [batch_size, seq_len, D] (raw charge activations)
        """
        if x_u.ndim != 3 or x_q.ndim != 3:
            raise ValueError(
                f"x_u and x_q must be rank-3 tensors, got {x_u.ndim=} and {x_q.ndim=}."
            )
        if x_u.shape[0] != x_q.shape[0]:
            raise ValueError(
                f"Batch mismatch between x_u and x_q: {x_u.shape[0]} vs {x_q.shape[0]}."
            )
        if x_u.shape[-1] != self.hidden_dim or x_q.shape[-1] != self.hidden_dim:
            raise ValueError(
                "Hidden dimension mismatch: "
                f"expected {self.hidden_dim}, got x_u={x_u.shape[-1]}, x_q={x_q.shape[-1]}."
            )

        C_u = self.charge_mlp(x_u)  # C_u:[batch_size, seq_len, D]
        lambdas_fast = self.lambdas_fast.clamp(min=1.0e-4, max=1.0 - 1.0e-4)
        lambdas_slow = self.lambdas_slow.clamp(min=1.0e-4, max=1.0 - 1.0e-4)

        Phi_fwd_fast = causal_iir_filter_parallel(
            C_u,
            lambdas_fast,
            time_dim=1,
        )  # [batch_size, seq_len, D]
        Phi_fwd_slow = causal_iir_filter_parallel(
            C_u,
            lambdas_slow,
            time_dim=1,
        )  # [batch_size, seq_len, D]

        if use_bidirectional:
            C_u_reversed = torch.flip(C_u, dims=[1])

            Phi_bwd_fast = causal_iir_filter_parallel(
                C_u_reversed,
                lambdas_fast,
                time_dim=1,
            )
            Phi_bwd_fast = torch.flip(Phi_bwd_fast, dims=[1])

            Phi_bwd_slow = causal_iir_filter_parallel(
                C_u_reversed,
                lambdas_slow,
                time_dim=1,
            )
            Phi_bwd_slow = torch.flip(Phi_bwd_slow, dims=[1])

            Phi_fast = Phi_fwd_fast + Phi_bwd_fast
            Phi_slow = Phi_fwd_slow + Phi_bwd_slow
        else:
            Phi_fast = Phi_fwd_fast
            Phi_slow = Phi_fwd_slow

        Phi_total = torch.cat([Phi_fast, Phi_slow], dim=-1)  # [batch_size, seq_len, 2*D]
        R_q = self.receptor_mlp(x_q)  # [batch_size, query_len, 2*D]

        return Phi_total, R_q, C_u


def compute_bias_scores(
    Phi_u: torch.Tensor,
    R_q: torch.Tensor,
) -> torch.Tensor:
    """
    Compute s_{q,u} = R_q · Phi_u for all query/key positions.

    Args:
        Phi_u: [batch_size, kv_len, feature_dim]
        R_q: [batch_size, q_len, feature_dim]
    Returns:
        bias_scores: [batch_size, q_len, kv_len]
    """
    if Phi_u.ndim != 3 or R_q.ndim != 3:
        raise ValueError(
            f"Phi_u and R_q must be rank-3 tensors, got {Phi_u.ndim=} and {R_q.ndim=}."
        )
    if Phi_u.shape[0] != R_q.shape[0]:
        raise ValueError(
            f"Batch mismatch between Phi_u and R_q: {Phi_u.shape[0]} vs {R_q.shape[0]}."
        )
    if Phi_u.shape[-1] != R_q.shape[-1]:
        raise ValueError(
            f"Feature dim mismatch between Phi_u and R_q: {Phi_u.shape[-1]} vs {R_q.shape[-1]}."
        )
    return torch.einsum("bqd,bud->bqu", R_q, Phi_u)


def infonce_loss(
    bias_scores: torch.Tensor,
    evidence_mask: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """
    InfoNCE using numerically stable logsumexp.

    Args:
        bias_scores: [batch_size, seq_len] or [batch_size, q_len, seq_len]
        evidence_mask: [batch_size, seq_len], binary {0,1}
        tau: temperature scalar
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}.")
    if evidence_mask.ndim != 2:
        raise ValueError(
            f"evidence_mask must have shape [batch_size, seq_len], got {tuple(evidence_mask.shape)}."
        )
    if bias_scores.ndim not in (2, 3):
        raise ValueError(
            f"bias_scores must be rank-2 or rank-3, got shape {tuple(bias_scores.shape)}."
        )
    if bias_scores.shape[0] != evidence_mask.shape[0]:
        raise ValueError(
            f"Batch mismatch: bias_scores {bias_scores.shape[0]} vs evidence_mask {evidence_mask.shape[0]}."
        )
    if bias_scores.shape[-1] != evidence_mask.shape[-1]:
        raise ValueError(
            f"Seq mismatch: bias_scores seq={bias_scores.shape[-1]} vs "
            f"evidence_mask seq={evidence_mask.shape[-1]}."
        )

    scores = bias_scores / tau
    evidence_bool = evidence_mask.to(dtype=torch.bool)
    if scores.ndim == 3:
        evidence_bool = evidence_bool.unsqueeze(1).expand(-1, scores.shape[1], -1)

    if torch.any(~evidence_bool.any(dim=-1)):
        raise ValueError("Each sample must contain at least one evidence token (mask == 1).")

    # numerator: logsumexp over only evidence tokens
    evidence_scores = scores.masked_fill(~evidence_bool, float("-inf"))
    numerator = torch.logsumexp(evidence_scores, dim=-1)

    # denominator: logsumexp over all tokens
    denominator = torch.logsumexp(scores, dim=-1)

    loss = (denominator - numerator).mean()
    return loss


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    device = _pick_device()
    torch.manual_seed(11)

    model = C2FAB_Heads(hidden_dim=4096, D=8, num_layers=4, num_q_heads=32).to(device)

    # x_u:[2, 3000, 4096], x_q:[2, 1, 4096]
    x_u = torch.randn(2, 3000, 4096, device=device)
    x_q = torch.randn(2, 1, 4096, device=device)

    # evidence_mask:[2, 3000], with one positive span per sample.
    evidence_mask = torch.zeros(2, 3000, dtype=torch.long, device=device)
    evidence_mask[0, 1500:1520] = 1
    evidence_mask[1, 2200:2225] = 1

    Phi_total, R_q, C_u = model(x_u, x_q)
    all_bias_scores = compute_bias_scores(Phi_u=Phi_total, R_q=R_q)  # [2, 1, 3000]
    bias_scores = all_bias_scores.squeeze(1)  # [2, 3000] since query_len == 1 in this smoke test
    loss = infonce_loss(bias_scores=bias_scores, evidence_mask=evidence_mask, tau=0.1)

    print(f"Phi_total shape: {tuple(Phi_total.shape)}")
    print(f"R_q shape: {tuple(R_q.shape)}")
    print(f"C_u shape: {tuple(C_u.shape)}")
    print(f"all_bias_scores shape: {tuple(all_bias_scores.shape)}")
    print(f"bias_scores shape: {tuple(bias_scores.shape)}")
    print(f"InfoNCE loss: {loss.item():.6f}")
