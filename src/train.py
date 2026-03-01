from __future__ import annotations

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ in (None, ""):
    # Supports direct script execution: `python src/train.py`
    from config import MODEL_ID
    from data_gen import generate_synthetic_example
    from modules import C2FAB_Heads, infonce_loss
else:
    # Supports module execution: `python -m src.train`
    from .config import MODEL_ID
    from .data_gen import generate_synthetic_example
    from .modules import C2FAB_Heads, infonce_loss


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _recall_at_k(
    bias_scores: torch.Tensor,
    evidence_mask: torch.Tensor,
    k: int = 32,
) -> float:
    """
    Args:
        bias_scores: [batch_size, seq_len]
        evidence_mask: [batch_size, seq_len] with binary {0,1}
    """
    topk = min(k, bias_scores.shape[-1])
    topk_indices = torch.topk(bias_scores, k=topk, dim=-1).indices
    evidence_hits = evidence_mask.gather(dim=-1, index=topk_indices) > 0
    return float(evidence_hits.any(dim=-1).float().mean().item())


def train_phase3(steps: int = 50) -> None:
    device = _pick_device()
    torch.manual_seed(42)

    print(f"Loading tokenizer/model: {MODEL_ID}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
        ).to(device)
    except Exception as exc:
        print(
            "Online Hugging Face load failed "
            f"({type(exc).__name__}: {exc}). "
            "Retrying with local cache only..."
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to(device)

    # Freeze base model completely.
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    heads = C2FAB_Heads(hidden_dim=4096, D=8, num_layers=4, num_q_heads=32).to(device)
    optimizer = AdamW(heads.parameters(), lr=1e-3)

    print(f"Starting Phase 3 training on device={device} for {steps} steps.")
    for step in range(1, steps + 1):
        input_ids, query_ids, evidence_mask = generate_synthetic_example(tokenizer)

        # input_ids:[ctx_len], query_ids:[query_len], evidence_mask:[ctx_len]
        input_ids = input_ids.to(device)
        query_ids = query_ids.to(device)
        evidence_mask = evidence_mask.to(device)

        full_ids = torch.cat([input_ids, query_ids], dim=0)  # full_ids:[full_len]
        query_pad_mask = torch.zeros_like(query_ids, dtype=evidence_mask.dtype, device=device)
        full_evidence_mask = torch.cat([evidence_mask, query_pad_mask], dim=0)  # [full_len]

        full_ids_batch = full_ids.unsqueeze(0)  # [1, full_len]
        with torch.inference_mode():
            outputs = model(
                input_ids=full_ids_batch,
                output_hidden_states=True,
                use_cache=False,
            )
        layer_22_states = outputs.hidden_states[22].detach()  # [1, full_len, hidden_dim]

        context_len = input_ids.shape[0]
        query_len = query_ids.shape[0]
        full_len = full_ids.shape[0]

        # x_u:[1, ctx_len, hidden_dim], x_q:[1, 1, hidden_dim]
        x_u = layer_22_states[:, :context_len, :].to(dtype=torch.float32)
        x_q = layer_22_states[:, full_len - 1 : full_len, :].to(dtype=torch.float32)
        del outputs, layer_22_states

        Phi_u, R_q, C_u = heads(x_u, x_q)
        bias_scores_context = (R_q * Phi_u).sum(dim=-1)  # [1, ctx_len]

        # Pad query positions with zeros to keep alignment with full_evidence_mask:[1, full_len].
        query_bias_padding = torch.zeros((1, query_len), dtype=bias_scores_context.dtype, device=device)
        bias_scores = torch.cat([bias_scores_context, query_bias_padding], dim=-1)  # [1, full_len]

        loss = infonce_loss(bias_scores, full_evidence_mask.unsqueeze(0))
        loss = loss + 0.01 * C_u.abs().mean()  # L1 sparsity regularization on charges.

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            recall_at_32 = _recall_at_k(
                bias_scores=bias_scores,
                evidence_mask=full_evidence_mask.unsqueeze(0),
                k=32,
            )
            sparsity = float((C_u.abs() < 1e-3).float().mean().item())

        print(
            f"Step {step:03d} | "
            f"Loss: {loss.item():.6f} | "
            f"Recall@32: {recall_at_32:.3f} | "
            f"Sparsity: {sparsity:.3f}"
        )

        del x_u, x_q, Phi_u, R_q, C_u, bias_scores_context, bias_scores, loss


if __name__ == "__main__":
    train_phase3(steps=50)
