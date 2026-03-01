from __future__ import annotations

import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ in (None, ""):
    # Supports direct script execution: `python src/train_overnight.py`
    from config import MODEL_ID
    from data_gen import generate_synthetic_example
    from modules import C2FAB_Heads, infonce_loss
else:
    # Supports module execution: `python -m src.train_overnight`
    from .config import MODEL_ID
    from .data_gen import generate_synthetic_example
    from .modules import C2FAB_Heads, infonce_loss


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _llm_dtype_for_device(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "mps" else torch.bfloat16


def _recall_at_k(
    bias_scores: torch.Tensor,
    evidence_mask: torch.Tensor,
    k: int = 32,
) -> float:
    topk = min(k, bias_scores.shape[-1])
    topk_indices = torch.topk(bias_scores, k=topk, dim=-1).indices
    evidence_hits = evidence_mask.gather(dim=-1, index=topk_indices) > 0
    return float(evidence_hits.any(dim=-1).float().mean().item())


def _pad_or_trim_context(
    input_ids: torch.Tensor,
    evidence_mask: torch.Tensor,
    target_len: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Make context tensors exactly target_len while preserving evidence if trimming.
    """
    seq_len = int(input_ids.shape[0])
    if seq_len == target_len:
        return input_ids, evidence_mask

    if seq_len < target_len:
        pad = target_len - seq_len
        ids_pad = torch.full((pad,), pad_id, dtype=input_ids.dtype)
        mask_pad = torch.zeros((pad,), dtype=evidence_mask.dtype)
        return torch.cat([input_ids, ids_pad], dim=0), torch.cat([evidence_mask, mask_pad], dim=0)

    # seq_len > target_len: trim while keeping evidence span inside the window.
    evidence_positions = torch.where(evidence_mask > 0)[0]
    if evidence_positions.numel() == 0:
        start = 0
    else:
        center = int((evidence_positions[0] + evidence_positions[-1]) // 2)
        start = max(0, min(center - (target_len // 2), seq_len - target_len))
    end = start + target_len
    return input_ids[start:end], evidence_mask[start:end]


def pre_generate_examples(
    tokenizer,
    *,
    num_examples: int = 200,
    min_tokens: int = 1500,
    max_tokens: int = 3500,
    context_len: int = 3500,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Fast pre-generation only (no model forward):
    stores (input_ids_padded_3500, query_ids, evidence_mask_padded_3500) on CPU.
    """
    if tokenizer.pad_token_id is not None:
        pad_id = int(tokenizer.pad_token_id)
    elif tokenizer.eos_token_id is not None:
        pad_id = int(tokenizer.eos_token_id)
    else:
        pad_id = 0

    t0 = time.time()
    dataset: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for _ in range(num_examples):
        input_ids, query_ids, evidence_mask = generate_synthetic_example(
            tokenizer,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
        input_ids, evidence_mask = _pad_or_trim_context(
            input_ids=input_ids.cpu(),
            evidence_mask=evidence_mask.cpu(),
            target_len=context_len,
            pad_id=pad_id,
        )
        dataset.append((input_ids, query_ids.cpu(), evidence_mask))

    elapsed = time.time() - t0
    print(
        f"Pre-generated {num_examples} examples in {elapsed:.2f}s "
        f"(target: ~2s, context_len={context_len})."
    )
    return dataset


def _extract_layer22_states(
    model,
    input_ids_batch: torch.Tensor,
) -> torch.Tensor:
    """
    Run one forward pass and capture layer 22 hidden states via hook.
    Returns tensor [batch, seq_len, hidden_dim].
    """
    saved_hidden_states: dict[str, torch.Tensor] = {}

    def hook_fn(module, hook_input, hook_output):
        hidden = hook_output[0] if isinstance(hook_output, tuple) else hook_output
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        saved_hidden_states["layer_22"] = hidden.detach()

    hook = model.model.layers[22].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids_batch, use_cache=False)
    finally:
        hook.remove()

    if "layer_22" not in saved_hidden_states:
        raise RuntimeError("Layer 22 forward hook did not capture hidden states.")
    return saved_hidden_states["layer_22"]


def train_overnight(
    *,
    steps: int = 1000,
    checkpoint_every: int = 20,
    checkpoint_dir: str = "checkpoints",
) -> None:
    device = _pick_device()
    torch.manual_seed(42)
    random.seed(42)

    llm_dtype = _llm_dtype_for_device(device)
    print(f"Loading tokenizer/model: {MODEL_ID}")
    print(f"Using device={device}, llm_dtype={llm_dtype}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=llm_dtype,
        ).to(device)
    except Exception as exc:
        print(
            "Online Hugging Face load failed "
            f"({type(exc).__name__}: {exc}). Retrying with local cache only..."
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=llm_dtype,
            local_files_only=True,
        ).to(device)

    # Freeze base model completely.
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    heads = C2FAB_Heads(hidden_dim=4096, D=8, num_layers=4, num_q_heads=32).to(device)
    optimizer = AdamW(heads.parameters(), lr=1e-3)

    dataset = pre_generate_examples(
        tokenizer=tokenizer,
        num_examples=200,
        min_tokens=1500,
        max_tokens=3500,
        context_len=3500,
    )

    run = wandb.init(
        project="c2fab-hackathon",
        name="multi-scale-bidi",
        mode=os.environ.get("WANDB_MODE", "online"),
    )
    run.config.update(
        {
            "model_id": MODEL_ID,
            "steps": steps,
            "checkpoint_every": checkpoint_every,
            "context_len": 3500,
            "D": 8,
            "bidirectional": True,
            "l1_weight": 0.01,
            "ortho_weight": 0.05,
        }
    )

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting training for {steps} steps...")
    for step in range(1, steps + 1):
        step_t0 = time.time()

        input_ids_cpu, _query_ids_cpu, evidence_mask_cpu = random.choice(dataset)
        input_ids = input_ids_cpu.to(device=device, dtype=torch.long)  # [3500]
        evidence_mask = evidence_mask_cpu.to(device=device, dtype=torch.long)  # [3500]

        layer_22_states = _extract_layer22_states(model, input_ids.unsqueeze(0))  # [1, 3500, hidden_dim]
        if layer_22_states.dim() != 3:
            raise RuntimeError(
                f"Expected layer_22_states rank-3, got shape {tuple(layer_22_states.shape)}."
            )

        # Use context hidden states for both field source and single-token receptor query.
        x_u = layer_22_states.to(dtype=torch.float32)  # [1, 3500, hidden_dim]
        x_q = layer_22_states[:, -1:, :].to(dtype=torch.float32)  # [1, 1, hidden_dim]

        Phi_total, R_q, C_u = heads(x_u, x_q, use_bidirectional=True)
        bias_scores = (R_q * Phi_total).sum(dim=-1)  # [1, 3500]

        info_nce = infonce_loss(bias_scores, evidence_mask.unsqueeze(0))
        l1_sparsity = C_u.abs().mean()

        # Orthogonal penalty: Gram matrix across sequence for charge dimensions.
        C_flat = C_u.reshape(-1, C_u.shape[-1])  # [N, D]
        C_norm = F.normalize(C_flat, p=2, dim=0)
        gram = C_norm.transpose(0, 1) @ C_norm  # [D, D]
        eye = torch.eye(C_u.shape[-1], dtype=gram.dtype, device=gram.device)
        ortho_loss = torch.norm(gram - eye, p="fro")

        loss = info_nce + (0.01 * l1_sparsity) + (0.05 * ortho_loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            recall_32 = _recall_at_k(
                bias_scores=bias_scores,
                evidence_mask=evidence_mask.unsqueeze(0),
                k=32,
            )
            sparsity = float((C_u.abs() < 1e-3).float().mean().item())

        step_time = time.time() - step_t0

        wandb.log(
            {
                "loss": float(loss.item()),
                "recall_32": float(recall_32),
                "sparsity": float(sparsity),
                "step_time": float(step_time),
            },
            step=step,
        )

        print(
            f"Step {step:04d} | "
            f"Loss: {loss.item():.6f} | "
            f"Recall@32: {recall_32:.3f} | "
            f"Sparsity: {sparsity:.3f} | "
            f"StepTime: {step_time:.3f}s"
        )

        if step % checkpoint_every == 0:
            ckpt_file = ckpt_dir / f"c2fab_weights_step_{step:03d}.pt"
            torch.save(
                {
                    "step": step,
                    "heads_state_dict": heads.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": float(loss.item()),
                    "recall_32": float(recall_32),
                    "sparsity": float(sparsity),
                },
                ckpt_file,
            )
            print(f"Saved checkpoint: {ckpt_file}")

        del (
            input_ids,
            evidence_mask,
            layer_22_states,
            x_u,
            x_q,
            Phi_total,
            R_q,
            C_u,
            bias_scores,
            info_nce,
            l1_sparsity,
            C_flat,
            C_norm,
            gram,
            eye,
            ortho_loss,
            loss,
        )

    wandb.finish()


if __name__ == "__main__":
    train_overnight(steps=1000, checkpoint_every=20)
