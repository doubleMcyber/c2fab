from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
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


DEFAULT_CONTEXT_LEN = 32768
DEFAULT_MIN_TOKENS = 24000
DEFAULT_MAX_TOKENS = 32768
DEFAULT_NUM_EXAMPLES = 12
DEFAULT_MAX_CACHE_GB = 8.0
DEFAULT_HOOK_LAYER = 22
DEFAULT_PRECOMPUTE_BATCH_SIZE = 0  # 0 => auto heuristic
DEFAULT_DATASET_SPLIT = "train"


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _llm_dtype_for_device(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "mps" else torch.bfloat16


def _cache_storage_dtype() -> torch.dtype:
    # bfloat16 keeps float32-like dynamic range for long contexts while
    # retaining 2-byte cache footprint.
    return torch.bfloat16


def _load_tokenizer_with_fix(model_id: str, *, local_files_only: bool = False):
    kwargs = {"local_files_only": local_files_only}
    try:
        return AutoTokenizer.from_pretrained(
            model_id,
            fix_mistral_regex=True,
            **kwargs,
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id, **kwargs)


def _recall_at_k(
    bias_scores: torch.Tensor,
    evidence_mask: torch.Tensor,
    k: int = 32,
) -> float:
    topk = min(k, bias_scores.shape[-1])
    topk_indices = torch.topk(bias_scores, k=topk, dim=-1).indices
    evidence_hits = evidence_mask.gather(dim=-1, index=topk_indices) > 0
    return float(evidence_hits.any(dim=-1).float().mean().item())


def _resolve_pad_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return 0


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


def _pack_example_with_query(
    *,
    context_ids: torch.Tensor,
    evidence_mask: torch.Tensor,
    query_ids: torch.Tensor | None,
    target_len: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the training sequence as [context, query], preserving evidence alignment.
    Query positions are always masked out (0) in evidence_mask.
    """
    context_ids = context_ids.to(dtype=torch.long, device="cpu").reshape(-1)
    evidence_mask = evidence_mask.to(dtype=torch.long, device="cpu").reshape(-1)
    if context_ids.shape[0] != evidence_mask.shape[0]:
        raise ValueError(
            "context_ids and evidence_mask must have equal length, "
            f"got {context_ids.shape[0]} vs {evidence_mask.shape[0]}."
        )

    if query_ids is None:
        query_ids = torch.empty((0,), dtype=torch.long, device="cpu")
    else:
        query_ids = query_ids.to(dtype=torch.long, device="cpu").reshape(-1)

    query_len = int(query_ids.shape[0])
    if query_len == 0:
        return _pad_or_trim_context(
            input_ids=context_ids,
            evidence_mask=evidence_mask,
            target_len=target_len,
            pad_id=pad_id,
        )
    if query_len >= target_len:
        raise ValueError(
            f"query_len ({query_len}) must be smaller than target_len ({target_len})."
        )

    context_target_len = target_len - query_len
    context_ids_fixed, evidence_fixed = _pad_or_trim_context(
        input_ids=context_ids,
        evidence_mask=evidence_mask,
        target_len=context_target_len,
        pad_id=pad_id,
    )
    query_evidence = torch.zeros((query_len,), dtype=evidence_fixed.dtype)
    full_ids = torch.cat([context_ids_fixed, query_ids], dim=0)
    full_evidence_mask = torch.cat([evidence_fixed, query_evidence], dim=0)
    return full_ids, full_evidence_mask


def pre_generate_examples(
    tokenizer,
    *,
    num_examples: int = DEFAULT_NUM_EXAMPLES,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    context_len: int = DEFAULT_CONTEXT_LEN,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Fast pre-generation only (no model forward):
    stores ([context,query]_fixed, evidence_mask_fixed) on CPU.
    """
    pad_id = _resolve_pad_id(tokenizer)

    t0 = time.time()
    dataset: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(num_examples):
        input_ids, query_ids, evidence_mask = generate_synthetic_example(
            tokenizer,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
        full_ids, full_evidence_mask = _pack_example_with_query(
            context_ids=input_ids.cpu(),
            evidence_mask=evidence_mask.cpu(),
            query_ids=query_ids.cpu(),
            target_len=context_len,
            pad_id=pad_id,
        )
        dataset.append((full_ids, full_evidence_mask))

    elapsed = time.time() - t0
    print(
        f"Pre-generated {num_examples} examples in {elapsed:.2f}s "
        f"(target: ~2s, context_len={context_len})."
    )
    return dataset


def _load_prebuilt_dataset_examples(
    *,
    dataset_path: str,
    dataset_split: str,
    context_len: int,
    pad_id: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Load pre-generated examples from disk.

    Supported formats:
      - {"splits": {"train": [example, ...], "eval": [...]}, ...}
      - {"examples": [example, ...], ...}
      - [example, ...]

    An example can be:
      - {"input_ids": ..., "evidence_mask": ..., "query_ids": optional}
      - {"context_input_ids": ..., "evidence_mask": ..., "query_ids": optional}
      - tuple/list of (input_ids, evidence_mask) or (input_ids, query_ids, evidence_mask)
    """
    payload: Any = torch.load(dataset_path, map_location="cpu")

    if isinstance(payload, dict):
        if "splits" in payload:
            splits = payload["splits"]
            if dataset_split not in splits:
                available = sorted(splits.keys())
                raise KeyError(
                    f"Split {dataset_split!r} not found in dataset. Available splits: {available}."
                )
            raw_examples = splits[dataset_split]
        elif "examples" in payload:
            raw_examples = payload["examples"]
        else:
            raise ValueError(
                "Unsupported dataset dict format. Expected keys 'splits' or 'examples'."
            )
    elif isinstance(payload, list):
        raw_examples = payload
    else:
        raise ValueError(
            "Unsupported dataset payload type. Expected dict or list, "
            f"got {type(payload).__name__}."
        )

    if not raw_examples:
        raise ValueError(f"No examples found in dataset {dataset_path!r}.")

    dataset: list[tuple[torch.Tensor, torch.Tensor]] = []
    for idx, example in enumerate(raw_examples):
        context_ids: torch.Tensor | None = None
        query_ids: torch.Tensor | None = None
        evidence_mask: torch.Tensor | None = None

        if isinstance(example, dict):
            context_ids = example.get("context_input_ids")
            if context_ids is None:
                context_ids = example.get("input_ids")
            query_ids = example.get("query_ids")
            evidence_mask = example.get("evidence_mask")
        elif isinstance(example, (tuple, list)):
            if len(example) == 2:
                context_ids, evidence_mask = example
                query_ids = None
            elif len(example) >= 3:
                context_ids, query_ids, evidence_mask = example[0], example[1], example[2]
            else:
                raise ValueError(f"Example #{idx} has invalid tuple length: {len(example)}.")
        else:
            raise ValueError(
                f"Example #{idx} has unsupported type {type(example).__name__}; "
                "expected dict, tuple, or list."
            )

        if context_ids is None or evidence_mask is None:
            raise ValueError(
                f"Example #{idx} missing required fields input_ids/context_input_ids or evidence_mask."
            )

        full_ids, full_evidence_mask = _pack_example_with_query(
            context_ids=torch.as_tensor(context_ids),
            evidence_mask=torch.as_tensor(evidence_mask),
            query_ids=torch.as_tensor(query_ids) if query_ids is not None else None,
            target_len=context_len,
            pad_id=pad_id,
        )
        dataset.append((full_ids, full_evidence_mask))

    return dataset


def _resolve_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Expected a model with model.layers (Mistral-style structure).")


def _resolve_hook_layer_index(model, requested_layer: int) -> tuple[int, int]:
    layers = _resolve_transformer_layers(model)
    num_layers = len(layers)
    if num_layers <= 0:
        raise ValueError("Model has no transformer layers.")
    layer_index = int(max(0, min(requested_layer, num_layers - 1)))
    return layer_index, num_layers


def _extract_layer_states(
    model,
    input_ids_batch: torch.Tensor,
    *,
    layer_index: int,
) -> torch.Tensor:
    """
    Run one forward pass and capture one transformer layer's hidden states via hook.
    Returns tensor [batch, seq_len, hidden_dim].
    """
    saved_hidden_states: dict[str, torch.Tensor] = {}
    key = f"layer_{layer_index}"
    layers = _resolve_transformer_layers(model)

    def hook_fn(module, hook_input, hook_output):
        hidden = hook_output[0] if isinstance(hook_output, tuple) else hook_output
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        saved_hidden_states[key] = hidden.detach()

    hook = layers[layer_index].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids_batch, use_cache=False)
    finally:
        hook.remove()

    if key not in saved_hidden_states:
        raise RuntimeError(f"Forward hook did not capture hidden states for layer {layer_index}.")
    return saved_hidden_states[key]


def _max_examples_for_cache_budget(
    *,
    context_len: int,
    hidden_dim: int,
    cache_dtype: torch.dtype,
    max_cache_gb: float,
    requested_examples: int,
) -> int:
    elem_size = torch.tensor([], dtype=cache_dtype).element_size()
    bytes_per_example = context_len * hidden_dim * elem_size
    budget_bytes = int(max_cache_gb * (1024**3))
    max_examples = max(1, budget_bytes // max(1, bytes_per_example))
    return max(1, min(requested_examples, max_examples))


def _auto_precompute_batch_size(context_len: int) -> int:
    if context_len >= 32768:
        return 1
    if context_len >= 16384:
        return 2
    if context_len >= 8192:
        return 4
    if context_len >= 4096:
        return 8
    return 16


def _precompute_layer22_cache(
    *,
    model,
    dataset: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    layer_index: int,
    cache_dtype: torch.dtype,
    batch_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    One-time base-model pass to cache selected-layer hidden states on CPU.

    Returns:
        list[(layer_states_cpu[seq_len, hidden_dim], evidence_mask_cpu[seq_len])]
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    t0 = time.time()
    total = len(dataset)
    start = 0
    while start < total:
        end = min(start + batch_size, total)
        batch_slice = dataset[start:end]
        input_ids_batch = torch.stack([item[0] for item in batch_slice], dim=0).to(
            device=device,
            dtype=torch.long,
        )
        try:
            layer_states = _extract_layer_states(
                model,
                input_ids_batch,
                layer_index=layer_index,
            )  # [batch, seq_len, hidden_dim]
        except RuntimeError as exc:
            err = str(exc).lower()
            if "out of memory" in err and batch_size > 1:
                new_batch_size = max(1, batch_size // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                print(
                    f"OOM during cache precompute at batch_size={batch_size}; "
                    f"retrying with batch_size={new_batch_size}."
                )
                batch_size = new_batch_size
                continue
            raise

        layer_states_cpu = layer_states.to(dtype=cache_dtype, device="cpu").contiguous()
        for local_idx, (_ids_cpu, evidence_mask_cpu) in enumerate(batch_slice):
            cached.append((layer_states_cpu[local_idx], evidence_mask_cpu.contiguous()))
        start = end

        if start == len(batch_slice) or start % 5 == 0 or start == total:
            print(f"Cached layer states: {start}/{total} (batch_size={batch_size})")
        del input_ids_batch, layer_states, layer_states_cpu

    elapsed = time.time() - t0
    print(
        f"Precomputed layer-{layer_index} cache for {total} examples "
        f"in {elapsed:.2f}s."
    )
    return cached


def _cleanup_base_model(model) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def train_overnight(
    *,
    steps: int = 1000,
    checkpoint_every: int = 20,
    checkpoint_dir: str = "checkpoints",
    context_len: int = DEFAULT_CONTEXT_LEN,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    num_examples: int = DEFAULT_NUM_EXAMPLES,
    max_cache_gb: float = DEFAULT_MAX_CACHE_GB,
    hook_layer: int = DEFAULT_HOOK_LAYER,
    precompute_batch_size: int = DEFAULT_PRECOMPUTE_BATCH_SIZE,
    dataset_path: str | None = None,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
) -> None:
    if min_tokens > max_tokens:
        raise ValueError(
            f"min_tokens must be <= max_tokens, got min_tokens={min_tokens}, max_tokens={max_tokens}."
        )
    if context_len <= 0:
        raise ValueError(f"context_len must be positive, got {context_len}.")
    if num_examples <= 0:
        raise ValueError(f"num_examples must be positive, got {num_examples}.")
    if precompute_batch_size < 0:
        raise ValueError(
            "precompute_batch_size must be >= 0 "
            f"(0 means auto), got {precompute_batch_size}."
        )
    if dataset_split.strip() == "":
        raise ValueError("dataset_split must be non-empty.")

    device = _pick_device()
    torch.manual_seed(42)
    random.seed(42)

    llm_dtype = _llm_dtype_for_device(device)
    print(f"Loading tokenizer/model: {MODEL_ID}")
    print(f"Using device={device}, llm_dtype={llm_dtype}")

    try:
        tokenizer = _load_tokenizer_with_fix(MODEL_ID, local_files_only=False)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=llm_dtype,
        ).to(device)
    except Exception as exc:
        print(
            "Online Hugging Face load failed "
            f"({type(exc).__name__}: {exc}). Retrying with local cache only..."
        )
        tokenizer = _load_tokenizer_with_fix(MODEL_ID, local_files_only=True)
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
    # Alphas are applied in patched attention at inference time, not in this
    # standalone heads-training objective.
    heads.alphas.requires_grad_(False)
    trainable_params = [p for p in heads.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=1e-3)

    cache_dtype = _cache_storage_dtype()
    hidden_dim = int(getattr(model.config, "hidden_size", 4096))
    max_examples_by_budget = _max_examples_for_cache_budget(
        context_len=context_len,
        hidden_dim=hidden_dim,
        cache_dtype=cache_dtype,
        max_cache_gb=max_cache_gb,
        requested_examples=num_examples,
    )

    hook_layer_index, num_layers = _resolve_hook_layer_index(model, hook_layer)
    if hook_layer_index != hook_layer:
        print(
            f"Requested hook_layer={hook_layer} is out of range for "
            f"{num_layers} layers; using hook_layer={hook_layer_index}."
        )

    if precompute_batch_size == 0:
        precompute_batch_size = _auto_precompute_batch_size(context_len)
    print(
        f"Caching layer={hook_layer_index} with cache_dtype={cache_dtype} "
        f"and precompute_batch_size={precompute_batch_size}."
    )

    pad_id = _resolve_pad_id(tokenizer)
    if dataset_path is not None:
        print(f"Loading pre-generated dataset from: {dataset_path} (split={dataset_split})")
        loaded_dataset = _load_prebuilt_dataset_examples(
            dataset_path=dataset_path,
            dataset_split=dataset_split,
            context_len=context_len,
            pad_id=pad_id,
        )
        available_examples = len(loaded_dataset)
        requested_examples = min(num_examples, available_examples)
        effective_examples = min(requested_examples, max_examples_by_budget)
        if requested_examples < num_examples:
            print(
                f"Dataset only has {available_examples} examples in split={dataset_split}; "
                f"using {requested_examples}."
            )
        if effective_examples < requested_examples:
            print(
                f"Reducing examples from {requested_examples} to {effective_examples} "
                f"to respect hidden-state cache budget ({max_cache_gb:.1f} GB)."
            )
        random.shuffle(loaded_dataset)
        dataset = loaded_dataset[:effective_examples]
        del loaded_dataset
    else:
        effective_examples = min(num_examples, max_examples_by_budget)
        if effective_examples < num_examples:
            print(
                f"Reducing num_examples from {num_examples} to {effective_examples} "
                f"to respect hidden-state cache budget ({max_cache_gb:.1f} GB)."
            )
        dataset = pre_generate_examples(
            tokenizer=tokenizer,
            num_examples=effective_examples,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            context_len=context_len,
        )

    if len(dataset) == 0:
        raise RuntimeError("No training examples available after dataset preparation.")

    feature_cache = _precompute_layer22_cache(
        model=model,
        dataset=dataset,
        device=device,
        layer_index=hook_layer_index,
        cache_dtype=cache_dtype,
        batch_size=precompute_batch_size,
    )
    _cleanup_base_model(model)
    del dataset

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
            "context_len": context_len,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "num_examples": effective_examples,
            "dataset_path": dataset_path,
            "dataset_split": dataset_split,
            "max_cache_gb": max_cache_gb,
            "hook_layer": hook_layer_index,
            "precompute_batch_size": precompute_batch_size,
            "cache_dtype": str(cache_dtype),
            "D": 8,
            "bidirectional": True,
            "l1_weight": 0.01,
            "ortho_weight": 0.05,
        }
    )

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting training for {steps} steps...")
    sample_order = list(range(len(feature_cache)))
    random.shuffle(sample_order)
    for step in range(1, steps + 1):
        step_t0 = time.time()

        if not sample_order:
            sample_order = list(range(len(feature_cache)))
            random.shuffle(sample_order)
        sample_idx = sample_order.pop()
        layer_22_states_cpu, evidence_mask_cpu = feature_cache[sample_idx]
        evidence_mask = evidence_mask_cpu.to(device=device, dtype=torch.long)
        layer_22_states = layer_22_states_cpu.unsqueeze(0).to(device=device, dtype=torch.float32)
        if layer_22_states.dim() != 3:  # [1, seq_len, hidden_dim]
            raise RuntimeError(
                f"Expected layer_22_states rank-3, got shape {tuple(layer_22_states.shape)}."
            )

        # Use context hidden states for both field source and single-token receptor query.
        x_u = layer_22_states
        x_q = layer_22_states[:, -1:, :]

        Phi_total, R_q, C_u = heads(x_u, x_q, use_bidirectional=True)
        bias_scores = (R_q * Phi_total).sum(dim=-1)  # [1, 3500]

        info_nce = infonce_loss(bias_scores, evidence_mask.unsqueeze(0))
        l1_sparsity = C_u.abs().mean()

        # Orthogonal penalty: Gram matrix across sequence for charge dimensions.
        C_flat = C_u.reshape(-1, C_u.shape[-1])  # [N, D]
        col_norms = torch.linalg.vector_norm(C_flat, ord=2, dim=0, keepdim=True)
        C_norm = C_flat / torch.clamp(col_norms, min=1e-8)
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train C2FAB heads with one-time layer22 cache precompute."
    )
    parser.add_argument("--steps", type=int, default=1000, help="Training steps.")
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=20,
        help="Checkpoint save interval in steps.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory for training checkpoints.",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=DEFAULT_CONTEXT_LEN,
        help=f"Fixed context length after pad/trim (default: {DEFAULT_CONTEXT_LEN}).",
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=DEFAULT_MIN_TOKENS,
        help=f"Minimum synthetic context tokens (default: {DEFAULT_MIN_TOKENS}).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum synthetic context tokens (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=DEFAULT_NUM_EXAMPLES,
        help=f"Requested number of pre-generated examples (default: {DEFAULT_NUM_EXAMPLES}).",
    )
    parser.add_argument(
        "--max_cache_gb",
        type=float,
        default=DEFAULT_MAX_CACHE_GB,
        help=f"CPU cache budget for layer22 states in GB (default: {DEFAULT_MAX_CACHE_GB}).",
    )
    parser.add_argument(
        "--hook_layer",
        type=int,
        default=DEFAULT_HOOK_LAYER,
        help=(
            f"Transformer layer index to cache hidden states from "
            f"(default: {DEFAULT_HOOK_LAYER}, auto-clamped to valid range)."
        ),
    )
    parser.add_argument(
        "--precompute_batch_size",
        type=int,
        default=DEFAULT_PRECOMPUTE_BATCH_SIZE,
        help=(
            "Batch size for cache precompute forward passes. "
            "Use 0 for automatic context-length-based heuristic."
        ),
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help=(
            "Optional path to a pre-generated dataset file. "
            "When provided, synthetic generation is skipped."
        ),
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default=DEFAULT_DATASET_SPLIT,
        help=f"Dataset split to load when --dataset_path is set (default: {DEFAULT_DATASET_SPLIT}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_overnight(
        steps=args.steps,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        context_len=args.context_len,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        num_examples=args.num_examples,
        max_cache_gb=args.max_cache_gb,
        hook_layer=args.hook_layer,
        precompute_batch_size=args.precompute_batch_size,
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
    )
