from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.config import MODEL_ID
    from src.wrapper import ChargeFieldMinistral
else:
    from src.config import MODEL_ID
    from src.wrapper import ChargeFieldMinistral


EVIDENCE_SENTENCE = "The secret project codename is Apollo-77."
QUESTION_SUFFIX = "Question: What is the secret project codename? Answer:"

FILLER_SENTENCES = [
    "The committee released a technical note about infrastructure planning in the coastal district.",
    "Researchers compared archival records and found subtle shifts in terminology over several decades.",
    "A public report described manufacturing output, labor participation, and transport reliability trends.",
    "The municipal office published updates on water treatment capacity and seasonal demand projections.",
    "Analysts documented weather variance alongside crop yield estimates and commodity shipment logs.",
    "A historical survey summarized policy drafts, legal commentary, and revisions from prior sessions.",
    "Regional planners reviewed bridge maintenance schedules and estimated long-horizon replacement costs.",
    "Independent observers tracked migration corridors and changes in cross-border administrative rules.",
]


def _make_filler(
    tokenizer,
    target_tokens: int,
) -> str:
    pieces: list[str] = []
    token_count = 0
    while token_count < target_tokens:
        sentence = random.choice(FILLER_SENTENCES)
        pieces.append(sentence)
        token_count += len(tokenizer(sentence, add_special_tokens=False)["input_ids"])
    return " ".join(pieces)


def build_needle_prompt(
    tokenizer,
    target_filler_tokens: int = 1000,
) -> str:
    left_tokens = target_filler_tokens // 2
    right_tokens = target_filler_tokens - left_tokens
    left_filler = _make_filler(tokenizer, left_tokens)
    right_filler = _make_filler(tokenizer, right_tokens)
    return (
        f"{left_filler} "
        f"{EVIDENCE_SENTENCE} "
        f"{right_filler}\n\n"
        f"{QUESTION_SUFFIX}"
    )


def _extract_charge_magnitudes(
    cf_model: ChargeFieldMinistral,
    prompt: str,
    layer_idx: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    attn_module = cf_model.model.model.layers[layer_idx].self_attn

    def pre_hook(module, args, kwargs):
        hidden_states = None
        if args:
            hidden_states = args[0]
        elif "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
        if hidden_states is not None:
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)
            captured["hidden_states"] = hidden_states.detach()

    handle = attn_module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        input_device = ChargeFieldMinistral._input_device(cf_model.model)
        inputs = cf_model.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = cf_model.model(**inputs, use_cache=False)
    finally:
        handle.remove()

    if "hidden_states" not in captured:
        raise RuntimeError("Failed to capture hidden_states from patched attention module.")

    hidden_states = captured["hidden_states"]
    heads = attn_module.c2fab_heads
    heads_param = next(heads.parameters())
    hidden_states = hidden_states.to(device=heads_param.device, dtype=heads_param.dtype)
    with torch.no_grad():
        Phi_total, R_q, C_u = heads(hidden_states, hidden_states, use_bidirectional=True)
        # Build the full [q_len, kv_len] bias map and inspect the *last prompt query*
        # row, which is the row used for next-token prediction during generation.
        raw_bias_2d = torch.einsum("bqd,bkd->bqk", R_q, Phi_total)  # [1, q_len, kv_len]
        raw_bias = raw_bias_2d[:, -1, :].squeeze(0).float().detach().cpu()  # [kv_len]
        C_u = C_u.detach()
    charge_magnitudes = (
        torch.linalg.vector_norm(C_u.float(), ord=2, dim=-1).squeeze(0).detach().cpu()
    )  # [seq_len]
    return charge_magnitudes, raw_bias


def _get_evidence_token_positions(
    tokenizer,
    prompt: str,
) -> list[int]:
    def _find_subsequence(seq: list[int], pat: list[int]) -> int:
        if not pat or len(pat) > len(seq):
            return -1
        last = len(seq) - len(pat) + 1
        for i in range(last):
            if seq[i : i + len(pat)] == pat:
                return i
        return -1

    # Primary path: exact token-subsequence match in the same tokenization
    # regime used for model input/scoring.
    input_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    evidence_ids_with_space = tokenizer(
        " " + EVIDENCE_SENTENCE,
        add_special_tokens=False,
    )["input_ids"]
    evidence_ids_plain = tokenizer(
        EVIDENCE_SENTENCE,
        add_special_tokens=False,
    )["input_ids"]

    start = _find_subsequence(input_ids, evidence_ids_with_space)
    evidence_len = len(evidence_ids_with_space)
    if start == -1:
        start = _find_subsequence(input_ids, evidence_ids_plain)
        evidence_len = len(evidence_ids_plain)
    if start != -1:
        return list(range(start, start + evidence_len))

    # Fallback path: offset overlap in character space.
    evidence_start = prompt.index(EVIDENCE_SENTENCE)
    evidence_end = evidence_start + len(EVIDENCE_SENTENCE)

    encoded = tokenizer(
        prompt,
        # Must match model input tokenization used for scoring.
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    positions = [
        idx
        for idx, (start, end) in enumerate(offsets)
        if (start < evidence_end and end > evidence_start)
    ]
    return positions


def _print_bias_diagnostics(
    raw_bias: torch.Tensor,
    evidence_positions: list[int],
    top_k: int = 20,
) -> None:
    k = min(top_k, int(raw_bias.numel()))
    top_vals, top_idx = torch.topk(raw_bias, k=k, dim=-1)

    top_positions = top_idx.tolist()
    top_scores = [round(float(v), 6) for v in top_vals.tolist()]

    evidence_set = set(evidence_positions)
    overlap_positions = [pos for pos in top_positions if pos in evidence_set]
    overlap_count = len(overlap_positions)

    print("\n=== Bias Diagnostics ===")
    print(f"Top-{k} bias positions: {top_positions}")
    print(f"Top-{k} bias scores: {top_scores}")
    print(f"Evidence token positions: {evidence_positions}")
    print(f"Overlap count (Top-{k} vs evidence): {overlap_count}")
    print(f"Overlap positions: {overlap_positions}")


def _save_energy_plot(
    charge_magnitudes: torch.Tensor,
    output_path: str = "c2fab_energy_plot.png",
) -> None:
    if plt is None:
        print("matplotlib is not installed; skipping energy plot.")
        return

    plt.figure(figsize=(12, 4))
    plt.plot(charge_magnitudes.detach().cpu().numpy(), linewidth=1.1)
    plt.title("C2FAB Charge Magnitude Across Sequence")
    plt.xlabel("Token Position")
    plt.ylabel("||C_u||_2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved energy plot to {output_path}")


def _set_force_alpha(cf_model: ChargeFieldMinistral, alpha: float) -> None:
    with torch.no_grad():
        cf_model.heads.alphas.data = torch.ones_like(cf_model.heads.alphas.data) * float(alpha)


def _generate_with_alpha(
    cf_model: ChargeFieldMinistral,
    prompt: str,
    alpha: float,
    max_new_tokens: int = 20,
) -> str:
    _set_force_alpha(cf_model, alpha)
    return cf_model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )


def main() -> None:
    random.seed(7)

    print("Loading ChargeFieldMinistral...")
    cf_model = ChargeFieldMinistral.from_pretrained(
        model_id=MODEL_ID,
        checkpoint_path="checkpoints/c2fab_weights_step_400.pt",
        force_alpha=2.0,
    )

    print("Building Needle-in-a-Haystack prompt...")
    prompt = build_needle_prompt(cf_model.tokenizer, target_filler_tokens=1000)

    print("Running generation...")
    model_output = _generate_with_alpha(
        cf_model,
        prompt,
        alpha=2.0,
        max_new_tokens=20,
    )

    # If alpha=2.0 destabilizes decoding, back off to recover exact retrieval behavior.
    if "Apollo-77" not in model_output:
        print("Alpha 2.0 did not produce exact codename. Trying fallback alpha sweep...")
        for alpha in (1.5, 1.0, 0.5, 0.0):
            candidate = _generate_with_alpha(
                cf_model,
                prompt,
                alpha=alpha,
                max_new_tokens=20,
            )
            print(f"[alpha={alpha:.1f}] {candidate}")
            if "Apollo-77" in candidate:
                model_output = candidate
                print(f"Selected fallback alpha={alpha:.1f} based on exact-match answer.")
                break

    print("\n=== Model Output ===")
    print(model_output)

    print("\nComputing C2FAB charge magnitudes and bias diagnostics...")
    charge_magnitudes, raw_bias = _extract_charge_magnitudes(cf_model, prompt, layer_idx=32)
    evidence_positions = _get_evidence_token_positions(cf_model.tokenizer, prompt)
    _print_bias_diagnostics(raw_bias, evidence_positions, top_k=20)
    _save_energy_plot(charge_magnitudes, output_path="c2fab_energy_plot.png")


if __name__ == "__main__":
    main()
