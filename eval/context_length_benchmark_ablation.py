from __future__ import annotations

"""
Ablation benchmark for context-length retrieval.

Usage:
  PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 venv/bin/python eval/context_length_benchmark_ablation.py
  PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 venv/bin/python eval/context_length_benchmark_ablation.py --seed 2026
  PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 venv/bin/python eval/context_length_benchmark_ablation.py --force_alpha 1.5

Default variants:
  1) vanilla base model
  2) c2fab ablation (force_alpha=0.0)

Optional variant:
  3) c2fab with custom --force_alpha
"""

import argparse
import gc
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.config import MODEL_ID
    from src.wrapper import ChargeFieldMinistral
else:
    from src.config import MODEL_ID
    from src.wrapper import ChargeFieldMinistral


DEFAULT_CONTEXT_LENGTH = 32768
NEEDLE_POSITIONS = ("early", "middle", "late")
DEFAULT_RUNS_PER_POSITION = 1
CHECKPOINT_PATH = "checkpoints/c2fab_weights_step_400.pt"
QUERY = "Question: What is the planetary defense shield frequency? Answer:"
DEFAULT_MAX_NEW_TOKENS = 16

FILLER_SENTENCES = [
    "The research council released an extensive memorandum describing civil engineering projects and policy revisions.",
    "Historians reviewed archival testimony and found subtle disagreement around dates, titles, and procedural details.",
    "A technical appendix summarized weather anomalies, transport reliability, and storage capacity trends for the region.",
    "Regional observers documented legal updates, migration reports, and infrastructure modernization schedules.",
    "An annual digest compiled trade figures, public health indicators, and educational enrollment statistics.",
    "Planning offices compared draft legislation with municipal outcomes and long-term maintenance projections.",
    "Independent analysts cross-referenced economic bulletins with manufacturing updates and procurement notices.",
    "Survey teams mapped urban corridors, utility distribution, and emergency response pathways in detail.",
]


@dataclass(frozen=True)
class Trial:
    position: str
    run_idx: int
    target_mhz: int
    prompt: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vanilla vs C2FAB(alpha=0) ablation on identical context-length trials."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Deterministic seed for trial generation (default: 1337).",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help=(
            "Target filler token budget used by the prompt builder. "
            f"Use >=32000 for long-context sensitivity (default: {DEFAULT_CONTEXT_LENGTH})."
        ),
    )
    parser.add_argument(
        "--runs_per_position",
        type=int,
        default=DEFAULT_RUNS_PER_POSITION,
        help=f"Trials per insertion position (default: {DEFAULT_RUNS_PER_POSITION}).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Generation budget per trial (default: {DEFAULT_MAX_NEW_TOKENS}).",
    )
    parser.add_argument(
        "--force_alpha",
        type=float,
        default=None,
        help=(
            "Optional extra C2FAB variant alpha. "
            "If omitted, only vanilla + alpha=0 ablation are run."
        ),
    )
    return parser.parse_args()


def _input_device(model) -> torch.device:
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    return next(model.parameters()).device


def _cleanup_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _precompute_filler_token_lengths(tokenizer) -> list[int]:
    return [
        len(tokenizer(sentence, add_special_tokens=False)["input_ids"])
        for sentence in FILLER_SENTENCES
    ]


def _make_filler(target_tokens: int, filler_token_lengths: list[int]) -> str:
    pieces: list[str] = []
    token_count = 0
    while token_count < target_tokens:
        idx = random.randrange(len(FILLER_SENTENCES))
        sentence = FILLER_SENTENCES[idx]
        pieces.append(sentence)
        token_count += filler_token_lengths[idx]
    return " ".join(pieces)


def _build_prompt(
    tokenizer,
    context_length: int,
    fact_sentence: str,
    needle_position: str,
    filler_token_lengths: list[int],
) -> str:
    position_to_fraction = {
        "early": 0.2,
        "middle": 0.5,
        "late": 0.8,
    }
    if needle_position not in position_to_fraction:
        raise ValueError(
            f"needle_position must be one of {tuple(position_to_fraction.keys())}, got {needle_position!r}."
        )

    left_target = int(context_length * position_to_fraction[needle_position])
    right_target = context_length - left_target
    left = _make_filler(left_target, filler_token_lengths)
    right = _make_filler(right_target, filler_token_lengths)
    return f"{left} {fact_sentence} {right}\n\n{QUERY}"


def _safe_decode(tokenizer, token_ids) -> str:
    # Normalize decode input to plain CPU list[int] to avoid tokenizer overflow conversions.
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
    elif hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, int):
        token_ids = [token_ids]

    token_ids = [int(t) for t in token_ids]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _decode_new_tokens(tokenizer, generated_ids: torch.Tensor, input_len: int) -> str:
    new_tokens = generated_ids[0, input_len:]
    return _safe_decode(tokenizer, new_tokens).strip()


def _extract_answer_segment(text: str) -> str:
    marker = "Answer:"
    if marker in text:
        return text.rsplit(marker, maxsplit=1)[-1].strip()
    return text.strip()


def _extract_first_4_to_6_digit_candidate(text: str) -> str | None:
    match = re.search(r"\b(\d{4,6})\b", text)
    return match.group(1) if match else None


def _load_tokenizer(model_source: str):
    kwargs = {"use_fast": False}
    try:
        return AutoTokenizer.from_pretrained(
            model_source,
            local_files_only=True,
            fix_mistral_regex=True,
            **kwargs,
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(
            model_source,
            local_files_only=True,
            **kwargs,
        )
    except Exception:
        try:
            return AutoTokenizer.from_pretrained(
                model_source,
                fix_mistral_regex=True,
                **kwargs,
            )
        except TypeError:
            return AutoTokenizer.from_pretrained(model_source, **kwargs)


def _load_vanilla_model(model_source: str):
    dtype = torch.float16 if torch.backends.mps.is_available() else torch.bfloat16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=dtype,
            device_map="auto",
            local_files_only=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=dtype,
            device_map="auto",
        )
    model.eval()
    return model


def _set_force_alpha(cf_model: ChargeFieldMinistral, alpha: float) -> None:
    with torch.no_grad():
        cf_model.heads.alphas.data = torch.ones_like(cf_model.heads.alphas.data) * float(alpha)


def _run_model_generate(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    use_cache: bool,
) -> str:
    device = _input_device(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
        )

    full_text = _safe_decode(tokenizer, generated_ids[0])
    answer_segment = _extract_answer_segment(full_text)
    if answer_segment:
        return answer_segment
    return _decode_new_tokens(tokenizer, generated_ids, inputs["input_ids"].shape[-1])


def _snippet(text: str, max_len: int = 140) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1] + "..."


def _build_trials(
    tokenizer,
    *,
    context_length: int,
    runs_per_position: int,
    filler_token_lengths: list[int],
) -> list[Trial]:
    trials: list[Trial] = []
    for position in NEEDLE_POSITIONS:
        for run_idx in range(1, runs_per_position + 1):
            target_mhz = random.randint(1000, 9999)
            fact_sentence = f"The planetary defense shield frequency is {target_mhz} MHz."
            prompt = _build_prompt(
                tokenizer=tokenizer,
                context_length=context_length,
                fact_sentence=fact_sentence,
                needle_position=position,
                filler_token_lengths=filler_token_lengths,
            )
            trials.append(
                Trial(
                    position=position,
                    run_idx=run_idx,
                    target_mhz=target_mhz,
                    prompt=prompt,
                )
            )
    return trials


def _print_summary_table(
    trials: list[Trial],
    variant_passes: dict[str, list[bool]],
) -> None:
    headers = ["Position", *variant_passes.keys()]
    rows: list[list[str]] = []

    for position in NEEDLE_POSITIONS:
        idxs = [i for i, t in enumerate(trials) if t.position == position]
        row = [position]
        for variant_name in variant_passes:
            hits = sum(1 for i in idxs if variant_passes[variant_name][i])
            total = len(idxs)
            rate = (100.0 * hits / total) if total > 0 else 0.0
            row.append(f"{hits}/{total} ({rate:.1f}%)")
        rows.append(row)

    all_idxs = list(range(len(trials)))
    overall_row = ["overall"]
    for variant_name in variant_passes:
        hits = sum(1 for i in all_idxs if variant_passes[variant_name][i])
        total = len(all_idxs)
        rate = (100.0 * hits / total) if total > 0 else 0.0
        overall_row.append(f"{hits}/{total} ({rate:.1f}%)")
    rows.append(overall_row)

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line() -> str:
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def _fmt(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    print("\n=== Pass Rate by Position ===")
    print(_line())
    print(_fmt(headers))
    print(_line())
    for row in rows:
        print(_fmt(row))
    print(_line())


def main() -> None:
    args = _parse_args()
    if args.context_length < 1024:
        raise ValueError(f"--context_length must be >= 1024, got {args.context_length}.")
    if args.runs_per_position <= 0:
        raise ValueError(
            f"--runs_per_position must be positive, got {args.runs_per_position}."
        )
    if args.max_new_tokens <= 0:
        raise ValueError(f"--max_new_tokens must be positive, got {args.max_new_tokens}.")


    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    hf_logging.set_verbosity_error()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        model_source = snapshot_download(repo_id=MODEL_ID, local_files_only=True)
    except Exception:
        model_source = MODEL_ID

    print(f"Loading tokenizer from: {model_source}")
    tokenizer = _load_tokenizer(model_source)
    filler_token_lengths = _precompute_filler_token_lengths(tokenizer)
    print(
        f"Preparing trials: context={args.context_length}, positions={list(NEEDLE_POSITIONS)}, "
        f"runs_per_position={args.runs_per_position}, seed={args.seed}"
    )
    trials = _build_trials(
        tokenizer,
        context_length=args.context_length,
        runs_per_position=args.runs_per_position,
        filler_token_lengths=filler_token_lengths,
    )

    variant_passes: dict[str, list[bool]] = {}

    print("\nLoading Vanilla Ministral once...")
    vanilla_t0 = time.time()
    vanilla_model = _load_vanilla_model(model_source)
    print(f"Vanilla load time: {time.time() - vanilla_t0:.1f}s")
    vanilla_name = "vanilla"
    variant_passes[vanilla_name] = []
    for trial in trials:
        output = _run_model_generate(
            vanilla_model,
            tokenizer,
            trial.prompt,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
        pred = _extract_first_4_to_6_digit_candidate(output)
        passed = pred == str(trial.target_mhz)
        variant_passes[vanilla_name].append(passed)
        print(
            f"[{vanilla_name}] {trial.position} run {trial.run_idx}: "
            f"{'PASS' if passed else 'FAIL'} "
            f"(pred={pred!r}, target={trial.target_mhz}) -> {_snippet(output)!r}"
        )
    _cleanup_model(vanilla_model)

    print("\nLoading C2FAB Ministral once (ablation alpha=0.0)...")
    c2fab_t0 = time.time()
    c2fab_model = ChargeFieldMinistral.from_pretrained(
        model_id=model_source,
        checkpoint_path=CHECKPOINT_PATH,
        force_alpha=0.0,
    )
    print(f"C2FAB load time: {time.time() - c2fab_t0:.1f}s")

    alpha0_name = "c2fab_a0"
    variant_passes[alpha0_name] = []
    for trial in trials:
        output = _run_model_generate(
            c2fab_model.model,
            tokenizer,
            trial.prompt,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
        pred = _extract_first_4_to_6_digit_candidate(output)
        passed = pred == str(trial.target_mhz)
        variant_passes[alpha0_name].append(passed)
        print(
            f"[{alpha0_name}] {trial.position} run {trial.run_idx}: "
            f"{'PASS' if passed else 'FAIL'} "
            f"(pred={pred!r}, target={trial.target_mhz}) -> {_snippet(output)!r}"
        )

    if args.force_alpha is not None:
        if abs(args.force_alpha) < 1e-12:
            print(
                "\nSkipping optional --force_alpha run because it matches the ablation variant (0.0)."
            )
        else:
            _set_force_alpha(c2fab_model, args.force_alpha)
            custom_name = f"c2fab_a{args.force_alpha:g}"
            variant_passes[custom_name] = []
            print(f"\nRunning optional C2FAB variant with force_alpha={args.force_alpha:g}...")
            for trial in trials:
                output = _run_model_generate(
                    c2fab_model.model,
                    tokenizer,
                    trial.prompt,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
                pred = _extract_first_4_to_6_digit_candidate(output)
                passed = pred == str(trial.target_mhz)
                variant_passes[custom_name].append(passed)
                print(
                    f"[{custom_name}] {trial.position} run {trial.run_idx}: "
                    f"{'PASS' if passed else 'FAIL'} "
                    f"(pred={pred!r}, target={trial.target_mhz}) -> {_snippet(output)!r}"
                )

    _cleanup_model(c2fab_model.model)
    del c2fab_model
    _print_summary_table(trials, variant_passes)


if __name__ == "__main__":
    main()
