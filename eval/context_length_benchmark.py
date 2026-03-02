from __future__ import annotations

import gc
import random
import re
import sys
import time
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


CONTEXT_LENGTH = 2500
NEEDLE_POSITIONS = ("early", "middle", "late")
RUNS_PER_POSITION = 3
CHECKPOINT_PATH = "checkpoints/c2fab_weights_step_400.pt"
QUERY = "Question: What is the planetary defense shield frequency? Answer:"
MAX_NEW_TOKENS = 16

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


def _make_filler(tokenizer, target_tokens: int) -> str:
    pieces: list[str] = []
    token_count = 0
    while token_count < target_tokens:
        sentence = random.choice(FILLER_SENTENCES)
        pieces.append(sentence)
        token_count += len(tokenizer(sentence, add_special_tokens=False)["input_ids"])
    return " ".join(pieces)


def _build_prompt(
    tokenizer,
    context_length: int,
    fact_sentence: str,
    needle_position: str,
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
    left = _make_filler(tokenizer, left_target)
    right = _make_filler(tokenizer, right_target)
    return f"{left} {fact_sentence} {right}\n\n{QUERY}"


def _decode_new_tokens(tokenizer, generated_ids: torch.Tensor, input_len: int) -> str:
    new_tokens = generated_ids[0, input_len:]
    return _safe_decode(tokenizer, new_tokens).strip()


def _safe_decode(tokenizer, token_ids) -> str:
    # Avoid tokenizer backend overflow/conversion issues by forcing a plain
    # CPU list[int] before decode.
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
    elif hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, int):
        token_ids = [token_ids]

    token_ids = [int(t) for t in token_ids]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _extract_answer_segment(text: str) -> str:
    marker = "Answer:"
    if marker in text:
        return text.rsplit(marker, maxsplit=1)[-1].strip()
    return text.strip()


def _extract_first_mhz_candidate(text: str) -> str | None:
    # We care about recovering the planted 4-digit frequency.
    match = re.search(r"\b(\d{4})\b", text)
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
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    model.eval()
    return model


def _run_vanilla(model, tokenizer, prompt: str, max_new_tokens: int = 16) -> str:
    device = _input_device(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    # Decode from the full sequence and parse the answer segment explicitly.
    full_text = _safe_decode(tokenizer, generated_ids[0])
    answer_segment = _extract_answer_segment(full_text)
    if answer_segment:
        return answer_segment

    # Fallback if parsing fails unexpectedly.
    return _decode_new_tokens(tokenizer, generated_ids, inputs["input_ids"].shape[-1])


def _run_c2fab(model: ChargeFieldMinistral, prompt: str, max_new_tokens: int = 16) -> str:
    output = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return _extract_answer_segment(output)


def _print_results_table(results: list[dict[str, object]]) -> None:
    headers = ["Position", "Run", "Target MHz", "Vanilla EM", "C2FAB EM"]
    rows = [
        [
            str(item["position"]),
            str(item["run"]),
            str(item["target_mhz"]),
            "PASS" if item["vanilla_pass"] else "FAIL",
            "PASS" if item["c2fab_pass"] else "FAIL",
        ]
        for item in results
    ]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line() -> str:
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def _fmt(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    print("\n=== PoC Benchmark (Exact Match) ===")
    print(_line())
    print(_fmt(headers))
    print(_line())
    for row in rows:
        print(_fmt(row))
    print(_line())


def _print_summary(results: list[dict[str, object]]) -> None:
    headers = ["Position", "Vanilla", "C2FAB"]
    rows: list[list[str]] = []

    for position in NEEDLE_POSITIONS:
        subset = [r for r in results if r["position"] == position]
        vanilla_hits = sum(1 for r in subset if r["vanilla_pass"])
        c2fab_hits = sum(1 for r in subset if r["c2fab_pass"])
        rows.append(
            [
                position,
                f"{vanilla_hits}/{len(subset)}",
                f"{c2fab_hits}/{len(subset)}",
            ]
        )

    vanilla_total = sum(1 for r in results if r["vanilla_pass"])
    c2fab_total = sum(1 for r in results if r["c2fab_pass"])
    rows.append(
        [
            "overall",
            f"{vanilla_total}/{len(results)}",
            f"{c2fab_total}/{len(results)}",
        ]
    )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line() -> str:
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def _fmt(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    print("\n=== Summary (passes / runs) ===")
    print(_line())
    print(_fmt(headers))
    print(_line())
    for row in rows:
        print(_fmt(row))
    print(_line())


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        # Ensure progress lines are flushed when redirected to terminal logs.
        sys.stdout.reconfigure(line_buffering=True)

    hf_logging.set_verbosity_error()

    random.seed(17)
    torch.manual_seed(17)

    # Prefer local snapshot to avoid network/proxy interruptions.
    try:
        model_source = snapshot_download(repo_id=MODEL_ID, local_files_only=True)
    except Exception:
        model_source = MODEL_ID

    print(f"Loading tokenizer from: {model_source}")
    tokenizer = _load_tokenizer(model_source)
    results: list[dict[str, object]] = []

    print(
        f"Preparing PoC runs: context={CONTEXT_LENGTH}, positions={list(NEEDLE_POSITIONS)}, "
        f"runs_per_position={RUNS_PER_POSITION}"
    )
    for position in NEEDLE_POSITIONS:
        for run_idx in range(1, RUNS_PER_POSITION + 1):
            target_mhz = random.randint(1000, 9999)
            fact_sentence = (
                f"The planetary defense shield frequency is {target_mhz} MHz."
            )
            prompt = _build_prompt(
                tokenizer=tokenizer,
                context_length=CONTEXT_LENGTH,
                fact_sentence=fact_sentence,
                needle_position=position,
            )
            results.append(
                {
                    "position": position,
                    "run": run_idx,
                    "target_mhz": target_mhz,
                    "prompt": prompt,
                    "vanilla_output": "",
                    "c2fab_output": "",
                    "vanilla_pass": False,
                    "c2fab_pass": False,
                }
            )

    print("\nLoading Vanilla Ministral once...")
    vanilla_t0 = time.time()
    vanilla_model = _load_vanilla_model(model_source)
    print(f"Vanilla load time: {time.time() - vanilla_t0:.1f}s")
    for row in results:
        vanilla_output = _run_vanilla(
            vanilla_model,
            tokenizer,
            row["prompt"],
            max_new_tokens=MAX_NEW_TOKENS,
        )
        vanilla_pred_mhz = _extract_first_mhz_candidate(vanilla_output)
        row["vanilla_output"] = vanilla_output
        row["vanilla_pass"] = vanilla_pred_mhz == str(row["target_mhz"])
        print(
            f"[vanilla] {row['position']} run {row['run']}: "
            f"{'PASS' if row['vanilla_pass'] else 'FAIL'} "
            f"(pred={vanilla_pred_mhz!r}, target={row['target_mhz']}) -> {vanilla_output!r}"
        )
    _cleanup_model(vanilla_model)

    print("\nLoading C2FAB Ministral once...")
    c2fab_t0 = time.time()
    c2fab_model = ChargeFieldMinistral.from_pretrained(
        model_id=model_source,
        checkpoint_path=CHECKPOINT_PATH,
        force_alpha=2.0,
    )
    print(f"C2FAB load time: {time.time() - c2fab_t0:.1f}s")
    for row in results:
        c2fab_output = _run_c2fab(
            c2fab_model,
            row["prompt"],
            max_new_tokens=MAX_NEW_TOKENS,
        )
        c2fab_pred_mhz = _extract_first_mhz_candidate(c2fab_output)
        row["c2fab_output"] = c2fab_output
        row["c2fab_pass"] = c2fab_pred_mhz == str(row["target_mhz"])
        print(
            f"[c2fab ] {row['position']} run {row['run']}: "
            f"{'PASS' if row['c2fab_pass'] else 'FAIL'} "
            f"(pred={c2fab_pred_mhz!r}, target={row['target_mhz']}) -> {c2fab_output!r}"
        )
    _cleanup_model(c2fab_model.model)
    del c2fab_model

    _print_results_table(results)
    _print_summary(results)


if __name__ == "__main__":
    main()
