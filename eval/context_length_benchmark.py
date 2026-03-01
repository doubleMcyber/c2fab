from __future__ import annotations

import gc
import random
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.config import MODEL_ID
    from src.wrapper import ChargeFieldMinistral
else:
    from src.config import MODEL_ID
    from src.wrapper import ChargeFieldMinistral


CONTEXT_LENGTHS = [1000, 2000, 4000, 6000, 8000]
CHECKPOINT_PATH = "checkpoints/c2fab_weights_step_400.pt"
QUERY = "Question: What is the planetary defense shield frequency? Answer:"

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
) -> str:
    left_target = context_length // 2
    right_target = context_length - left_target
    left = _make_filler(tokenizer, left_target)
    right = _make_filler(tokenizer, right_target)
    return f"{left} {fact_sentence} {right}\n\n{QUERY}"


def _decode_new_tokens(tokenizer, generated_ids: torch.Tensor, input_len: int) -> str:
    new_tokens = generated_ids[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _load_tokenizer(model_source: str):
    try:
        return AutoTokenizer.from_pretrained(model_source, local_files_only=True, use_fast=False)
    except Exception:
        return AutoTokenizer.from_pretrained(model_source, use_fast=False)


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


def _run_vanilla(model, tokenizer, prompt: str, max_new_tokens: int = 10) -> str:
    device = _input_device(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
        )
    return _decode_new_tokens(tokenizer, generated_ids, inputs["input_ids"].shape[-1])


def _run_c2fab(model: ChargeFieldMinistral, prompt: str, max_new_tokens: int = 10) -> str:
    return model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )


def _print_results_table(results: list[dict[str, str]]) -> None:
    headers = ["Context Len", "Target MHz", "Vanilla EM", "C2FAB EM"]
    rows = [
        [
            str(item["context_len"]),
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

    print("\n=== Vanilla vs C2FAB (Exact Match) ===")
    print(_line())
    print(_fmt(headers))
    print(_line())
    for row in rows:
        print(_fmt(row))
    print(_line())


def main() -> None:
    random.seed(17)
    torch.manual_seed(17)

    # Prefer local snapshot to avoid network/proxy interruptions.
    try:
        model_source = snapshot_download(repo_id=MODEL_ID, local_files_only=True)
    except Exception:
        model_source = MODEL_ID

    print(f"Loading tokenizer from: {model_source}")
    tokenizer = _load_tokenizer(model_source)
    results: list[dict[str, str]] = []

    for context_len in CONTEXT_LENGTHS:
        target_mhz = random.randint(1000, 9999)
        fact_sentence = (
            f"The planetary defense shield frequency is {target_mhz} MHz."
        )
        prompt = _build_prompt(tokenizer, context_len, fact_sentence)

        print(f"\n--- Context length: {context_len} ---")

        print("Running Vanilla Ministral...")
        vanilla_model = _load_vanilla_model(model_source)
        vanilla_output = _run_vanilla(vanilla_model, tokenizer, prompt, max_new_tokens=10)
        vanilla_pass = str(target_mhz) in vanilla_output
        print(f"Vanilla output: {vanilla_output!r}")
        _cleanup_model(vanilla_model)

        print("Running C2FAB Ministral...")
        c2fab_model = ChargeFieldMinistral.from_pretrained(
            model_id=model_source,
            checkpoint_path=CHECKPOINT_PATH,
            force_alpha=2.0,
        )
        c2fab_output = _run_c2fab(c2fab_model, prompt, max_new_tokens=10)
        c2fab_pass = str(target_mhz) in c2fab_output
        print(f"C2FAB output: {c2fab_output!r}")
        _cleanup_model(c2fab_model.model)
        del c2fab_model

        results.append(
            {
                "context_len": context_len,
                "target_mhz": target_mhz,
                "vanilla_pass": vanilla_pass,
                "c2fab_pass": c2fab_pass,
            }
        )

    _print_results_table(results)


if __name__ == "__main__":
    main()
