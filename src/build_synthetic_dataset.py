from __future__ import annotations

"""
Build a richer synthetic long-context dataset for C2FAB training/eval.

This script creates three splits:
  - train: seen template families
  - eval_seen: held-out examples from seen families
  - eval_holdout: examples from holdout families only

Each example stores:
  - context_input_ids
  - query_ids
  - evidence_mask
  - metadata (family/style/target etc.)

Usage example:
  PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 venv/bin/python src/build_synthetic_dataset.py \
    --output_path data/synthetic/c2fab_synth_v2.pt \
    --seed 2026 \
    --train_examples 256 \
    --eval_seen_examples 64 \
    --eval_holdout_examples 64 \
    --min_tokens 24000 \
    --max_tokens 32768 \
    --holdout_families date_iso,alnum_code
"""

import argparse
import random
import string
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import torch
from transformers import AutoTokenizer

if __package__ in (None, ""):
    from config import MODEL_ID
    from data_gen import generate_synthetic_example
else:
    from .config import MODEL_ID
    from .data_gen import generate_synthetic_example


DEFAULT_OUTPUT_PATH = "data/synthetic/c2fab_synth_v2.pt"
DEFAULT_SEED = 1337
DEFAULT_TRAIN_EXAMPLES = 192
DEFAULT_EVAL_SEEN_EXAMPLES = 48
DEFAULT_EVAL_HOLDOUT_EXAMPLES = 48
DEFAULT_MIN_TOKENS = 24000
DEFAULT_MAX_TOKENS = 32768
DEFAULT_BASELINE_RATIO = 0.10
DEFAULT_HOLDOUT_FAMILIES = "date_iso,alnum_code"

FAMILY_NAMES = (
    "numeric_mhz",
    "date_iso",
    "alnum_code",
    "entity_name",
    "quantity_unit",
    "baseline_apollo",
)

STYLE_BANKS: dict[str, list[str]] = {
    "prose": [
        "The operations office circulated a policy memo covering procurement and maintenance constraints.",
        "Analysts compared district-level outcomes and identified recurring bottlenecks in infrastructure projects.",
        "A technical appendix summarized logistics timelines, weather disruptions, and staffing adjustments.",
        "Regional observers documented legal changes, permitting delays, and audit recommendations.",
        "An annual digest consolidated safety statistics, budget revisions, and contractor performance notes.",
        "Program managers reviewed transport data and proposed staggered rollout schedules.",
    ],
    "bullet": [
        "- Field unit alpha completed diagnostics for corridor segment seven.",
        "- Procurement report flagged a delay in ceramic shield fabrication.",
        "- Infrastructure board approved emergency maintenance for sector relay towers.",
        "- Documentation team revised archival labels and compliance metadata.",
        "- Capacity planning memo updated storage thresholds for peak demand windows.",
        "- Civil engineering group issued an addendum on drainage and containment.",
    ],
    "log": [
        "2026-02-11 09:14:12 INFO relay-status: checksum verified for archive stream.",
        "2026-02-11 09:22:41 INFO dispatch: maintenance crew assigned to platform C.",
        "2026-02-11 10:03:17 WARN telemetry: packet jitter exceeded baseline for 3m.",
        "2026-02-11 10:18:59 INFO compliance: records synced to immutable ledger.",
        "2026-02-11 10:37:04 INFO routing: detour approved for uplink pipeline.",
        "2026-02-11 10:52:30 WARN power: reserve margin dipped below advisory level.",
    ],
    "technical": [
        "Section 4.2 details transfer coefficients, damping constraints, and thermal drift assumptions.",
        "Validation notes compare baseline calibration vectors against updated boundary conditions.",
        "Revision records enumerate test harness changes and acceptance threshold deltas.",
        "The annex lists subsystem dependencies, integration gates, and rollback procedures.",
        "Control-plane snapshots capture queue depth, retry intervals, and failover timing.",
        "Bench reports summarize nominal throughput, variance bands, and outlier remediation steps.",
    ],
}

ENTITY_NAMES = [
    "Avery Solano",
    "Mina Park",
    "Rohan Ilyas",
    "Lea Navarro",
    "Iris Mendez",
    "Noah Sterling",
    "Talia Brooks",
    "Jonah Mercer",
    "Rina Okafor",
    "Kian Vega",
]

UNITS = ["kPa", "ms", "kW", "ppm", "mT"]


@dataclass(frozen=True)
class FamilyPayload:
    family: str
    evidence_sentence: str
    query_text: str
    answer_text: str
    distractors: tuple[str, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build synthetic long-context dataset with family holdouts and hard negatives."
    )
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train_examples", type=int, default=DEFAULT_TRAIN_EXAMPLES)
    parser.add_argument("--eval_seen_examples", type=int, default=DEFAULT_EVAL_SEEN_EXAMPLES)
    parser.add_argument("--eval_holdout_examples", type=int, default=DEFAULT_EVAL_HOLDOUT_EXAMPLES)
    parser.add_argument("--min_tokens", type=int, default=DEFAULT_MIN_TOKENS)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--holdout_families",
        type=str,
        default=DEFAULT_HOLDOUT_FAMILIES,
        help=(
            "Comma-separated family names reserved for eval_holdout "
            f"(available: {', '.join(FAMILY_NAMES)})."
        ),
    )
    parser.add_argument(
        "--baseline_ratio",
        type=float,
        default=DEFAULT_BASELINE_RATIO,
        help="Probability of sampling baseline_apollo when that family is available (0..1).",
    )
    return parser.parse_args()


def _load_tokenizer_with_fix(model_source: str):
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


def _precompute_style_lengths(tokenizer) -> dict[str, list[int]]:
    return {
        style: [len(tokenizer(s, add_special_tokens=False)["input_ids"]) for s in sents]
        for style, sents in STYLE_BANKS.items()
    }


def _sample_filler_text(
    *,
    rng: random.Random,
    style: str,
    target_tokens: int,
    style_lengths: dict[str, list[int]],
) -> str:
    bank = STYLE_BANKS[style]
    lengths = style_lengths[style]
    pieces: list[str] = []
    token_count = 0
    while token_count < target_tokens:
        idx = rng.randrange(len(bank))
        pieces.append(bank[idx])
        token_count += lengths[idx]
    sep = "\n" if style in ("bullet", "log") else " "
    return sep.join(pieces)


def _sample_needle_fraction(rng: random.Random) -> float:
    bucket = rng.choice(("very_early", "early", "middle", "late", "very_late"))
    if bucket == "very_early":
        return rng.uniform(0.05, 0.15)
    if bucket == "early":
        return rng.uniform(0.18, 0.32)
    if bucket == "middle":
        return rng.uniform(0.42, 0.58)
    if bucket == "late":
        return rng.uniform(0.68, 0.82)
    return rng.uniform(0.85, 0.95)


def _date_to_str(d: date) -> str:
    return d.isoformat()


def _mk_numeric_payload(rng: random.Random) -> FamilyPayload:
    target = rng.randint(1000, 999999)
    d1 = max(1000, min(999999, target + rng.choice((-73, -29, 41, 87))))
    d2 = max(1000, min(999999, target + rng.choice((-151, -97, 113, 179))))
    evidence = f"The planetary defense shield frequency is {target} MHz."
    query = rng.choice(
        (
            "What is the planetary defense shield frequency in MHz?",
            "Report the shield frequency value for planetary defense.",
            "Give the exact defense shield frequency (MHz).",
        )
    )
    distractors = (
        f"A retired prototype once used a shield frequency of {d1} MHz.",
        f"An outdated maintenance sheet listed {d2} MHz for a legacy calibration.",
    )
    return FamilyPayload("numeric_mhz", evidence, query, str(target), distractors)


def _mk_date_payload(rng: random.Random) -> FamilyPayload:
    start = date(2012, 1, 1)
    target = start + timedelta(days=rng.randint(0, 8000))
    d1 = target + timedelta(days=rng.choice((-11, -5, 7, 13)))
    d2 = target + timedelta(days=rng.choice((-19, -8, 9, 17)))
    target_s = _date_to_str(target)
    evidence = f"The incident response rehearsal date is {target_s}."
    query = rng.choice(
        (
            "What is the incident response rehearsal date (YYYY-MM-DD)?",
            "Provide the exact rehearsal date for incident response.",
            "Return only the rehearsal date.",
        )
    )
    distractors = (
        f"A draft agenda incorrectly referenced {_date_to_str(d1)} as the rehearsal date.",
        f"A preliminary checklist used {_date_to_str(d2)} before final approval.",
    )
    return FamilyPayload("date_iso", evidence, query, target_s, distractors)


def _mk_alnum_payload(rng: random.Random) -> FamilyPayload:
    prefix = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
    digits = rng.randint(1000, 9999)
    target = f"{prefix}-{digits}"
    d1 = f"{prefix}-{max(1000, min(9999, digits + rng.choice((-9, -4, 6, 11))))}"
    mutated_prefix = prefix[:2] + rng.choice(string.ascii_uppercase)
    d2 = f"{mutated_prefix}-{digits}"
    evidence = f"The secure vault access code is {target}."
    query = rng.choice(
        (
            "What is the secure vault access code?",
            "Provide the exact vault access code.",
            "Return only the secure code token.",
        )
    )
    distractors = (
        f"An expired training key used the code {d1}.",
        f"A neighboring subsystem accepted {d2} during a prior release.",
    )
    return FamilyPayload("alnum_code", evidence, query, target, distractors)


def _mk_entity_payload(rng: random.Random) -> FamilyPayload:
    target = rng.choice(ENTITY_NAMES)
    decoys = [name for name in ENTITY_NAMES if name != target]
    rng.shuffle(decoys)
    evidence = f"The project lead for uplink validation is {target}."
    query = rng.choice(
        (
            "Who is the project lead for uplink validation?",
            "Name the lead responsible for uplink validation.",
            "Provide the exact lead name for uplink validation.",
        )
    )
    distractors = (
        f"A separate review committee was chaired by {decoys[0]}.",
        f"The adjacent modernization task force listed {decoys[1]} as coordinator.",
    )
    return FamilyPayload("entity_name", evidence, query, target, distractors)


def _mk_quantity_payload(rng: random.Random) -> FamilyPayload:
    unit = rng.choice(UNITS)
    base = rng.uniform(12.0, 980.0)
    target_value = f"{base:.1f}"
    d1_value = f"{max(0.1, base + rng.choice((-8.2, -3.7, 4.9, 9.1))):.1f}"
    d2_value = f"{max(0.1, base + rng.choice((-14.4, -6.8, 7.6, 15.3))):.1f}"
    target = f"{target_value} {unit}"
    evidence = f"The reactor containment threshold is {target}."
    query = rng.choice(
        (
            "What is the reactor containment threshold?",
            "Provide the exact containment threshold value and unit.",
            "Return the containment threshold measurement.",
        )
    )
    distractors = (
        f"Legacy documentation recorded {d1_value} {unit} for a decommissioned unit.",
        f"A temporary safety override referenced {d2_value} {unit} in testing notes.",
    )
    return FamilyPayload("quantity_unit", evidence, query, target, distractors)


def _make_family_payload(rng: random.Random, family: str) -> FamilyPayload:
    if family == "numeric_mhz":
        return _mk_numeric_payload(rng)
    if family == "date_iso":
        return _mk_date_payload(rng)
    if family == "alnum_code":
        return _mk_alnum_payload(rng)
    if family == "entity_name":
        return _mk_entity_payload(rng)
    if family == "quantity_unit":
        return _mk_quantity_payload(rng)
    raise ValueError(f"Unsupported family: {family}")


def _encode_with_mask(tokenizer, context_text: str, evidence_sentence: str, query_text: str):
    evidence_start = context_text.index(evidence_sentence)
    evidence_end = evidence_start + len(evidence_sentence)
    encoded_context = tokenizer(
        context_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoded_context["input_ids"].squeeze(0).to(dtype=torch.long)
    offsets = encoded_context["offset_mapping"].squeeze(0)
    evidence_mask = ((offsets[:, 0] < evidence_end) & (offsets[:, 1] > evidence_start)).to(
        dtype=torch.long
    )
    if int(evidence_mask.sum().item()) == 0:
        raise RuntimeError("Evidence mask is empty after tokenization.")

    query_ids = tokenizer(
        query_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].squeeze(0).to(dtype=torch.long)
    return input_ids, query_ids, evidence_mask


def _build_custom_example(
    *,
    tokenizer,
    rng: random.Random,
    family: str,
    style_lengths: dict[str, list[int]],
    min_tokens: int,
    max_tokens: int,
) -> dict[str, object]:
    payload = _make_family_payload(rng, family)
    style = rng.choice(tuple(STYLE_BANKS.keys()))
    target_tokens = rng.randint(min_tokens, max_tokens)
    frac = _sample_needle_fraction(rng)
    left_target = max(1, int(target_tokens * frac))
    right_target = max(1, target_tokens - left_target)

    left = _sample_filler_text(
        rng=rng,
        style=style,
        target_tokens=left_target,
        style_lengths=style_lengths,
    )
    right = _sample_filler_text(
        rng=rng,
        style=style,
        target_tokens=right_target,
        style_lengths=style_lengths,
    )

    distractors = list(payload.distractors)
    rng.shuffle(distractors)
    joiner = "\n" if style in ("bullet", "log") else " "
    parts: list[str] = [left]
    if distractors and rng.random() < 0.9:
        parts.append(distractors[0])
    parts.append(payload.evidence_sentence)
    if len(distractors) > 1 and rng.random() < 0.9:
        parts.append(distractors[1])
    parts.append(right)
    context = joiner.join(part for part in parts if part.strip())

    input_ids, query_ids, evidence_mask = _encode_with_mask(
        tokenizer,
        context_text=context,
        evidence_sentence=payload.evidence_sentence,
        query_text=payload.query_text,
    )
    return {
        "context_input_ids": input_ids.cpu(),
        "query_ids": query_ids.cpu(),
        "evidence_mask": evidence_mask.cpu(),
        "metadata": {
            "family": payload.family,
            "style": style,
            "answer_text": payload.answer_text,
            "needle_fraction": float(frac),
        },
    }


def _build_baseline_example(*, tokenizer, min_tokens: int, max_tokens: int) -> dict[str, object]:
    context_ids, query_ids, evidence_mask = generate_synthetic_example(
        tokenizer,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    return {
        "context_input_ids": context_ids.cpu().to(dtype=torch.long),
        "query_ids": query_ids.cpu().to(dtype=torch.long),
        "evidence_mask": evidence_mask.cpu().to(dtype=torch.long),
        "metadata": {
            "family": "baseline_apollo",
            "style": "prose",
            "answer_text": "Apollo-77",
            "needle_fraction": None,
        },
    }


def _build_split(
    *,
    name: str,
    tokenizer,
    rng: random.Random,
    count: int,
    families: list[str],
    baseline_ratio: float,
    style_lengths: dict[str, list[int]],
    min_tokens: int,
    max_tokens: int,
) -> list[dict[str, object]]:
    if count <= 0:
        return []
    if not families:
        raise ValueError(f"Split {name!r} has no available families.")

    examples: list[dict[str, object]] = []
    families_no_baseline = [f for f in families if f != "baseline_apollo"]
    can_sample_baseline = "baseline_apollo" in families

    t0 = time.time()
    for idx in range(1, count + 1):
        use_baseline = can_sample_baseline and (rng.random() < baseline_ratio)
        if use_baseline:
            example = _build_baseline_example(
                tokenizer=tokenizer,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
            )
        else:
            if not families_no_baseline:
                families_no_baseline = list(families)
            family = rng.choice(families_no_baseline)
            example = _build_custom_example(
                tokenizer=tokenizer,
                rng=rng,
                family=family,
                style_lengths=style_lengths,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
            )
        examples.append(example)

        if idx == 1 or idx % 25 == 0 or idx == count:
            elapsed = time.time() - t0
            print(f"[{name}] built {idx}/{count} examples ({elapsed:.1f}s elapsed)")
    return examples


def _parse_holdout_families(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        return []
    unknown = sorted(set(items) - set(FAMILY_NAMES))
    if unknown:
        raise ValueError(f"Unknown holdout family names: {unknown}")
    return items


def main() -> None:
    args = _parse_args()
    if args.min_tokens <= 0:
        raise ValueError("--min_tokens must be positive.")
    if args.max_tokens < args.min_tokens:
        raise ValueError("--max_tokens must be >= --min_tokens.")
    if args.train_examples <= 0:
        raise ValueError("--train_examples must be positive.")
    if args.eval_seen_examples < 0 or args.eval_holdout_examples < 0:
        raise ValueError("--eval_*_examples must be >= 0.")
    if not (0.0 <= args.baseline_ratio <= 1.0):
        raise ValueError("--baseline_ratio must be in [0, 1].")

    rng = random.Random(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading tokenizer from: {args.model_id}")
    tokenizer = _load_tokenizer_with_fix(args.model_id)
    style_lengths = _precompute_style_lengths(tokenizer)

    holdout_families = _parse_holdout_families(args.holdout_families)
    train_families = [f for f in FAMILY_NAMES if f not in holdout_families]
    if not train_families:
        raise ValueError("All families are in holdout_families; train_families would be empty.")

    eval_holdout_families = list(holdout_families)
    if not eval_holdout_families and args.eval_holdout_examples > 0:
        print(
            "No holdout families provided; setting eval_holdout_examples to 0 "
            "for this run."
        )
        args.eval_holdout_examples = 0

    print(f"Train families: {train_families}")
    print(f"Holdout families: {eval_holdout_families}")

    train_split = _build_split(
        name="train",
        tokenizer=tokenizer,
        rng=rng,
        count=args.train_examples,
        families=train_families,
        baseline_ratio=args.baseline_ratio,
        style_lengths=style_lengths,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )
    eval_seen_split = _build_split(
        name="eval_seen",
        tokenizer=tokenizer,
        rng=rng,
        count=args.eval_seen_examples,
        families=train_families,
        baseline_ratio=args.baseline_ratio,
        style_lengths=style_lengths,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )
    eval_holdout_split = _build_split(
        name="eval_holdout",
        tokenizer=tokenizer,
        rng=rng,
        count=args.eval_holdout_examples,
        families=eval_holdout_families,
        baseline_ratio=0.0,
        style_lengths=style_lengths,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "c2fab_synth_v2",
        "model_id": args.model_id,
        "seed": args.seed,
        "family_names": list(FAMILY_NAMES),
        "holdout_families": eval_holdout_families,
        "config": {
            "train_examples": args.train_examples,
            "eval_seen_examples": args.eval_seen_examples,
            "eval_holdout_examples": args.eval_holdout_examples,
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
            "baseline_ratio": args.baseline_ratio,
        },
        "splits": {
            "train": train_split,
            "eval_seen": eval_seen_split,
            "eval_holdout": eval_holdout_split,
        },
    }
    torch.save(payload, output_path)

    print("\nSaved dataset:")
    print(f"  path: {output_path}")
    print(f"  train: {len(train_split)}")
    print(f"  eval_seen: {len(eval_seen_split)}")
    print(f"  eval_holdout: {len(eval_holdout_split)}")


if __name__ == "__main__":
    main()
