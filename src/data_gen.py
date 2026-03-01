from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


EVIDENCE_SENTENCE = "The secret project codename is Apollo-77."
QUERY_TEXT = "What is the secret project codename?"

WIKI_LIKE_SENTENCES = [
    "The committee published an annual report describing regional trade patterns and civic infrastructure.",
    "Researchers documented seasonal migration across mountain corridors and compared findings with earlier surveys.",
    "Historical sources note that the harbor expanded rapidly after the railway connection was completed.",
    "A public registry recorded demographic changes, education levels, and household employment across districts.",
    "Several independent teams reviewed climate records and found gradual shifts in rainfall timing.",
    "The archive includes correspondence, technical drawings, and policy memoranda from multiple decades.",
    "Geographers mapped river basins in detail and measured sediment transport during peak flow events.",
    "The city council funded restoration projects focused on schools, libraries, and transit stations.",
    "Analysts compared manufacturing output before and after reforms to estimate long-term productivity effects.",
    "A follow-up assessment summarized public health indicators and recommended targeted community programs.",
]


def _build_background_sentences(
    tokenizer: PreTrainedTokenizerBase,
    target_tokens: int,
) -> list[str]:
    """Generate enough generic sentences to reach at least target_tokens."""
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be positive, got {target_tokens}.")

    pieces: list[str] = []
    token_count = 0
    while token_count < target_tokens:
        sentence = random.choice(WIKI_LIKE_SENTENCES)
        pieces.append(sentence)
        sentence_ids = tokenizer(sentence, add_special_tokens=False)["input_ids"]
        token_count += len(sentence_ids)
    return pieces


def generate_synthetic_example(
    tokenizer: PreTrainedTokenizerBase,
    min_tokens: int = 2000,
    max_tokens: int = 4000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build one synthetic long-context contrastive example.

    Returns:
        input_ids: [seq_len] token ids for the context
        query_ids: [query_len] token ids for the query
        evidence_mask: [seq_len] binary tensor (1 only on evidence tokens)
    """
    if min_tokens <= 0:
        raise ValueError(f"min_tokens must be positive, got {min_tokens}.")
    if max_tokens < min_tokens:
        raise ValueError(
            f"max_tokens must be >= min_tokens, got min={min_tokens}, max={max_tokens}."
        )

    target_seq_tokens = random.randint(min_tokens, max_tokens)
    evidence_ids = tokenizer(EVIDENCE_SENTENCE, add_special_tokens=False)["input_ids"]
    background_target = max(1, target_seq_tokens - len(evidence_ids))

    background_sentences = _build_background_sentences(tokenizer, background_target)
    if len(background_sentences) < 3:
        background_sentences.extend(random.choice(WIKI_LIKE_SENTENCES) for _ in range(3))

    # Insert evidence in the middle band to avoid trivial beginning/end positions.
    left = max(1, len(background_sentences) // 3)
    right = max(left, (2 * len(background_sentences)) // 3)
    insert_idx = random.randint(left, right)

    prefix = " ".join(background_sentences[:insert_idx]).strip()
    suffix = " ".join(background_sentences[insert_idx:]).strip()
    context_parts = [part for part in (prefix, EVIDENCE_SENTENCE, suffix) if part]
    full_context = " ".join(context_parts)

    evidence_start = full_context.index(EVIDENCE_SENTENCE)
    evidence_end = evidence_start + len(EVIDENCE_SENTENCE)

    # encoded_context["input_ids"]:[1, seq_len]
    # encoded_context["offset_mapping"]:[1, seq_len, 2]
    encoded_context = tokenizer(
        full_context,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    input_ids = encoded_context["input_ids"].squeeze(0)  # input_ids:[seq_len]
    offsets = encoded_context["offset_mapping"].squeeze(0)  # offsets:[seq_len, 2]

    evidence_mask = (
        (offsets[:, 0] < evidence_end) & (offsets[:, 1] > evidence_start)
    ).to(dtype=torch.long)  # evidence_mask:[seq_len]

    if int(evidence_mask.sum().item()) == 0:
        raise RuntimeError(
            "Evidence mask is empty. Token/character alignment failed unexpectedly."
        )

    query_ids = tokenizer(
        QUERY_TEXT,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].squeeze(0)  # query_ids:[query_len]

    return input_ids, query_ids, evidence_mask


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Ministral-8B-Instruct-2410")
    input_ids, query_ids, evidence_mask = generate_synthetic_example(tokenizer)

    seq_len = int(input_ids.numel())
    evidence_token_ids = input_ids[evidence_mask.bool()]
    evidence_tokens = tokenizer.convert_ids_to_tokens(evidence_token_ids.tolist())
    evidence_decoded = tokenizer.decode(
        evidence_token_ids.tolist(),
        skip_special_tokens=False,
    )
    query_decoded = tokenizer.decode(query_ids.tolist(), skip_special_tokens=False)

    print(f"Total sequence length: {seq_len}")
    print(f"Query text: {query_decoded!r}")
    print(f"Evidence token count: {int(evidence_mask.sum().item())}")
    print("Masked evidence tokens:")
    print(evidence_tokens)
    print("Decoded text where evidence_mask == 1:")
    print(repr(evidence_decoded))
