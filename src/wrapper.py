from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ in (None, ""):
    from modules import C2FAB_Heads
    from patcher import apply_c2fab_patch
else:
    from .modules import C2FAB_Heads
    from .patcher import apply_c2fab_patch


@dataclass
class ChargeFieldMinistral:
    model: Any
    tokenizer: Any
    heads: C2FAB_Heads

    @staticmethod
    def _input_device(model) -> torch.device:
        # With device_map="auto", model.device can be misleading.
        if hasattr(model, "get_input_embeddings"):
            emb = model.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                return emb.weight.device
        return next(model.parameters()).device

    @staticmethod
    def _preferred_dtype() -> torch.dtype:
        # bfloat16 on MPS commonly forces CPU/offload; float16 is the faster path.
        if torch.backends.mps.is_available():
            return torch.float16
        return torch.bfloat16

    @classmethod
    def from_pretrained(
        cls,
        model_id,
        checkpoint_path: str | None = None,
        force_alpha: float | None = None,
    ) -> "ChargeFieldMinistral":
        preferred_dtype = cls._preferred_dtype()

        def _load_tokenizer(source: str, *, local_files_only: bool):
            # For offline/air-gapped runs, slow tokenizer avoids extra
            # fast-tokenizer metadata calls in some transformers versions.
            try:
                return AutoTokenizer.from_pretrained(
                    source,
                    local_files_only=local_files_only,
                    use_fast=False,
                )
            except Exception:
                return AutoTokenizer.from_pretrained(
                    source,
                    local_files_only=local_files_only,
                )

        try:
            tokenizer = _load_tokenizer(model_id, local_files_only=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=preferred_dtype,
                device_map="auto",
            )
        except Exception as online_exc:
            try:
                local_snapshot = snapshot_download(
                    repo_id=model_id,
                    local_files_only=True,
                )
            except Exception as snapshot_exc:
                raise RuntimeError(
                    "Online model/tokenizer load failed and no local snapshot was found.\n"
                    f"Online error: {type(online_exc).__name__}: {online_exc}\n"
                    f"Snapshot error: {type(snapshot_exc).__name__}: {snapshot_exc}"
                ) from snapshot_exc

            try:
                tokenizer = _load_tokenizer(local_snapshot, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(
                    local_snapshot,
                    dtype=preferred_dtype,
                    device_map="auto",
                    local_files_only=True,
                )
            except Exception as offline_exc:
                raise RuntimeError(
                    "Failed to load model/tokenizer from local snapshot cache.\n"
                    f"Snapshot path: {local_snapshot}\n"
                    f"Online error: {type(online_exc).__name__}: {online_exc}\n"
                    f"Offline error: {type(offline_exc).__name__}: {offline_exc}"
                ) from offline_exc
        model.eval()

        trained_heads = C2FAB_Heads(
            hidden_dim=4096,
            D=8,
            num_layers=4,
            num_q_heads=32,
        )

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "heads_state_dict" in checkpoint:
                trained_heads.load_state_dict(checkpoint["heads_state_dict"])
            elif isinstance(checkpoint, dict):
                trained_heads.load_state_dict(checkpoint)
            else:
                raise ValueError(
                    "checkpoint_path must point to a dict-like checkpoint or "
                    "a checkpoint containing 'heads_state_dict'."
                )

        if force_alpha is not None:
            with torch.no_grad():
                trained_heads.alphas.data = (
                    torch.ones_like(trained_heads.alphas.data) * float(force_alpha)
                )

        apply_c2fab_patch(model, trained_heads)
        return cls(model=model, tokenizer=tokenizer, heads=trained_heads)

    def generate(self, prompt: str, **kwargs) -> str:
        input_device = self._input_device(self.model)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(input_device)
        kwargs.setdefault("do_sample", False)
        # For patched biasing, full-sequence attention is more reliable than cache mode.
        kwargs.setdefault("use_cache", False)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **kwargs)
        input_len = inputs["input_ids"].shape[-1]
        new_tokens = generated_ids[0, input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
