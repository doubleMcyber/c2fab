from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

if __package__ in (None, ""):
    # Supports direct script execution: `python src/test_patch.py`
    from config import MODEL_ID
    from modules import C2FAB_Heads
    from patcher import apply_c2fab_patch
else:
    # Supports module execution: `python -m src.test_patch`
    from .config import MODEL_ID
    from .modules import C2FAB_Heads
    from .patcher import apply_c2fab_patch


def _input_device_for_model(model: AutoModelForCausalLM) -> torch.device:
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight.device
    return next(model.parameters()).device


print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

input_device = _input_device_for_model(model)

print("Initializing random C2FAB heads...")
heads = C2FAB_Heads(hidden_dim=4096, D=8, num_layers=4, num_q_heads=32).to(input_device)

print("Applying patch to layers 32-35...")
apply_c2fab_patch(model, heads)

print("Running dummy forward pass...")
dummy_input = torch.randint(
    low=0,
    high=model.config.vocab_size,
    size=(1, 150),
    device=input_device,
)

try:
    with torch.no_grad():
        outputs = model(dummy_input)
    print(f"✅ SUCCESS! Forward pass completed. Logits shape: {outputs.logits.shape}")
except Exception as e:
    print(f"❌ FAILED! Error in patched attention: {type(e).__name__}: {e}")
