import torch
import inspect
from transformers import AutoModelForCausalLM
from src.config import MODEL_ID

print(f"Loading {MODEL_ID} (this may take a minute to download if it's your first time)...")

# Load the model in bfloat16 to save memory
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    device_map="auto" # or "mps" if on Mac
)

print("\n--- MODEL ARCHITECTURE ---")
print(f"Num Query Heads: {model.config.num_attention_heads} (Expected: 32)")
print(f"Num KV Heads: {model.config.num_key_value_heads} (Expected: 8)")
print(f"Hidden Size: {model.config.hidden_size}")

attn_module = model.model.layers[0].self_attn
print(f"Attention Class: {type(attn_module).__name__}")

# Extract the EXACT source code of the forward pass
forward_source = inspect.getsource(attn_module.forward)

# Save it to a reference file for Cursor
with open("MISTRAL_ATTN_REFERENCE.py", "w") as f:
    f.write("# THIS IS THE EXACT SOURCE CODE FROM HUGGINGFACE.\n")
    f.write("# DO NOT GUESS HOW MISTRAL WORKS. COPY THIS STRUCTURE.\n\n")
    f.write(forward_source)

print("\n✅ Introspection complete. The exact attention source code has been saved to MISTRAL_ATTN_REFERENCE.py")