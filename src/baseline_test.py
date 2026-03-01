import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import MODEL_ID

def test_baseline_parity():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map=None,
    ).to("mps") 
       
    prompt = "The secret ingredient in the potion is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 1. Get baseline logits
    with torch.no_grad():
        original_logits = model(**inputs).logits
    
    # 2. TODO: Later, we will apply our C2FAB patch here with alpha=0
    # apply_c2fab_patch(model, alpha=0.0)
    # patched_logits = model(**inputs).logits
    
    # 3. Assert they are identical
    # assert torch.allclose(original_logits, patched_logits, atol=1e-4), "PATCH BROKE THE BASELINE!"
    print("Baseline test ready. We will uncomment this once the patcher is built.")

if __name__ == "__main__":
    test_baseline_parity()