import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "mistralai/Ministral-8B-Instruct-2410"

def main():
    device = "mps"  # or "cuda" if GPU
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map=None
    ).to(device)

    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    # --- Build NIAH prompt (reuse identical code from patched script) ---
    from eval.mini_niah import build_niah_prompt
    prompt = build_niah_prompt(context_length=3000)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.0,
        )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n=== Vanilla Output ===")
    print(result.split("Answer:")[-1].strip())

if __name__ == "__main__":
    main()