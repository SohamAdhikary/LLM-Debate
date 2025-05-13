!pip install -q transformers sentence-transformers torch

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize model (CPU-compatible)
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)

# Set pad token to avoid warnings
tokenizer.pad_token = tokenizer.eos_token

def generate(prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def free_debate(claim):
    # Skeptic
    skeptic_out = generate(f"Question this claim: {claim}\nPotential issues:")
    
    # Advocate
    advocate_out = generate(f"Defend this claim: {claim}\nCounterarguments: {skeptic_out}\nDefense:")
    
    return {
        "claim": claim,
        "skeptic": skeptic_out.replace("Potential issues:", "").strip(),
        "advocate": advocate_out.replace("Defense:", "").strip()
    }

# Run debate
print("🚀 Starting debate (this may take 1-2 minutes on CPU)...")

result = free_debate("Vaccines cause autism")
print("\n💬 Debate Results:")
print(f"Claim: {result['claim']}")
print(f"Skeptic: {result['skeptic']}")
print(f"Advocate: {result['advocate']}")

# Simple evaluation
def evaluate(response):
    return {
        "length_score": len(response.split())/100,
        "evidence_score": 1 if any(w in response.lower() for w in ["study", "research"]) else 0
    }

print("\n📊 Evaluation:", evaluate(result["advocate"]))