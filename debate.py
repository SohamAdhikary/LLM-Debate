from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "distilgpt2"
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        print(f"🚀 Loading model {model_name} (this may take a moment)...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to("cpu")
        # Set pad token to avoid warnings
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

def generate(prompt, max_length=150):
    tokenizer, model = load_model()
    inputs = tokenizer(prompt, return_tensors="pt")
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
    advocate_out = generate(
        f"Defend this claim: {claim}\nCounterarguments: {skeptic_out}\nDefense:"
    )
    
    return {
        "claim": claim,
        "skeptic": skeptic_out.replace("Potential issues:", "").strip(),
        "advocate": advocate_out.replace("Defense:", "").strip()
    }

def evaluate(response):
    return {
        "length_score": len(response.split())/100,
        "evidence_score": 1 if any(w in response.lower() for w in ["study", "research"]) else 0
    }

if __name__ == "__main__":
    print("🚀 Starting debate (this may take a moment on CPU)...")
    result = free_debate("Vaccines cause autism")
    print("\n💬 Debate Results:")
    print(f"Claim: {result['claim']}")
    print(f"Skeptic: {result['skeptic']}")
    print(f"Advocate: {result['advocate']}")
    print("\n📊 Evaluation:", evaluate(result["advocate"]))
