# debate.py (Streamlit Version)
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize model
@st.cache_resource
def load_model():
    model_name = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate(model, tokenizer, prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def free_debate(model, tokenizer, claim):
    skeptic_out = generate(model, tokenizer, f"Question this claim: {claim}\nPotential issues:")
    advocate_out = generate(model, tokenizer, 
                          f"Defend this claim: {claim}\nCounterarguments: {skeptic_out}\nDefense:")
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

# Streamlit UI
st.title("🧠 LLM Debate System")
st.caption("A two-agent debate platform using Phi-2")

claim = st.text_input("Enter a controversial claim:", "Vaccines cause autism")

if st.button("Start Debate"):
    model, tokenizer = load_model()
    with st.spinner("🚀 Agents are debating (this may take 1-2 minutes)..."):
        result = free_debate(model, tokenizer, claim)
    
    st.subheader("💬 Debate Results")
    st.markdown(f"**Claim:** {result['claim']}")
    st.markdown(f"**Skeptic:** {result['skeptic']}")
    st.markdown(f"**Advocate:** {result['advocate']}")
    
    evaluation = evaluate(result["advocate"])
    st.subheader("📊 Evaluation Metrics")
    st.metric("Evidence Score", evaluation["evidence_score"])
    st.metric("Response Length Score", f"{evaluation['length_score']:.2f}")