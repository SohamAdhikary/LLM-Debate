import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import asyncio

# Fix event loop and torch path issues
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Lightweight model loader
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model_name = "microsoft/phi-2"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    except Exception as e:
        st.error(f"⚠️ Model loading failed: {str(e)}")
        return None, None

# Streamlit App
def main():
    st.set_page_config(page_title="LLM Debate", layout="centered")
    st.title("🧠 LLM Debate System")
    
    model, tokenizer = load_model()
    if model is None:
        st.stop()
    
    claim = st.text_area("Enter a claim:", "Vaccines cause autism", height=100)
    
    if st.button("Start Debate", type="primary"):
        with st.spinner("🤖 Agents are debating..."):
            try:
                # Skeptic
                skeptic = generate(model, tokenizer, 
                                 f"Critique: {claim}\nIssues:")
                # Advocate
                advocate = generate(model, tokenizer,
                                  f"Defend: {claim}\nRebuttal to: {skeptic}\nEvidence:")
                
                st.subheader("💬 Results")
                st.markdown(f"**Skeptic:**\n\n{skeptic}")
                st.markdown(f"**Advocate:**\n\n{advocate}")
                
                # Simple evaluation
                evidence = "✅" if any(w in advocate.lower() for w in ["study", "research"]) else "❌"
                st.metric("Evidence Found", evidence)
                
            except Exception as e:
                st.error(f"Debate failed: {str(e)}")

def generate(model, tokenizer, prompt, max_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    main()
