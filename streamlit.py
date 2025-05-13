# debate_streamlit.py
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import asyncio

# Fix event loop issues
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Simplified model loader
@st.cache_resource
def load_model():
    model_name = "microsoft/phi-2"
    try:
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
        st.error(f"Model loading failed: {str(e)}")
        return None, None

def generate(model, tokenizer, prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit App
def main():
    st.set_page_config(page_title="LLM Debate System", layout="wide")
    st.title("🧠 LLM Debate System")
    
    model, tokenizer = load_model()
    if model is None:
        return
        
    claim = st.text_area("Enter a controversial claim:", "Vaccines cause autism", height=100)
    
    if st.button("Start Debate", type="primary"):
        with st.spinner("🤖 Agents are debating..."):
            try:
                # Skeptic
                skeptic = generate(model, tokenizer, 
                                 f"Critique this claim: {claim}\nPotential issues:")
                # Advocate
                advocate = generate(model, tokenizer,
                                  f"Defend this claim: {claim}\nCounterarguments: {skeptic}\nDefense:")
                
                st.subheader("💬 Debate Results")
                cols = st.columns(2)
                with cols[0]:
                    st.markdown(f"**Skeptic:**\n\n{skeptic}")
                with cols[1]:
                    st.markdown(f"**Advocate:**\n\n{advocate}")
                
                # Evaluation
                evidence = 1 if any(w in advocate.lower() for w in ["study", "research"]) else 0
                st.metric("Evidence Score", evidence)
                
            except Exception as e:
                st.error(f"Debate failed: {str(e)}")

if __name__ == "__main__":
    main()
