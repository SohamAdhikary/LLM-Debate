import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# Configuration
MODEL_NAME = "microsoft/phi-2"
MAX_TOKENS = 80  # Reduced for stability

@st.cache_resource(ttl=3600, show_spinner=False)  # Cache for 1 hour
def load_model():
    try:
        # Initialize with progress
        progress = st.progress(0, text="🚀 Loading AI model...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        progress.progress(30)
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        progress.progress(70)
        
        tokenizer.pad_token = tokenizer.eos_token
        progress.progress(100)
        time.sleep(0.5)  # Let progress complete
        
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Model failed to load: {str(e)}")
        return None, None

def main():
    # App config
    st.set_page_config(
        page_title="LLM Debate System",
        page_icon="🧠",
        layout="centered"
    )
    st.title("🧠 LLM Debate System")
    
    # Load model with visual feedback
    model, tokenizer = load_model()
    if model is None:
        st.stop()
    
    # User input
    claim = st.text_area(
        "Enter a controversial claim:",
        "Vaccines cause autism",
        height=100
    )
    
    if st.button("Start Debate", type="primary"):
        with st.spinner("🤖 Agents are debating..."):
            try:
                # Generate debate
                skeptic = generate(
                    model, tokenizer,
                    f"Critique this claim concisely: {claim}\nIssues:",
                    MAX_TOKENS
                )
                
                advocate = generate(
                    model, tokenizer,
                    f"Defend this claim concisely: {claim}\nRebuttal to: {skeptic}\nEvidence:",
                    MAX_TOKENS
                )
                
                # Display results
                st.subheader("💬 Debate Results")
                st.markdown(f"**Claim:** {claim}")
                with st.expander("Skeptic's Arguments"):
                    st.write(skeptic)
                with st.expander("Advocate's Defense"):
                    st.write(advocate)
                
                # Evaluation
                st.metric(
                    "Evidence Quality",
                    "✅ Found" if any(w in advocate.lower() for w in ["study", "research"]) else "❌ Missing"
                )
                
            except Exception as e:
                st.error(f"Debate failed: {str(e)}")

def generate(model, tokenizer, prompt, max_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    main()
