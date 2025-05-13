import streamlit as st
from transformers import pipeline
import torch

# Configuration
MODEL_NAME = "gpt2"  # Small but more capable than distilgpt2
MAX_TOKENS = 80  # Balanced length

def main():
    st.set_page_config(
        page_title="LLM Debate System",
        page_icon="🧠",
        layout="centered"
    )
    st.title("🧠 LLM Debate System")
    
    @st.cache_resource(show_spinner=False)
    def load_model():
        try:
            return pipeline(
                'text-generation',
                model=MODEL_NAME,
                device="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            return None

    # Load model with progress
    with st.spinner("Loading debate AI..."):
        generator = load_model()
        if generator is None:
            st.stop()
    
    # User interface
    claim = st.text_area(
        "Enter a controversial claim:",
        "Vaccines cause autism",
        height=100
    )
    
    if st.button("Start Debate", type="primary"):
        with st.spinner("🤖 Agents are debating..."):
            try:
                # Skeptic
                skeptic = generator(
                    f"Critique this claim concisely: {claim}\nKey issues:",
                    max_length=MAX_TOKENS,
                    do_sample=True,
                    temperature=0.7
                )[0]['generated_text'].split("Key issues:")[-1].strip()
                
                # Advocate
                advocate = generator(
                    f"Defend this claim concisely: {claim}\nCounterarguments: {skeptic}\nEvidence:",
                    max_length=MAX_TOKENS,
                    do_sample=True,
                    temperature=0.7
                )[0]['generated_text'].split("Evidence:")[-1].strip()
                
                # Display results
                st.subheader("💬 Debate Results")
                cols = st.columns(2)
                with cols[0]:
                    st.markdown(f"**Skeptic:**\n\n{skeptic}")
                with cols[1]:
                    st.markdown(f"**Advocate:**\n\n{advocate}")
                
                # Evaluation
                st.metric(
                    "Evidence Quality",
                    "✅ Found" if any(w in advocate.lower() for w in ["study", "research"]) else "❌ Missing"
                )
                
            except Exception as e:
                st.error(f"Debate failed: {str(e)}")

if __name__ == "__main__":
    main()
