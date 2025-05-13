import streamlit as st
from transformers import pipeline
import time

# Configuration - Using a smaller model
MODEL_NAME = "facebook/opt-1.3b"  # More lightweight than Phi-2
MAX_TOKENS = 60  # Further reduced for stability

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        progress = st.progress(0, text="🚀 Loading AI model...")
        generator = pipeline(
            'text-generation', 
            model=MODEL_NAME,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        progress.progress(100)
        return generator
    except Exception as e:
        st.error(f"❌ Model failed to load: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="LLM Debate System",
        page_icon="🧠",
        layout="centered"
    )
    st.title("🧠 Debate System Lite")
    
    generator = load_model()
    if generator is None:
        st.stop()
    
    claim = st.text_area(
        "Enter a claim:",
        "Vaccines cause autism",
        height=100
    )
    
    if st.button("Start Debate"):
        with st.spinner("🤖 Agents are debating..."):
            try:
                # Skeptic
                skeptic = generator(
                    f"Critique this claim: {claim}\nIssues:",
                    max_length=MAX_TOKENS,
                    do_sample=True
                )[0]['generated_text']
                
                # Advocate
                advocate = generator(
                    f"Defend this claim: {claim}\nRebuttal to: {skeptic}\nEvidence:",
                    max_length=MAX_TOKENS,
                    do_sample=True
                )[0]['generated_text']
                
                # Display
                st.subheader("💬 Results")
                with st.expander("Skeptic"):
                    st.write(skeptic.split("Issues:")[-1].strip())
                with st.expander("Advocate"):
                    st.write(advocate.split("Evidence:")[-1].strip())
                
                # Simple evaluation
                evidence = "✅" if any(w in advocate.lower() for w in ["study", "research"]) else "❌"
                st.metric("Evidence Found", evidence)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    import torch  # Moved here to prevent early initialization
    main()
