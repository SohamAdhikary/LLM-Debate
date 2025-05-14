import streamlit as st
from transformers import pipeline
import torch

# Configuration - Using the smallest reliable model
MODEL_NAME = "distilgpt2"  # 82M parameters - guaranteed to work
MAX_TOKENS = 60  # Conservative length

def main():
    # Initialize Streamlit first
    st.set_page_config(
        page_title="Debate System",
        page_icon="💬",
        layout="centered"
    )
    st.title("💬 Mini Debate System")

    @st.cache_resource(show_spinner=False)
    def load_model():
        try:
            return pipeline(
                'text-generation',
                model=MODEL_NAME,
                device=torch.device("cpu"),  # Force CPU for maximum compatibility
                torch_dtype=torch.float32
            )
        except Exception as e:
            st.error(f"⚠️ Model loading failed: {str(e)}")
            return None

    # Load model with clear status
    status = st.empty()
    status.info("🚀 Initializing debate system...")
    generator = load_model()
    
    if generator is None:
        st.stop()

    status.empty()  # Clear loading message

    # User interface
    claim = st.text_input(
        "Enter a claim to debate:",
        "Social media improves mental health"
    )

    if st.button("Start Debate"):
        with st.spinner("Generating responses..."):
            try:
                # Skeptic
                skeptic = generator(
                    f"Critique this in one sentence: {claim}",
                    max_length=MAX_TOKENS,
                    num_return_sequences=1,
                    temperature=0.7
                )[0]['generated_text']
                
                # Advocate
                advocate = generator(
                    f"Defend this in one sentence: {claim}",
                    max_length=MAX_TOKENS,
                    num_return_sequences=1,
                    temperature=0.7
                )[0]['generated_text']
                
                # Display results
                st.subheader("Results")
                cols = st.columns(2)
                with cols[0]:
                    st.markdown(f"**Skeptic:**\n\n{skeptic}")
                with cols[1]:
                    st.markdown(f"**Advocate:**\n\n{advocate}")
                
                # Simple evaluation
                st.metric(
                    "Evidence Found",
                    "✅ Yes" if any(w in advocate.lower() for w in ["study", 
"research"]) else "❌ No"
                )
                
            except Exception as e:
                st.error(f"Generation error: {str(e)}")

if __name__ == "__main__":
    main()
