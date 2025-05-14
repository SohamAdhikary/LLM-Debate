import streamlit as st
from transformers import pipeline
import torch

# Improved model: larger, more coherent, but still lightweight and CPU-compatible
MODEL_NAME = "tiiuae/falcon-rw-1b"  # Better than distilgpt2
MAX_TOKENS = 100  # Increased for more complete answers

def main():
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
                device=torch.device("cpu"),
                torch_dtype=torch.float32
            )
        except Exception as e:
            st.error(f"⚠️ Model loading failed: {str(e)}")
            return None

    # Load model with status indicator
    status = st.empty()
    status.info("🚀 Initializing debate system...")
    generator = load_model()
    
    if generator is None:
        st.stop()
    status.empty()

    claim = st.text_input(
        "Enter a claim to debate:",
        "Social media improves mental health"
    )

    if st.button("Start Debate"):
        with st.spinner("Generating responses..."):
            try:
                skeptic = generator(
                    f"Critique this in one sentence: {claim}",
                    max_length=MAX_TOKENS,
                    num_return_sequences=1,
                    temperature=0.8,
                    top_p=0.95
                )[0]['generated_text']

                advocate = generator(
                    f"Defend this in one sentence: {claim}",
                    max_length=MAX_TOKENS,
                    num_return_sequences=1,
                    temperature=0.8,
                    top_p=0.95
                )[0]['generated_text']

                # Safeguard against model repetition
                if any(skeptic.lower().count(p) > 3 for p in ["it's", "is a", claim.lower()]):
                    skeptic = "⚠️ The model response was repetitive. Please rephrase the claim."

                if any(advocate.lower().count(p) > 3 for p in ["it's", "is a", claim.lower()]):
                    advocate = "⚠️ The model response was repetitive. Please rephrase the claim."

                st.subheader("Results")
                cols = st.columns(2)
                with cols[0]:
                    st.markdown(f"**Skeptic:**\n\n{skeptic}")
                with cols[1]:
                    st.markdown(f"**Advocate:**\n\n{advocate}")

                st.metric(
                    "Evidence Found",
                    "✅ Yes" if any(w in advocate.lower() for w in ["study", "research"]) else "❌ No"
                )

            except Exception as e:
                st.error(f"Generation error: {str(e)}")

if __name__ == "__main__":
    main()
