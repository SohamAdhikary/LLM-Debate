import streamlit as st
from transformers import pipeline
import torch
import psutil

# Configuration - Stronger instruction-tuned model
MODEL_NAME = "google/flan-t5-base"
TASK = "text2text-generation"
MAX_TOKENS = 120  # Longer responses supported by this model

def main():
    st.set_page_config(
        page_title="Debate System",
        page_icon="💬",
        layout="centered"
    )
    st.title("💬 Mini Debate System")

    @st.cache_resource(show_spinner=True)
    def load_model():
        try:
            return pipeline(
                TASK,
                model=MODEL_NAME,
                device=torch.device("cpu")
            )
        except Exception as e:
            st.error(f"⚠️ Model loading failed: {str(e)}")
            return None

    status = st.empty()
    status.info("🚀 Initializing debate system...")
    generator = load_model()
    if generator is None:
        st.stop()
    status.empty()

    # Show memory usage
    mem = psutil.virtual_memory()
    st.caption(f"💾 Memory usage: {mem.percent:.1f}%")

    # User input
    claim = st.text_input(
        "Enter a claim to debate:",
        "College should be free for everyone"
    )

    if st.button("Start Debate"):
        with st.spinner("Generating responses..."):
            try:
                # Best prompt format for skeptic and advocate
                skeptic_prompt = (
                    f'You are a critical analyst evaluating the following claim: "{claim}". '
                    'In one sentence, present a logical counterargument based on reasoning or evidence.'
                )
                advocate_prompt = (
                    f'You are an expert supporting the following claim: "{claim}". '
                    'In one sentence, provide a compelling argument or supporting evidence.'
                )

                skeptic = generator(skeptic_prompt, max_length=MAX_TOKENS)[0]["generated_text"]
                advocate = generator(advocate_prompt, max_length=MAX_TOKENS)[0]["generated_text"]

                st.subheader("Results")
                cols = st.columns(2)
                with cols[0]:
                    st.markdown(f"**Skeptic:**\n\n{skeptic}")
                with cols[1]:
                    st.markdown(f"**Advocate:**\n\n{advocate}")

                # Simple evidence check
                st.metric(
                    "Evidence Found",
                    "✅ Yes" if any(w in advocate.lower() for w in ["study", "research", "data", "proof"]) else "❌ No"
                )

            except Exception as e:
                st.error(f"⚠️ Generation error: {str(e)}")

if __name__ == "__main__":
    main()
