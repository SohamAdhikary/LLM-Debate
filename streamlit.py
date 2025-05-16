import streamlit as st
from transformers import pipeline
import torch
import psutil

# Configuration - Tiny, fast, and compatible model
MODEL_NAME = "mrm8488/t5-base-finetuned-common_gen"
MAX_TOKENS = 60  # Conservative for stability

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
                'text-generation',
                model=MODEL_NAME,
                device=torch.device("cpu"),
                torch_dtype=torch.float32
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

    # Show memory usage (optional)
    mem = psutil.virtual_memory()
    st.caption(f"💾 Memory usage: {mem.percent}%")

    # User input
    claim = st.text_input(
        "Enter a claim to debate:",
        "Climate change is a hoax"
    )

    if st.button("Start Debate"):
        with st.spinner("Generating responses..."):
            try:
                skeptic = generator(
                    f"Critique this in one sentence: {claim}",
                    max_length=MAX_TOKENS,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9
                )[0]['generated_text']

                advocate = generator(
                    f"Defend this in one sentence: {claim}",
                    max_length=MAX_TOKENS,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9
                )[0]['generated_text']

                st.subheader("Results")
                cols = st.columns(2)
                with cols[0]:
                    st.markdown(f"**Skeptic:**\n\n{skeptic}")
                with cols[1]:
                    st.markdown(f"**Advocate:**\n\n{advocate}")

                # Simple evidence check
                st.metric(
                    "Evidence Found",
                    "✅ Yes" if any(w in advocate.lower() for w in ["study", "research"]) else "❌ No"
                )

            except Exception as e:
                st.error(f"⚠️ Generation error: {str(e)}")

if __name__ == "__main__":
    main()
