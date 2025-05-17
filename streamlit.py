import streamlit as st
from transformers import pipeline
import torch
import psutil

MODEL_NAME = "google/flan-t5-base"
TASK = "text2text-generation"
MAX_TOKENS = 150  # a bit longer for detailed output

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

    mem = psutil.virtual_memory()
    st.caption(f"💾 Memory usage: {mem.percent:.1f}%")

    claim = st.text_input(
        "Enter a claim to debate:",
        "College should be free for everyone"
    )

    if st.button("Start Debate"):
        with st.spinner("Generating responses..."):
            try:
                skeptic_prompt = (
                    f'You are a critical analyst debating a claim. Here is the claim: "{claim}".\n'
                    "Provide one clear, logical reason why this claim could be false or misleading.\n"
                    "Example: For \"Homework is harmful,\" a good reason might be "
                    "\"Homework reinforces learning and helps students retain information.\"\n"
                    "Now, your reason:"
                )
                advocate_prompt = (
                    f'You are an expert advocating for a claim. Here is the claim: "{claim}".\n'
                    "Provide one clear, logical reason supporting the claim.\n"
                    "Example: For \"Homework is harmful,\" a good reason might be "
                    "\"Too much homework causes unnecessary stress and reduces free time.\"\n"
                    "Now, your reason:"
                )

                skeptic = generator(skeptic_prompt, max_length=MAX_TOKENS)[0]["generated_text"]
                advocate = generator(advocate_prompt, max_length=MAX_TOKENS)[0]["generated_text"]

                st.subheader("Results")
                cols = st.columns(2)
                with cols[0]:
                    st.markdown(f"**Skeptic:**\n\n{skeptic.strip()}")
                with cols[1]:
                    st.markdown(f"**Advocate:**\n\n{advocate.strip()}")

                st.metric(
                    "Evidence Found",
                    "✅ Yes" if any(w in advocate.lower() for w in ["study", "research", "data", "proof"]) else "❌ No"
                )
            except Exception as e:
                st.error(f"⚠️ Generation error: {str(e)}")

if __name__ == "__main__":
    main()
