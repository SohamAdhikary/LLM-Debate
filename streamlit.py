import streamlit as st
from transformers import pipeline

# Configuration - Using tiny model for reliability
MODEL_NAME = "distilgpt2"  # Very lightweight model
MAX_TOKENS = 50  # Conservative limit

def main():
    # Initialize first to prevent torch issues
    st.set_page_config(
        page_title="Debate System",
        page_icon="💬",
        layout="centered"
    )
    
    @st.cache_resource(show_spinner=False)
    def load_model():
        try:
            return pipeline(
                'text-generation',
                model=MODEL_NAME,
                device=-1  # Force CPU for stability
            )
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            return None

    st.title("💬 Mini Debate System")
    
    # Load model
    with st.spinner("Loading AI components..."):
        generator = load_model()
        if generator is None:
            st.stop()

    # User input
    claim = st.text_input(
        "Enter a claim to debate:",
        "Social media improves society"
    )

    if st.button("Start Debate"):
        with st.spinner("Generating responses..."):
            try:
                # Skeptic
                skeptic = generator(
                    f"Critique this in one sentence: {claim}",
                    max_length=MAX_TOKENS,
                    num_return_sequences=1
                )[0]['generated_text']
                
                # Advocate
                advocate = generator(
                    f"Defend this in one sentence: {claim}",
                    max_length=MAX_TOKENS,
                    num_return_sequences=1
                )[0]['generated_text']
                
                # Display
                st.subheader("Results")
                st.markdown(f"**Claim:** {claim}")
                st.markdown(f"**Skeptic:** {skeptic}")
                st.markdown(f"**Advocate:** {advocate}")
                
            except Exception as e:
                st.error(f"Error generating debate: {str(e)}")

if __name__ == "__main__":
    main()