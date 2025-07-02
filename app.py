import streamlit as st
import torch
from transformers import pipeline, set_seed

# Check if CUDA (GPU) is available
device = 0 if torch.cuda.is_available() else -1

# Cache the model loading to prevent reloading on every run
@st.cache_resource
def load_model():
    try:
        generator = pipeline(
            "text-generation",
            model="EleutherAI/gpt-neo-1.3B",
            device=device
        )
        return generator
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

# Set up the Streamlit interface
st.title("🤖 JAVA Assistant")
st.subheader("Ask anything about Design and Analysis of Algorithms or Operating Systems.")

# User input
user_input = st.text_area("💬 Ask a question:", height=100)

# Generate answer
if st.button("Get Answer"):
    if not user_input.strip():
        st.warning("Please enter a question.")
    elif model is None:
        st.error("Model failed to load. Check internet connection or Hugging Face access.")
    else:
        prompt = (
            "You are an expert tutor for B.Tech students in Design and Analysis of Algorithms and Operating Systems.\n"
            f"Q: {user_input}\nA:"
        )
        with st.spinner("Generating answer..."):
            try:
                response = model(
                    prompt,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    num_return_sequences=1
                )
                generated_text = response[0]["generated_text"]
                answer = generated_text[len(prompt):].strip()

                if not answer or len(answer) < 5:
                    st.warning("The model didn't return a good answer. Try rephrasing your question.")
                else:
                    st.success("Answer:")
                    st.write(answer)

            except Exception as e:
                st.error(f"Error during generation: {e}")
