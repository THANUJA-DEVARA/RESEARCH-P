import streamlit as st
import torch
--force-reinstall transformers==4.40.0


# Detect if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Cache the model loading to avoid reloading on each interaction
@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="EleutherAI/gpt-neo-1.3B",
        device=device,
    )

model = load_model()

st.title("🤖 JAVA")
st.subheader("Ask anything about Design and Analysis of Algorithms or Operating Systems.")

user_input = st.text_area("💬 Ask a question:", height=100)

if st.button("Get Answer"):
    if not user_input.strip():
        st.warning("Please enter a question.")
    else:
        prompt = (
            "You are an expert tutor for B.Tech students in Design and Analysis of Algorithms and Operating Systems.\n"
            f"Q: {user_input}\nA:"
        )
        with st.spinner("Thinking..."):
            response = model(
                prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                num_return_sequences=1
            )
            full_output = response[0]["generated_text"]
            answer = full_output[len(prompt):].strip()

            if not answer or len(answer) < 5:
                st.error("Hmm... couldn't generate a helpful response. Try rephrasing your question.")
            else:
                st.success("Answer:")
                st.write(answer)
