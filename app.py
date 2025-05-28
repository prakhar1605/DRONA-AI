import streamlit as st
import requests

# Configure page
st.set_page_config(page_title="DronaAI Chat", page_icon="💬")

# Hugging Face API call
def ask_ai(question):
    API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
    headers = {"Authorization": f"Bearer {st.secrets.HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": question})
    return response.json()[0]['generated_text']

# Chat UI
st.title("DronaAI Tutor 💡")
user_input = st.text_input("Ask any study question:")

if user_input:
    with st.spinner("Thinking..."):
        answer = ask_ai(user_input)
    st.success(f"**AI Tutor:** {answer}")
