import streamlit as st
import requests

st.title("💬 DRONACHARYA Chatbot")

user_input = st.text_input("Ask for a study plan (e.g., 'I need a 2-week JEE Physics plan')")

if user_input:
    response = requests.post(
        "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill",
        headers={"Authorization": "Bearer YOUR_HF_TOKEN"},
        json={"inputs": f"Create a study plan for: {user_input}"}
    )
    st.write("AI Tutor:", response.json()[0]['generated_text'])