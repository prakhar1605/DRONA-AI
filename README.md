# Drona AI – LLM Interview Platform

Drona AI is an intelligent interview simulation platform that leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to conduct adaptive technical interviews and generate grounded, resume-aware feedback.

## 🚀 Features 
- Adaptive technical interview simulation using LLMs 
- Resume-aware questioning via Retrieval-Augmented Generation (RAG)
- Document-grounded evaluation using uploaded PDF resumes
- Personalized feedback highlighting strengths and improvement areas 
- Real-time interaction through a Streamlit-based UI

## 🧠 System Architecture
1. User uploads resume (PDF)
2. Resume is chunked and embedded into a vector store
3. LLM generates interview questions grounded in retrieved resume chunks
4. Candidate responses are evaluated using retrieved context
5. Feedback is generated with reduced hallucination risk

## 🛠 Tech Stack
- Python
- LangChain
- OpenAI GPT-4
- Vector Databases (Embeddings)
- Streamlit

## 📌 Why RAG?
- Ensures responses are grounded in resume content
- Reduces hallucinations compared to vanilla prompting
- Enables personalized, context-aware interviews

## ⚠️ Limitations
- Dependent on resume quality and structure
- Evaluation quality varies with response length
- Not a replacement for human interviewers

## 🌐 Live Demo
👉 https://www.dronaai.in/

## 📄 Future Improvements
- Multi-round interview scoring
- Role-specific interview tracks
- Fine-grained rubric-based evaluation
- Analytics dashboard for performance tracking
