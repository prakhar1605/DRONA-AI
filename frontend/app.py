import streamlit as st
from datetime import datetime, timedelta

st.title("📅 DronaAI Study Planner")

# Inputs
subjects = st.multiselect("Subjects", ["Math", "Physics", "Chemistry"])
days = st.slider("Study Days", 1, 30, 7)
level = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])

if st.button("Generate Plan"):
    plan = []
    start_date = datetime.now()
    for day in range(days):
        plan.append(f"📌 **Day {day+1} ({start_date.strftime('%b %d')}):**")
        plan.append(f"- Study {subjects[0]} (Focus: {level})")
        start_date += timedelta(days=1)
    
    st.success("\n".join(plan))