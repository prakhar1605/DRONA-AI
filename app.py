import os
import re
import time
import json
from typing import List, Generator, Optional
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI

# ---------------- ENV / CLIENT ----------------
api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("OPENROUTER_API_KEY not found. Please set it in Streamlit secrets.")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


# ---------------- HELPERS ----------------
def extract_text_from_pdf(file) -> str:
    if not file:
        return ""
    reader = PdfReader(file)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def ask_llm(prompt: str) -> str:
    """Non-streaming call (returns full response)."""
    resp = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role":"system", "content":"You are an honest exam question generator and assessment advisor."},
            {"role":"user", "content":prompt}
        ],
        temperature=0.7  # Increased for more variety
    )
    return resp.choices[0].message.content

def ask_llm_stream(prompt: str) -> Optional[Generator[str, None, None]]:
    """Streaming generator."""
    try:
        stream_resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role":"system", "content":"You are an honest, concise exam-feedback generator. Produce structured markdown output with bullet points and short paragraphs."},
                {"role":"user", "content":prompt}
            ],
            temperature=0.15,
            stream=True
        )
        
        def gen():
            for chunk in stream_resp:
                try:
                    if hasattr(chunk.choices[0], "delta") and chunk.choices[0].delta:
                        text = chunk.choices[0].delta.content
                        if text:
                            yield text
                except Exception:
                    continue
        
        return gen()
    except Exception:
        return None

def safe_json(text: str):
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        raise ValueError("No JSON array found in model response.")
    return json.loads(match.group())

def normalize_options(opts: List[str]) -> List[str]:
    return [o.strip() for o in (opts or [])]

# ---------------- SESSION DEFAULTS ----------------
if "page" not in st.session_state:
    st.session_state.page = "setup"
if "quiz" not in st.session_state:
    st.session_state.quiz = []
if "current" not in st.session_state:
    st.session_state.current = 0
if "answers" not in st.session_state:
    st.session_state.answers = []
if "topics" not in st.session_state:
    st.session_state.topics = []
if "audience" not in st.session_state:
    st.session_state.audience = "School Student"
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "num_q" not in st.session_state:
    st.session_state.num_q = 10

st.set_page_config(page_title="DRONA AI - LLM Interview Platform", layout="centered")

# ---------------- MINIMAL CLEAN STYLES ----------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f0f0;
        margin-bottom: 2rem;
    }
    .question-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #1976D2;
    }
    .metric-box {
        background: transparent;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #444;
        text-align: center;
    }
    .topic-tag {
        display: inline-block;
        background: #e3f2fd;
        color: #1976D2;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 4px;
        font-size: 0.9rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        padding: 0.75rem;
        font-weight: 500;
    }
    /* Remove white boxes */
    [data-testid="column"] > div > div > div > div {
        background: transparent !important;
    }
    .stMetric {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HELPERS ----------------
def reset_quiz_state():
    st.session_state.quiz = []
    st.session_state.current = 0
    st.session_state.answers = []
    st.session_state.topics = []
    st.session_state.page = "setup"
    st.session_state.audience = "School Student"
    st.session_state.is_generating = False
    st.session_state.num_q = 10

# ---------------- HEADER ----------------
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("📚 DRONA AI - LLM Interview Platform")
st.markdown("Intelligent interview simulation with adaptive technical assessments")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SETUP PAGE ----------------
if st.session_state.page == "setup":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload PDF (Optional)")
        uploaded = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")
        
        st.subheader("Add Topics")
        col_input, col_btn = st.columns([3, 1])
        with col_input:
            new_topic = st.text_input("Enter topic name", placeholder="e.g., Python programming", label_visibility="collapsed")
        with col_btn:
            if st.button("Add", type="secondary"):
                if new_topic.strip() and new_topic.strip() not in st.session_state.topics:
                    st.session_state.topics.append(new_topic.strip())
                    st.rerun()
        
        if st.session_state.topics:
            st.write("**Selected Topics:**")
            for topic in st.session_state.topics:
                st.markdown(f'<span class="topic-tag">{topic}</span>', unsafe_allow_html=True)
            
            if st.button("Clear All Topics", type="secondary"):
                st.session_state.topics = []
                st.rerun()
    
    with col2:
        st.subheader("Quiz Settings")
        
        st.session_state.audience = st.selectbox(
            "Audience Level",
            ["School Student", "College Student", "Intern / Fresher", "Professional"]
        )
        
        target_role = st.selectbox(
            "Target Role",
            [
                "Learning / School–College Exam",
                "Software Engineer",
                "Data Scientist",
                "Data Analyst",
                "AI Engineer",
                "Web Developer",
                "DevOps Engineer",
                "Product Manager"
            ]
        )
        
        difficulty = st.selectbox(
            "Difficulty Level",
            ["Easy", "Moderate", "Tough", "Mixed (All levels)"]
        )
        
        st.subheader("Number of Questions")
        st.session_state.num_q = st.slider(
            "Select count",
            min_value=5,
            max_value=50,
            value=st.session_state.num_q,
            step=5,
            label_visibility="collapsed"
        )
        st.write(f"**{st.session_state.num_q} questions**")
    
    st.divider()
    
    # Generate button
    if st.session_state.is_generating:
        with st.status("Generating quiz questions...", expanded=True) as status:
            st.write(f"Creating {st.session_state.num_q} questions")
            st.write("This may take a moment...")
            time.sleep(1)
    else:
        if st.button("**🎯 Generate Quiz**", type="primary", use_container_width=True):
            if not st.session_state.topics and not uploaded:
                st.error("Please add at least one topic or upload a PDF")
            else:
                st.session_state.is_generating = True
                st.rerun()
    
    # Actual generation process
    if st.session_state.is_generating and st.session_state.page == "setup":
        with st.spinner("Processing..."):
            try:
                context = extract_text_from_pdf(uploaded) if uploaded else ""
                audience_instr = (
                    "Use very simple school-level language." if st.session_state.audience == "School Student"
                    else "Use clear, exam-style language suitable for college/intern/professional level."
                )
                
                # Enhanced prompt for variety and proper difficulty matching
                prompt = f"""
You must return a JSON array of {st.session_state.num_q} question objects. Each question object:
- question (string)
- options (array of 4 strings)
- correct_options (array of the correct option texts)
- topic (one of the supplied topics)
- difficulty (Easy/Moderate/Tough)
- marks (integer: Easy/Moderate -> 5, Tough -> 10)

Context:
Audience rule: {audience_instr}
Target role: {target_role}
Difficulty preference: {difficulty}
Topics: {st.session_state.topics}
Reference text (optional): {context}

CRITICAL RULES:
1. For difficulty "{difficulty}":
   - If "Easy": ALL questions must be Easy difficulty, basic concepts only
   - If "Moderate": ALL questions must be Moderate difficulty, intermediate concepts
   - If "Tough": ALL questions must be Tough difficulty, advanced/complex concepts
   - If "Mixed": Include variety of Easy, Moderate, and Tough

2. Question variety:
   - Create DIFFERENT types of questions (conceptual, practical, scenario-based, code-based)
   - Use DIFFERENT phrasings and question styles
   - Cover DIFFERENT aspects of each topic
   - Avoid repetitive patterns

3. Correct options:
   - Easy & Moderate -> exactly 1 correct option
   - Tough -> 2-3 correct options allowed

4. Distribution:
   - If multiple topics, spread questions evenly across topics
   - Each question should test different skills/knowledge

Return ONLY a JSON array, no commentary.
"""
                
                raw = ask_llm(prompt)
                quiz = safe_json(raw)
                
                for q in quiz:
                    q['options'] = normalize_options(q.get('options', []))
                    q['correct_options'] = [c.strip() for c in q.get('correct_options', [])]
                    if 'marks' not in q:
                        q['marks'] = 10 if q.get('difficulty','Tough').lower().startswith('t') else 5
                
                st.session_state.quiz = quiz
                st.session_state.current = 0
                st.session_state.answers = []
                st.session_state.page = "quiz"
                st.session_state.is_generating = False
                st.success("✅ Quiz generated successfully!")
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.is_generating = False

# ---------------- QUIZ PAGE ----------------
elif st.session_state.page == "quiz":
    quiz = st.session_state.quiz
    idx = st.session_state.current
    q = quiz[idx]
    total = len(quiz)
    
    # Progress bar and navigation
    st.progress((idx+1)/total, text=f"Question {idx+1} of {total}")
    
    col_nav = st.columns(5)
    for i in range(min(5, total - idx)):
        with col_nav[i]:
            if st.button(f"Q{idx+i+1}", type="primary" if i==0 else "secondary"):
                st.session_state.current = idx + i
                st.rerun()
    
    st.divider()
    
    # Question display
    st.markdown('<div class="question-box">', unsafe_allow_html=True)
    
    col_top = st.columns(3)
    with col_top[0]:
        st.markdown(f"**Topic:** {q.get('topic', '-')}")
    with col_top[1]:
        st.markdown(f"**Difficulty:** {q.get('difficulty', '-')}")
    with col_top[2]:
        st.markdown(f"**Marks:** {q.get('marks', 5)}")
    
    st.markdown(f"### {q.get('question')}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.write("")
    st.write("**Select your answer:**")
    
    # Answer selection
    correct_count = len(q.get('correct_options', []))
    is_tough = (q.get('difficulty','').lower() == 'tough') or (correct_count > 1)
    
    if is_tough:
        st.write("*(Multiple correct answers)*")
        selected = st.multiselect(
            "Choose all that apply",
            q.get('options', []),
            label_visibility="collapsed",
            key=f"q_{idx}"
        )
    else:
        selected = st.radio(
            "Choose one option",
            q.get('options', []),
            label_visibility="collapsed",
            key=f"q_{idx}"
        )
    
    st.divider()
    
    # Navigation buttons
    col_btns = st.columns(3)
    
    with col_btns[0]:
        if idx > 0:
            if st.button("⬅️ Previous"):
                st.session_state.current -= 1
                st.rerun()
    
    with col_btns[1]:
        if st.button("✅ Submit Answer", type="primary"):
            if (selected is None) or (isinstance(selected, list) and len(selected) == 0):
                st.warning("Please select at least one option")
                st.stop()
            else:
                chosen_set = set(selected) if isinstance(selected, list) else {selected}
                correct_set = set(q.get('correct_options', []))
                marks_total = int(q.get('marks', 5))
                
                if len(correct_set) == 1:
                    earned = marks_total if (chosen_set == correct_set) else 0
                    correct_flag = (chosen_set == correct_set)
                else:
                    matched = len(chosen_set & correct_set)
                    earned = marks_total * (matched / max(1, len(correct_set)))
                    correct_flag = (chosen_set == correct_set)
                
                st.session_state.answers.append({
                    "index": idx,
                    "question": q.get('question'),
                    "chosen": list(chosen_set),
                    "correct": list(correct_set),
                    "marks_earned": round(earned, 2),
                    "marks_total": marks_total,
                    "topic": q.get('topic'),
                    "difficulty": q.get('difficulty'),
                    "correct_flag": correct_flag
                })
                
                st.session_state.current += 1
                if st.session_state.current >= total:
                    st.session_state.page = "result"
                st.rerun()
    
    with col_btns[2]:
        if idx < total - 1:
            if st.button("Skip ➡️"):
                # Mark as skipped (no answer)
                st.session_state.answers.append({
                    "index": idx,
                    "question": q.get('question'),
                    "chosen": [],
                    "correct": q.get('correct_options', []),
                    "marks_earned": 0,
                    "marks_total": q.get('marks', 5),
                    "topic": q.get('topic'),
                    "difficulty": q.get('difficulty'),
                    "correct_flag": False
                })
                st.session_state.current += 1
                if st.session_state.current >= total:
                    st.session_state.page = "result"
                st.rerun()

# ---------------- RESULT PAGE ----------------
elif st.session_state.page == "result":
    answers = st.session_state.answers
    total_marks = sum(a['marks_total'] for a in answers) if answers else 0
    earned_marks = sum(a['marks_earned'] for a in answers) if answers else 0
    percent = round((earned_marks / total_marks) * 100, 1) if total_marks > 0 else 0
    num_correct = sum(1 for a in answers if a.get('correct_flag'))
    total_q = len(answers)
    
    st.title("🎯 Interview Assessment Results")
    st.divider()
    
    # Summary metrics - simplified without boxes
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### Score\n# {percent}%")
    with col2:
        st.markdown(f"### Marks\n# {earned_marks}/{total_marks}")
    with col3:
        st.markdown(f"### Correct\n# {num_correct}/{total_q}")
    
    st.divider()
    
    # Topic-wise performance
    st.subheader("📊 Topic Performance")
    topic_scores = {}
    for a in answers:
        t = a.get('topic','Other')
        topic_scores.setdefault(t, []).append(a)
    
    for t, qs in topic_scores.items():
        score = sum(q['marks_earned'] for q in qs)
        total = sum(q['marks_total'] for q in qs)
        pct = round((score / total) * 100) if total else 0
        cols = st.columns([1, 3, 1])
        with cols[0]:
            st.write(f"**{t}**")
        with cols[1]:
            st.progress(pct/100, text=f"{pct}%")
        with cols[2]:
            st.write(f"{score}/{total}")
    
    st.divider()
    
    # Detailed review
    st.subheader("📋 Question-wise Performance Analysis")
    
    with st.expander("View all questions and answers", expanded=False):
        for a in answers:
            qidx = a.get('index', 0) + 1
            ok = a.get('correct_flag', False)
            
            st.markdown(f"**Q{qidx}.** {a.get('question')}")
            st.caption(f"*{a.get('difficulty')} | {a.get('topic')} | {a.get('marks_earned')}/{a.get('marks_total')} marks*")
            
            col_ans = st.columns(2)
            with col_ans[0]:
                st.write("**Your answer:**")
                chosen = a.get('chosen', [])
                if chosen:
                    for ans in chosen:
                        st.write(f"• {ans}")
                else:
                    st.write("*(Skipped)*")
            
            with col_ans[1]:
                st.write("**Correct answer:**")
                for ans in a.get('correct', []):
                    st.write(f"• {ans}")
            
            if ok:
                st.success("Correct")
            else:
                st.error("Incorrect")
            
            st.divider()
    
    st.divider()
    
    # AI Report
    st.subheader("🧠 AI-Powered Performance Evaluation")
    
    if st.button("Generate Detailed Report", type="secondary"):
        with st.spinner("Analyzing your technical performance..."):
            try:
                weak_topics = [t for t, qs in topic_scores.items() 
                              if sum(q['marks_earned'] for q in qs) / max(1, sum(q['marks_total'] for q in qs)) < 0.6]
                strengths = [t for t, qs in topic_scores.items() 
                            if sum(q['marks_earned'] for q in qs) / max(1, sum(q['marks_total'] for q in qs)) >= 0.8]
                
                topic_perf_strings = [f"{t}: {round((sum(q['marks_earned'] for q in qs) / max(1, sum(q['marks_total'] for q in qs)) * 100))}%" 
                                     for t, qs in topic_scores.items()]
                
                prompt = f"""
Create a comprehensive technical interview performance report.

Interview Score: {percent}%
Marks: {earned_marks}/{total_marks}
Correct Responses: {num_correct}/{total_q}
Technical Areas Covered: {', '.join(topic_perf_strings)}
Areas Needing Improvement: {weak_topics}
Strong Areas: {strengths}

Generate a professional assessment including:
- Overall Technical Performance (2-3 key points)
- Technical Strengths Demonstrated
- Areas Requiring Focus & Improvement
- 5 Recommended Practice Tasks/Projects
- 7-Day Focused Learning Plan
- 3 Learning Resources (courses/documentation/books)

Keep it professional, actionable, and interview-focused.
"""
                
                stream_gen = ask_llm_stream(prompt)
                report_box = st.empty()
                
                if stream_gen is not None:
                    buf = ""
                    for token in stream_gen:
                        buf += token
                        report_box.markdown(buf)
                else:
                    report_text = ask_llm(prompt)
                    report_box.markdown(report_text)
                    
            except Exception as e:
                st.warning("Could not generate AI report")
    
    st.divider()
    
    # Action buttons
    col_actions = st.columns(2)
    with col_actions[0]:
        if st.button("🔄 Start New Interview", use_container_width=True):
            reset_quiz_state()
            st.rerun()
    with col_actions[1]:
        if st.button("📥 Download Assessment Report", use_container_width=True):
            # Create downloadable JSON
            results_data = {
                "score_percent": percent,
                "marks_earned": earned_marks,
                "marks_total": total_marks,
                "correct_answers": num_correct,
                "total_questions": total_q,
                "topic_performance": {
                    t: {
                        "score": sum(q['marks_earned'] for q in qs),
                        "total": sum(q['marks_total'] for q in qs),
                        "percentage": round((sum(q['marks_earned'] for q in qs) / sum(q['marks_total'] for q in qs)) * 100, 1)
                    } for t, qs in topic_scores.items()
                },
                "answers": answers
            }
            
            import io
            import json as json_module
            
            json_str = json_module.dumps(results_data, indent=2)
            st.download_button(
                label="Download Assessment JSON",
                data=json_str,
                file_name=f"interview_assessment_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
