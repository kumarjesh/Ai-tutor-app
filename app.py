import os
import re
import streamlit as st
from typing import TypedDict, List
from dotenv import load_dotenv
from groq import Groq

# ==================================================
# 🔑 CONFIG & SETUP
# ==================================================
st.set_page_config(page_title="AI Syllabus Tutor", page_icon="🎓", layout="centered")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file. Please set it to proceed.")
    st.stop()

MODEL_NAME = "llama-3.1-8b-instant"
client = Groq(api_key=GROQ_API_KEY)

# ==================================================
# 🧠 STATE MANAGEMENT
# ==================================================
class TutorState(TypedDict, total=False):
    syllabus: str
    topics: str
    topic_list: List[str]
    selected_topic: str
    explanation: str
    quiz: str
    answer_key: str
    user_answer: str
    evaluation: str
    deep_dive: str

# Initialize session state variables
if "tutor_state" not in st.session_state:
    st.session_state.tutor_state = TutorState(
        syllabus="", topics="", topic_list=[], selected_topic="",
        explanation="", quiz="", answer_key="", user_answer="", evaluation="", deep_dive=""
    )
if "app_step" not in st.session_state:
    st.session_state.app_step = "plan" # 'plan', 'select_topic', 'learn', 'evaluate'

# ==================================================
# 🤖 LLM CALL
# ==================================================
def call_llm(prompt: str, temperature: float = 0.4) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "You are an expert AI tutor."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ API Error: {str(e)}"

# ==================================================
# 🧹 HELPERS
# ==================================================
def parse_numbered_list(text: str) -> List[str]:
    items = []
    for line in text.split("\n"):
        line = line.strip()
        if not line: continue
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        line = line.replace("**", "")
        if len(line) > 2: items.append(line)
    return items

def reset_topic_state():
    st.session_state.tutor_state['explanation'] = ""
    st.session_state.tutor_state['quiz'] = ""
    st.session_state.tutor_state['answer_key'] = ""
    st.session_state.tutor_state['user_answer'] = ""
    st.session_state.tutor_state['evaluation'] = ""
    st.session_state.tutor_state['deep_dive'] = ""
    st.session_state.app_step = "learn"

# ==================================================
# 🤖 AGENTS
# ==================================================
def planner_agent(syllabus: str):
    prompt = f"Break syllabus into numbered topics.\nRules:\n- Only numbered topics\n- No headings\n- No markdown\n- Max 10 topics\nSyllabus:\n{syllabus}"
    topics = call_llm(prompt)
    return topics, parse_numbered_list(topics)

def teaching_agent(topic: str):
    prompt = f"Teach this topic clearly for beginner.\nTopic:\n{topic}\nInclude:\n1. Explanation\n2. Example\n3. Summary"
    return call_llm(prompt)

def quiz_agent(topic: str):
    quiz_prompt = f"Create 3 MCQs.\nTopic:\n{topic}\nRules:\n- Do NOT reveal answers\nFormat:\nQ1:\nQuestion\nA)\nB)\nC)\nD)\n..."
    key_prompt = f"Create answer key only.\nTopic:\n{topic}\nFormat:\nQ1: A\nQ2: C\nQ3: B"
    return call_llm(quiz_prompt), call_llm(key_prompt)

def evaluation_agent(answer_key: str, user_answer: str):
    prompt = f"Evaluate answers.\nCorrect Key:\n{answer_key}\nStudent Answers:\n{user_answer}\nReturn:\n1. Score /3\n2. Correct Answers\n3. Mistakes\n4. Suggestions"
    return call_llm(prompt)

def deep_dive_agent(topic: str):
    prompt = f"Deep dive into topic:\n{topic}\nInclude:\n1. Detailed explanation\n2. Advanced concepts\n3. Real-world examples\n4. Common mistakes"
    return call_llm(prompt)


# ==================================================
# 🚀 MAIN UI
# ==================================================
st.title("🎓 AI Syllabus Tutor")
st.markdown("---")

state = st.session_state.tutor_state

# --------------------------------------------------
# STEP 1: Syllabus Input
# --------------------------------------------------
st.header("1. Upload Syllabus")
syllabus_input = st.text_area("Paste your syllabus here:", value=state['syllabus'], height=150)

if st.button("Generate Topics"):
    if syllabus_input.strip():
        with st.spinner("Analyzing syllabus..."):
            state['syllabus'] = syllabus_input
            topics_text, topic_list = planner_agent(syllabus_input)
            state['topics'] = topics_text
            state['topic_list'] = topic_list
            st.session_state.app_step = "select_topic"
            st.rerun()
    else:
        st.warning("Please enter a syllabus first.")

st.markdown("---")

# --------------------------------------------------
# STEP 2: Topic Selection
# --------------------------------------------------
if state['topic_list']:
    st.header("2. Choose a Topic")
    
    selected_topic = st.radio(
        "Select a topic to learn:", 
        options=state['topic_list'],
        index=state['topic_list'].index(state['selected_topic']) if state['selected_topic'] in state['topic_list'] else 0
    )
    
    if st.button("Start Learning"):
        state['selected_topic'] = selected_topic
        reset_topic_state()
        
        with st.spinner("Preparing your lesson & quiz..."):
            state['explanation'] = teaching_agent(selected_topic)
            quiz_text, answer_key = quiz_agent(selected_topic)
            state['quiz'] = quiz_text
            state['answer_key'] = answer_key
            st.rerun()

st.markdown("---")

# --------------------------------------------------
# STEP 3: Learning & Quiz
# --------------------------------------------------
if state['explanation'] and st.session_state.app_step in ["learn", "evaluate"]:
    st.header(f"📖 {state['selected_topic']}")
    
    with st.expander("📚 Read Explanation", expanded=True):
        st.write(state['explanation'])

    st.subheader("📝 Quiz Time!")
    st.info(state['quiz'])
    
    user_answers = st.text_input("Enter your answers (e.g., Q1:A, Q2:B, Q3:C):", value=state['user_answer'])
    
    if st.button("Submit Answers"):
        if user_answers.strip():
            with st.spinner("Grading..."):
                state['user_answer'] = user_answers
                state['evaluation'] = evaluation_agent(state['answer_key'], user_answers)
                st.session_state.app_step = "evaluate"
                st.rerun()
        else:
            st.warning("Please enter your answers before submitting.")

# --------------------------------------------------
# STEP 4: Evaluation & Deep Dive
# --------------------------------------------------
if state['evaluation'] and st.session_state.app_step == "evaluate":
    st.markdown("---")
    st.header("📊 Evaluation Results")
    st.success(state['evaluation'])
    
    # Show Deep Dive if it was previously generated
    if state['deep_dive']:
        st.markdown("### 🚀 Deep Dive")
        st.write(state['deep_dive'])

    st.markdown("---")
    st.subheader("🔽 Options:")
    
    # The Interactive Menu
    next_action = st.radio(
        "What would you like to do next?",
        options=[
            "1. Deep dive into this topic",
            "2. Choose another topic from current syllabus",
            "3. Start a completely new syllabus",
            "4. Exit"
        ],
        label_visibility="collapsed"
    )
    
    if st.button("👉 Proceed"):
        
        # Choice 1: Deep Dive
        if next_action.startswith("1"):
            with st.spinner("Generating advanced content..."):
                state['deep_dive'] = deep_dive_agent(state['selected_topic'])
                st.rerun()
                
        # Choice 2: Back to current topic list
        elif next_action.startswith("2"):
            reset_topic_state() # Clears the quiz/explanation
            st.session_state.app_step = "select_topic"
            st.rerun()
            
        # Choice 3: Hard reset for a new syllabus
        elif next_action.startswith("3"):
            # Wipe the entire state clean
            st.session_state.tutor_state = TutorState(
                syllabus="", topics="", topic_list=[], selected_topic="",
                explanation="", quiz="", answer_key="", user_answer="", evaluation="", deep_dive=""
            )
            st.session_state.app_step = "plan"
            st.rerun()
            
        # Choice 4: Exit App
        elif next_action.startswith("4"):
            # Web apps don't "close" like terminal apps, so we clear the screen and show a message
            st.session_state.app_step = "exit"
            st.rerun()

# --------------------------------------------------
# EXIT SCREEN
# --------------------------------------------------
if st.session_state.app_step == "exit":
    st.empty() # Clears everything above
    st.title("👋 Goodbye!")
    st.info("Thank you for using the AI Syllabus Tutor. You can close this tab now, or refresh the page to start over.")
    st.stop()