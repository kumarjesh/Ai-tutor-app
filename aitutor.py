import os
import re
import sys
from typing import TypedDict, List
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, END

# ==================================================
# 🔑 CONFIG
# ==================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file")

MODEL_NAME = "llama-3.1-8b-instant"

client = Groq(api_key=GROQ_API_KEY)


# ==================================================
# 🤖 SAFE INPUT (exit anywhere)
# ==================================================
def safe_input(prompt_text: str) -> str:
    """
    If user types exit anywhere -> quit program
    """
    user_text = input(prompt_text).strip()

    if user_text.lower() == "exit":
        print("\n👋 Exiting Tutor. Goodbye!")
        sys.exit()

    return user_text


# ==================================================
# 🤖 LLM CALL
# ==================================================
def call_llm(prompt: str, temperature: float = 0.4) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert AI tutor."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ API Error: {str(e)}"


# ==================================================
# 🧠 STATE
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


# ==================================================
# 🧹 HELPERS
# ==================================================
def parse_numbered_list(text: str) -> List[str]:
    items = []

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        line = line.replace("**", "")

        if len(line) > 2:
            items.append(line)

    return items


def show_topics(topics: List[str]):
    print("\n📚 Topics:\n")

    for i, topic in enumerate(topics, start=1):
        print(f"{i}. {topic}")


# ==================================================
# 🤖 AGENTS
# ==================================================
def planner_agent(state: TutorState):
    prompt = f"""
Break syllabus into numbered topics.

Rules:
- Only numbered topics
- No headings
- No markdown
- Max 10 topics

Syllabus:
{state['syllabus']}
"""

    topics = call_llm(prompt)

    return {
        "topics": topics,
        "topic_list": parse_numbered_list(topics)
    }


def teaching_agent(state: TutorState):
    prompt = f"""
Teach this topic clearly for beginner.

Topic:
{state['selected_topic']}

Include:
1. Explanation
2. Example
3. Summary
"""

    return {"explanation": call_llm(prompt)}


def quiz_agent(state: TutorState):
    quiz_prompt = f"""
Create 3 MCQs.

Topic:
{state['selected_topic']}

Rules:
- Do NOT reveal answers

Format:

Q1:
Question
A)
B)
C)
D)

Q2:
...

Q3:
...
"""

    key_prompt = f"""
Create answer key only.

Topic:
{state['selected_topic']}

Format:
Q1: A
Q2: C
Q3: B
"""

    return {
        "quiz": call_llm(quiz_prompt),
        "answer_key": call_llm(key_prompt)
    }


def evaluation_agent(state: TutorState):
    prompt = f"""
Evaluate answers.

Correct Key:
{state['answer_key']}

Student Answers:
{state['user_answer']}

Return:
1. Score /3
2. Correct Answers
3. Mistakes
4. Suggestions
"""

    return {"evaluation": call_llm(prompt)}


def deep_dive_agent(state: TutorState):
    prompt = f"""
Deep dive into topic:

{state['selected_topic']}

Include:
1. Detailed explanation
2. Advanced concepts
3. Real-world examples
4. Common mistakes
"""

    return call_llm(prompt)


# ==================================================
# 🔁 GRAPH
# ==================================================
def build_graph():
    graph = StateGraph(TutorState)

    graph.add_node("plan", planner_agent)
    graph.add_node("teach", teaching_agent)
    graph.add_node("quiz", quiz_agent)
    graph.add_node("evaluate", evaluation_agent)

    graph.set_entry_point("plan")

    graph.add_edge("plan", "teach")
    graph.add_edge("teach", "quiz")
    graph.add_edge("quiz", "evaluate")
    graph.add_edge("evaluate", END)

    return graph.compile()


# ==================================================
# 🚀 MAIN
# ==================================================
def main():
    print("=" * 65)
    print("🎓 AI Syllabus Tutor")
    print("Type 'exit' anytime to quit.")
    print("=" * 65)

    app = build_graph()

    # ==================================================
    # NEW SYLLABUS LOOP
    # ==================================================
    while True:

        syllabus = safe_input("\n📘 Enter syllabus:\n> ")

        state: TutorState = {
            "syllabus": syllabus,
            "selected_topic": "",
            "user_answer": ""
        }

        result = app.invoke(state)
        state.update(result)

        if not state["topic_list"]:
            print("❌ Could not generate topics")
            continue

        # ==================================================
        # TOPIC LOOP
        # ==================================================
        while True:

            show_topics(state["topic_list"])

            try:
                choice = int(
                    safe_input("\n👉 Select topic number:\n> ")
                )

                if choice < 1 or choice > len(state["topic_list"]):
                    print("❌ Invalid topic")
                    continue

            except ValueError:
                print("❌ Enter valid number")
                continue

            state["selected_topic"] = state["topic_list"][choice - 1]

            # ==================================================
            # LEARNING LOOP
            # ==================================================
            while True:

                print(f"\n📖 Topic: {state['selected_topic']}")

                result = app.invoke(state)
                state.update(result)

                print("\n🧠 Explanation:\n")
                print(state["explanation"])

                print("\n📝 Quiz:\n")
                print(state["quiz"])

                answers = safe_input(
                    "\n✍️ Enter answers (Q1:A, Q2:B, Q3:C):\n> "
                )

                state["user_answer"] = answers

                eval_result = evaluation_agent(state)
                state.update(eval_result)

                print("\n📊 Evaluation:\n")
                print(state["evaluation"])

                # ==========================================
                # MENU
                # ==========================================
                print("\n🔽 Options:")
                print("1. Deep dive into this topic")
                print("2. Continue discussion entered syllabus")
                print("3. Choose new topic, ask for new syllabus")
                print("4. Exit")

                menu = safe_input("\n👉 Enter choice:\n> ")

                # --------------------------------------
                # 1 Deep Dive
                # --------------------------------------
                if menu == "1":
                    print("\n🚀 Deep Dive:\n")
                    print(deep_dive_agent(state))

                # --------------------------------------
                # 2 Choose another topic
                # --------------------------------------
                elif menu == "2":

                    print("\n📚 Choose another topic:\n")

                    for i, topic in enumerate(
                        state["topic_list"], start=1
                    ):
                        print(f"{i}. {topic}")

                    try:
                        t_choice = int(
                            safe_input(
                                "\n👉 Enter topic number:\n> "
                            )
                        )

                        if (
                            t_choice < 1 or
                            t_choice > len(state["topic_list"])
                        ):
                            print("❌ Invalid topic")
                            continue

                        state["selected_topic"] = \
                            state["topic_list"][t_choice - 1]

                        print(
                            f"\n📌 Switched to: "
                            f"{state['selected_topic']}"
                        )

                        continue

                    except ValueError:
                        print("❌ Enter valid number")
                        continue

                # --------------------------------------
                # 3 New syllabus
                # --------------------------------------
                elif menu == "3":
                    print("\n🔄 Starting new syllabus...")
                    break

                # --------------------------------------
                # 4 Exit
                # --------------------------------------
                elif menu == "4":
                    print("\n👋 Goodbye!")
                    return

                else:
                    print("❌ Invalid choice")

            if menu == "3":
                break


# ==================================================
# ▶️ RUN
# ==================================================
if __name__ == "__main__":
    main()