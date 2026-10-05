from datetime import date, timedelta
from pathlib import Path

import streamlit as st

from src.resources import get_subjects
from src.resources import get_categories
from src.resources import get_pdfs
from src.study_planner import create_study_plan
from src.study_planner import modify_study_plan


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="College Academic Assistant",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# Styling (colours and fonts are in .streamlit/config.toml)
# --------------------------------------------------

st.markdown(
    f"<style>{Path('assets/style.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">NMAMIT · Academic Assistant</div>
        <h1 class="hero-title">Your college, answered.</h1>
        <p class="hero-subtitle">
            Ask about official college rules, build a study plan that fits
            your schedule, and download notes, MCQs and past papers,
            all in one place.
        </p>
        <div class="hero-chips">
            <span class="hero-chip">💬 Academic Q&amp;A</span>
            <span class="hero-chip">📅 AI Study Planner</span>
            <span class="hero-chip">📚 Study Materials</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Chat avatars
USER_AVATAR = "🧑‍🎓"
ASSISTANT_AVATAR = "🎓"

AVATARS = {
    "user": USER_AVATAR,
    "assistant": ASSISTANT_AVATAR
}


# --------------------------------------------------
# Load the chatbot once
# --------------------------------------------------

# Importing the chatbot loads the embedding model and the FAISS
# database, which takes a few seconds, so it is cached and only
# loaded when the first question is asked.

@st.cache_resource(
    show_spinner="Loading college documents (only on the first question, "
                 "this can take a minute or two)..."
)
def load_chatbot():

    if not Path("vectorstore/index.faiss").exists():
        raise FileNotFoundError(
            "The document database was not found. Run "
            "`python src/build_vectorstore.py` from the project "
            "folder first."
        )

    from src.chatbot import ask_question

    return ask_question


# --------------------------------------------------
# Short error messages for students
# --------------------------------------------------

def friendly_error(error):

    message = str(error)

    if "RESOURCE_EXHAUSTED" in message or "429" in message:
        return (
            "The Gemini usage limit has been reached. "
            "Please try again later."
        )

    if "UNAVAILABLE" in message or "503" in message:
        return (
            "Gemini is busy right now. "
            "Please try again in a few minutes."
        )

    if "API key" in message:
        return message

    # Keep other errors short; the full error is printed in the terminal
    print(f"Error: {message}")

    return message.split("\n")[0][:200]


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "study_plan" not in st.session_state:
    st.session_state.study_plan = ""

if "plan_changes" not in st.session_state:
    st.session_state.plan_changes = []


chat_tab, planner_tab, materials_tab = st.tabs([
    "💬 Ask a Question",
    "📅 Study Planner",
    "📚 Study Materials"
])


# --------------------------------------------------
# 1. Chatbot tab
# --------------------------------------------------

with chat_tab:

    st.subheader("Ask about college academic information")

    st.markdown(
        '<p class="section-note">Answers come from the official college '
        "documents: academic regulations, examination guidelines, "
        "student mentoring and syllabus.</p>",
        unsafe_allow_html=True
    )

    # Conversation area, kept above the question box
    conversation = st.container()

    with conversation:

        # Example questions, shown until the first question is asked
        hint = st.empty()

        if not st.session_state.messages:
            hint.info(
                "**Try asking:** What is the minimum attendance required? · "
                "How is the SEE evaluated? · How many activity points "
                "are needed for the degree?",
                icon="✨"
            )

        for message in st.session_state.messages:

            with st.chat_message(
                message["role"],
                avatar=AVATARS[message["role"]]
            ):
                st.markdown(message["content"])

    question = st.chat_input("e.g. What is the minimum attendance required?")

    if question:

        hint.empty()

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with conversation, st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(question)

        with conversation, st.chat_message("assistant", avatar=ASSISTANT_AVATAR):

            try:
                ask_question = load_chatbot()

                with st.spinner("Searching the college documents..."):
                    answer = ask_question(question)

            except Exception as error:
                answer = f"⚠️ {friendly_error(error)}"

            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    if st.session_state.messages:

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()


# --------------------------------------------------
# 2. Study planner tab
# --------------------------------------------------

with planner_tab:

    st.subheader("Create your study plan")

    st.markdown(
        '<p class="section-note">Tell us your subjects and exam date. '
        "The plan gives extra time to your weak subject and keeps the "
        "last day for revision.</p>",
        unsafe_allow_html=True
    )

    with st.form("study_plan_form"):

        subjects = st.text_input(
            "Subjects (comma-separated)",
            placeholder="e.g. Maths, Physics, PSP"
        )

        column_1, column_2 = st.columns(2)

        with column_1:
            exam_date = st.date_input(
                "Exam date",
                value=date.today() + timedelta(days=14),
                min_value=date.today() + timedelta(days=1)
            )

        with column_2:
            hours_per_day = st.number_input(
                "Study hours per day",
                min_value=0.5,
                max_value=16.0,
                value=3.0,
                step=0.5
            )

        weak_subject = st.text_input(
            "Weak subject (optional)",
            placeholder="e.g. Maths"
        )

        generate = st.form_submit_button("Generate Plan", type="primary")

    if generate:

        try:
            with st.spinner("Creating your study plan..."):
                st.session_state.study_plan = create_study_plan(
                    subjects,
                    exam_date,
                    hours_per_day,
                    weak_subject
                )

            # A new plan has no changes yet
            st.session_state.plan_changes = []

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error(f"Couldn't create the study plan. {friendly_error(error)}")

    # Show the modify option and the plan once a plan exists
    if st.session_state.study_plan:

        st.divider()

        # Modify box sits above the plan, so the result
        # appears right below it
        st.subheader("Modify your plan")

        with st.form("modify_plan_form", clear_on_submit=True):

            modification_request = st.text_input(
                "What would you like to change?",
                placeholder="e.g. I can't study tomorrow"
            )

            modify = st.form_submit_button("Modify Plan")

        if modify:

            try:
                with st.spinner("Updating your study plan..."):
                    st.session_state.study_plan = modify_study_plan(
                        st.session_state.study_plan,
                        modification_request
                    )

                st.session_state.plan_changes.append(
                    modification_request.strip()
                )

            except ValueError as error:
                st.error(str(error))

            except Exception as error:
                st.error(
                    f"Couldn't update the plan. {friendly_error(error)} "
                    "Your previous plan is unchanged."
                )

        st.divider()

        st.subheader("Your study plan")

        # Keep showing which changes have been applied
        if st.session_state.plan_changes:

            changes = "\n".join(
                f"- {change}"
                for change in st.session_state.plan_changes
            )

            st.success(f"✅ Plan updated with your changes:\n{changes}")

        with st.container(key="plan_card"):
            st.markdown(st.session_state.study_plan)

        st.download_button(
            label="⬇ Download plan",
            data=st.session_state.study_plan,
            file_name="study_plan.md",
            mime="text/markdown"
        )


# --------------------------------------------------
# 3. Study materials tab
# --------------------------------------------------

with materials_tab:

    st.subheader("Study materials")

    st.markdown(
        '<p class="section-note">Pick a subject and material type to '
        "download notes, MCQs, question banks and previous year papers.</p>",
        unsafe_allow_html=True
    )

    # Readable names for the folder names (display only)
    CATEGORY_NAMES = {
        "NOTES": "📝 Notes",
        "MCQ": "✅ MCQs",
        "QUESTION_BANK": "❓ Question Bank",
        "PYQ": "🗂️ Previous Year Papers (PYQ)"
    }

    column_1, column_2 = st.columns(2)

    with column_1:
        selected_subject = st.selectbox(
            "Subject",
            get_subjects(),
            format_func=lambda name: name.replace("_", " ")
        )

    with column_2:
        selected_category = st.selectbox(
            "Material type",
            get_categories(selected_subject),
            format_func=lambda name: CATEGORY_NAMES.get(
                name, name.replace("_", " ").title()
            )
        )

    pdfs = get_pdfs(selected_subject, selected_category)

    if not pdfs:

        st.info(
            "No PDF files are currently available "
            "in this category."
        )

    else:

        st.caption(
            f"{len(pdfs)} file{'s' if len(pdfs) != 1 else ''} available"
        )

        with st.container(key="materials_list"):

            for pdf in pdfs:

                # The file is read only when its button is clicked
                st.download_button(
                    label=pdf["name"],
                    data=lambda path=pdf["path"]: Path(path).read_bytes(),
                    file_name=pdf["name"],
                    mime="application/pdf",
                    key=pdf["path"],
                    icon=":material/picture_as_pdf:",
                    width="stretch"
                )


st.markdown(
    '<p class="footer-note">Answers are based on official NMAMIT documents. '
    "Always confirm important dates and rules with your department.</p>",
    unsafe_allow_html=True
)
