from datetime import date, datetime
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from src.llm import get_llm


# --------------------------------------------------
# 1. LangGraph state
# --------------------------------------------------

class StudyState(TypedDict, total=False):
    subjects: list[str]
    exam_date: str
    hours_per_day: float
    weak_subject: str
    days_left: int
    study_plan: str
    modification_request: str


# --------------------------------------------------
# 2. Prompts
# --------------------------------------------------

GENERATE_PROMPT = ChatPromptTemplate.from_template(
    """You are a study planner for college students.

Create a day-by-day study plan using these details:

Today's date: {today}
Exam date: {exam_date}
Days available before the exam: {days_left}
Subjects: {subjects}
Study hours available per day: {hours_per_day}
Weak subject(s): {weak_subject}

Rules:
- Day 1 is today. Label each day with its date,
  e.g. "Day 1 (Mon, 06 Oct)".
- Each day's total hours must not exceed {hours_per_day}.
- Give the weak subject(s) noticeably more time.
- Cover every subject at least once.
- Keep the last day for revision of all subjects.
- If there are more than 14 days, group the plan by week
  and list the daily split once per week.
- Show each day as a short list: subject - hours - what to focus on.
- End with 2-3 short study tips.

Return only the study plan."""
)

MODIFY_PROMPT = ChatPromptTemplate.from_template(
    """You are a study planner for college students.

Today's date: {today}

Here is the student's current study plan:

{study_plan}

The student wants this change:
"{modification_request}"

Update the plan to apply the change.

Rules:
- Keep the same format as the current plan.
- Keep the first line (exam date and study hours) unchanged.
- Never schedule study on or after the exam date. All work must
  fit in the days before it. If there is not enough time,
  combine or shorten topics instead of adding days.
- Change only what the request requires.
- If time is removed from a day, do NOT move all of that work
  to the next day. Spread the missed topics across several of
  the following days, one or two extra topics per day at most,
  so every subject is still covered before the exam.
- Do not exceed the student's daily study hours unless they ask.
- Show a day with no study as one line, e.g.
  "Day 2 (Mon, 05 Oct) - Rest day: no study".
- Update any weekly summary or "Daily Split" lines so they
  match the changed days.

Return only the updated study plan."""
)


# --------------------------------------------------
# 3. Helper: parse the exam date
# --------------------------------------------------

def parse_exam_date(exam_date):
    """
    Accept a date object, "YYYY-MM-DD" or "DD/MM/YYYY"
    and return a date object.
    """

    if isinstance(exam_date, date):
        return exam_date

    for date_format in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]:

        try:
            return datetime.strptime(exam_date.strip(), date_format).date()

        except ValueError:
            continue

    raise ValueError(
        "Exam date must be in YYYY-MM-DD or DD/MM/YYYY format."
    )


# --------------------------------------------------
# 4. Node: collect and validate student details
# --------------------------------------------------

def collect_details(state):
    """
    Validate the student's input before calling the LLM.
    For a new plan, also calculate the days left.
    """

    # Modifying an existing plan only needs the plan and the request
    if state.get("modification_request") is not None:

        if not state.get("study_plan", "").strip():
            raise ValueError("There is no study plan to modify.")

        if not state["modification_request"].strip():
            raise ValueError("Please describe the change you want.")

        return {}

    subjects = [
        subject.strip()
        for subject in state.get("subjects", [])
        if subject.strip()
    ]

    if not subjects:
        raise ValueError("Please enter at least one subject.")

    if state.get("hours_per_day", 0) <= 0:
        raise ValueError("Study hours per day must be more than 0.")

    if state["hours_per_day"] > 16:
        raise ValueError("Study hours per day cannot be more than 16.")

    exam_date = parse_exam_date(state.get("exam_date", ""))

    days_left = (exam_date - date.today()).days

    if days_left <= 0:
        raise ValueError("The exam date must be in the future.")

    return {
        "subjects": subjects,
        "exam_date": exam_date.isoformat(),
        "days_left": days_left
    }


# --------------------------------------------------
# 5. Node: generate a new study plan
# --------------------------------------------------

def generate_plan(state):

    prompt = GENERATE_PROMPT.format_messages(
        today=date.today().strftime("%A, %d %B %Y"),
        exam_date=state["exam_date"],
        days_left=state["days_left"],
        subjects=", ".join(state["subjects"]),
        hours_per_day=state["hours_per_day"],
        weak_subject=state.get("weak_subject") or "None"
    )

    response = get_llm().invoke(prompt)

    # Fixed header so the modify step knows the exam date and daily hours
    exam_day = parse_exam_date(state["exam_date"]).strftime("%a, %d %b %Y")

    header = (
        f"**Exam date:** {exam_day} | "
        f"**Study hours per day:** {state['hours_per_day']:g}\n\n"
    )

    return {
        "study_plan": header + response.text
    }


# --------------------------------------------------
# 6. Node: modify an existing study plan
# --------------------------------------------------

def modify_plan(state):

    prompt = MODIFY_PROMPT.format_messages(
        today=date.today().strftime("%A, %d %B %Y"),
        study_plan=state["study_plan"],
        modification_request=state["modification_request"]
    )

    response = get_llm().invoke(prompt)

    return {
        "study_plan": response.text
    }


# --------------------------------------------------
# 7. Route: new plan or modification?
# --------------------------------------------------

def choose_next_step(state):

    if state.get("modification_request") is not None:
        return "modify_plan"

    return "generate_plan"


# --------------------------------------------------
# 8. Build the LangGraph workflow
# --------------------------------------------------
#
#   START → collect_details ─┬→ generate_plan → END
#                            └→ modify_plan   → END

graph = StateGraph(StudyState)

graph.add_node("collect_details", collect_details)
graph.add_node("generate_plan", generate_plan)
graph.add_node("modify_plan", modify_plan)

graph.add_edge(START, "collect_details")

graph.add_conditional_edges(
    "collect_details",
    choose_next_step,
    ["generate_plan", "modify_plan"]
)

graph.add_edge("generate_plan", END)
graph.add_edge("modify_plan", END)

study_planner = graph.compile()


# --------------------------------------------------
# 9. Functions for the UI (Person 4)
# --------------------------------------------------

def create_study_plan(subjects, exam_date, hours_per_day, weak_subject=""):
    """
    Create a new personalized study plan.

    Args:
        subjects: List of subjects, or a comma-separated string
                  such as "DSA, DBMS, Maths".
        exam_date: date object, "YYYY-MM-DD" or "DD/MM/YYYY".
        hours_per_day: Study hours available each day.
        weak_subject: Subject(s) that need extra time (optional).

    Returns:
        The study plan as text (Markdown).

    Raises:
        ValueError: If the input is invalid. The message can be
                    shown directly to the student.
    """

    if isinstance(subjects, str):
        subjects = subjects.split(",")

    result = study_planner.invoke({
        "subjects": subjects,
        "exam_date": exam_date,
        "hours_per_day": hours_per_day,
        "weak_subject": weak_subject
    })

    return result["study_plan"]


def modify_study_plan(study_plan, modification_request):
    """
    Modify an existing study plan.

    Args:
        study_plan: The plan returned by create_study_plan.
        modification_request: The student's change, e.g.
                              "I only have 2 hours tomorrow."

    Returns:
        The updated study plan as text (Markdown).

    Raises:
        ValueError: If the plan or the request is empty.
    """

    result = study_planner.invoke({
        "study_plan": study_plan,
        "modification_request": modification_request
    })

    return result["study_plan"]


# --------------------------------------------------
# 10. Test the study planner
# --------------------------------------------------

if __name__ == "__main__":

    print("\nStudy Planner")
    print("Run from the project root: python -m src.study_planner\n")

    subjects = input("Subjects (comma-separated): ")
    exam_date = input("Exam date (YYYY-MM-DD or DD/MM/YYYY): ")
    hours_per_day = float(input("Study hours per day: "))
    weak_subject = input("Weak subject (optional): ")

    try:
        plan = create_study_plan(
            subjects,
            exam_date,
            hours_per_day,
            weak_subject
        )

    except Exception as error:
        print(f"\nCouldn't create the study plan: {error}\n")
        raise SystemExit(1)

    print("\n" + plan + "\n")

    print("Type a change to the plan, or 'exit', 'done', or 'quit' to stop.\n")

    while True:

        request = input("Change: ").strip()

        if request.lower() in ["exit", "done", "quit"]:
            print("\nExiting Study Planner...")
            break

        if not request:
            continue

        try:
            plan = modify_study_plan(plan, request)
            print("\n" + plan + "\n")

        except Exception as error:
            print(f"\nCouldn't update the plan: {error}")
            print("Your previous plan is unchanged. Please try again.\n")
