from langchain_core.prompts import PromptTemplate

# Shared Gemini setup (reads GOOGLE_API_KEY from .env)
from src.llm import get_llm

# Retrieval function from rag.py inside src/
from src.rag import retrieve_documents

NOT_FOUND_MESSAGE = "I couldn't find this information in the college documents."

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a college academic assistant.

Answer the student's question using the provided college documents.

If the answer is not available in the documents, say:
"I couldn't find this information in the college documents."

Context:
{context}

Question:
{question}"""
)

def ask_question(question: str) -> str:
    docs = retrieve_documents(question)

    # No relevant documents: skip the LLM call
    if not docs:
        return NOT_FOUND_MESSAGE

    context = "\n\n".join([doc.page_content for doc in docs])

    formatted_prompt = prompt_template.format(
        context=context,
        question=question
    )

    try:
        response = get_llm().invoke(formatted_prompt)
        return response.text

    except Exception as error:
        # Gemini unavailable (e.g. daily limit reached):
        # still show what was found in the documents
        print(f"Gemini error: {error}")
        return format_passages(docs)


def format_passages(docs, limit=3):
    """
    Show the retrieved document passages directly,
    used when Gemini cannot generate an answer.
    """

    passages = [
        "⚠️ The AI answer is not available right now (Gemini limit "
        "reached or busy). Here are the most relevant passages "
        "from the college documents:"
    ]

    for doc in docs[:limit]:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        page_text = f", page {page + 1}" if isinstance(page, int) else ""
        text = " ".join(doc.page_content.split())

        passages.append(f"**{source}{page_text}**\n\n> {text}")

    return "\n\n".join(passages)


# Run from the project root: python -m src.chatbot
if __name__ == "__main__":
    print("\nCollege Academic Assistant Chatbot")
    print("Type 'exit', 'done', or 'quit' to stop.\n")

    while True:
        question = input("Enter your question: ").strip()

        if question.lower() in ["exit", "done", "quit"]:
            print("\nExiting chatbot...")
            break

        if not question:
            print("\nPlease enter a question.\n")
            continue

        try:
            answer = ask_question(question)
            print("\nAnswer:", answer, "\n")

        except Exception as error:
            print(f"\nCouldn't get an answer right now: {error}")
            print("Please try again.\n")