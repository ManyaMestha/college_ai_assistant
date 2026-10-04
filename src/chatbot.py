import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load environment variables (GOOGLE_API_KEY)
load_dotenv()

# Initialize LLM with a valid model name
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

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

# Import retrieval function directly from rag.py inside src/
from rag import retrieve_documents


def ask_question(question: str) -> str:
    docs = retrieve_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    formatted_prompt = prompt_template.format(
        context=context,
        question=question
    )

    response = llm.invoke(formatted_prompt)
    return response.content


if __name__ == "__main__":
    test_question = "What is the attendance requirement?"
    answer = ask_question(test_question)
    print("Answer:", answer)