from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# 1. Load the embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 2. Load the existing FAISS vector database
# --------------------------------------------------

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)


# --------------------------------------------------
# 3. Create retriever
# --------------------------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# --------------------------------------------------
# 4. Retrieve relevant academic documents
# --------------------------------------------------

def retrieve_documents(question, k=3):
    """
    Retrieve relevant chunks from official
    academic documents.

    Args:
        question: Student's academic question.
        k: Number of chunks to retrieve.

    Returns:
        List of relevant document chunks.
    """

    results = retriever.invoke(question)

    return results[:k]

if __name__ == "__main__":

    question = "What are the attendance requirements?"

    results = retrieve_documents(question)

    print("\nRetrieved results:")

    for i, doc in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content[:500])