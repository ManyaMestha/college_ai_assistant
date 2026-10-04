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
# 3. Retrieve relevant academic documents
# --------------------------------------------------

def retrieve_documents(question, k=5):
    """
    Retrieve relevant chunks from official
    college academic documents.

    Args:
        question: Student's academic question.
        k: Number of relevant chunks to retrieve.

    Returns:
        List of relevant document chunks.
        Returns an empty list if the question
        is not related to college information.
    """

    # Search the FAISS vector database
    results = vectorstore.similarity_search_with_score(
        question,
        k=k
    )

    # --------------------------------------------------
    # 4. Relevance threshold
    # --------------------------------------------------

    # Lower score = better similarity
    RELEVANCE_THRESHOLD = 1.0

    relevant_documents = []

    for doc, score in results:

        print(f"Retrieved score: {score:.4f}")

        if score <= RELEVANCE_THRESHOLD:
            relevant_documents.append(doc)

    return relevant_documents


# --------------------------------------------------
# 5. Test the RAG retrieval system
# --------------------------------------------------

if __name__ == "__main__":

    print("\nCollege Academic Assistant")
    print("Ask questions related to college academic information.")
    print("Type 'exit', 'done', or 'quit' to stop.\n")

    while True:

        question = input("Enter your question: ").strip()

        # --------------------------------------------------
        # 6. Exit commands
        # --------------------------------------------------

        if question.lower() in ["exit", "done", "quit"]:
            print("\nExiting College Academic Assistant...")
            break

        # --------------------------------------------------
        # 7. Handle empty questions
        # --------------------------------------------------

        if not question:
            print("\nPlease enter a question.\n")
            continue

        # --------------------------------------------------
        # 8. Retrieve relevant documents
        # --------------------------------------------------

        results = retrieve_documents(question)

        # --------------------------------------------------
        # 9. Handle unrelated questions
        # --------------------------------------------------

        if not results:

            print(
                "\nI'm sorry, I can only answer questions "
                "related to the college's official academic information.\n"
            )

            continue

        # --------------------------------------------------
        # 10. Display retrieved information
        # --------------------------------------------------

        print("\nRelevant academic information found:")

        for i, doc in enumerate(results, start=1):

            print(f"\n--- Result {i} ---")

            print(doc.page_content[:700])

            print("\nSource:")

            print(
                doc.metadata.get(
                    "source",
                    "Unknown"
                )
            )

        print()