from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# 1. Locate official academic documents
# --------------------------------------------------

academic_folder = Path("documents/ACADEMICS")

pdf_files = list(academic_folder.glob("*.pdf"))

print("PDFs found:", len(pdf_files))


# --------------------------------------------------
# 2. Load all academic PDFs
# --------------------------------------------------

all_documents = []

for pdf_file in pdf_files:

    print(f"Loading: {pdf_file.name}")

    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()

    # Store the PDF name as metadata
    for document in documents:
        document.metadata["source"] = pdf_file.name

    all_documents.extend(documents)

    print(f"Pages loaded: {len(documents)}")


print("\nTotal pages loaded:", len(all_documents))


# --------------------------------------------------
# 3. Split documents into smaller chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(all_documents)

print("Total chunks created:", len(chunks))


# --------------------------------------------------
# 4. Create embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# --------------------------------------------------
# 5. Create FAISS vector database
# --------------------------------------------------

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector database created.")


# --------------------------------------------------
# 6. Save FAISS database
# --------------------------------------------------

vectorstore.save_local("vectorstore")

print("FAISS vector database saved successfully.")