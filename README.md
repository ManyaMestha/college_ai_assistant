# AI-Based College Academic Assistant

## Project Overview

The AI-Based College Academic Assistant is a college-focused application designed to help students access official academic information and study resources through a single platform.

The project combines Retrieval-Augmented Generation (RAG), an LLM-based chatbot, a study planner, and a study-material resource system.

The system consists of four major components:

1. Academic Document RAG
2. AI Chatbot
3. Study Planner
4. Study Material Resource Access

The Academic Document RAG component processes official college academic documents and retrieves relevant information for student queries.

The AI Chatbot uses the retrieved academic information to provide answers to student questions.

The Study Planner is designed to help students create and modify study plans.

The Study Material Resource Access component allows students to browse and download available study materials such as Notes, MCQs, Question Banks, and Previous Year Question Papers (PYQs).

Study materials are treated only as downloadable resources. They are not processed through the RAG pipeline and are not used for embeddings or chatbot responses.

---

## Project Architecture

```text
                    AI-Based College Academic Assistant
                                  |
          +-----------------------+-----------------------+
          |                       |                       |
   Academic Document          AI Chatbot            Study Planner
          RAG                     |                       |
          |                       |                    LangGraph
   Official Academic              |                       |
          PDFs                    |                       |
          |                       |                       |
     PDF Loading                LLM                       |
          |                       |                       |
    Text Chunking                 |                       |
          |                       |                       |
     Embeddings                   |                       |
          |                       |                       |
        FAISS                    |                       |
          |                       |                       |
      Retrieval -----------------+                       |
          |                                               |
          +-----------------------------------------------+
                              |
                       Student Query
                              |
                 Relevant Academic Information


                    Study Material Resources
                              |
                       Subject Folders
                              |
          +-------------------+-------------------+
          |                   |                   |
        NOTES                MCQ          QUESTION_BANK
          |                   |                   |
          +-------------------+-------------------+
                              |
                             PYQ
                              |
                       PDF Listing
                              |
                       Open / Download