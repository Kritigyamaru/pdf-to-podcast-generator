# PDF RAG and Podcast Generator

## **1. Objective**

The objective of this project is to build a **Streamlit-based application** that:

- Enables **Retrieval-Augmented Generation (RAG)** on PDF documents.
- Allows users to **ask questions** about PDF content in **Nepali or English**.
- Converts PDF content into a **friendly conversational podcast** with distinct **Host and Guest voices**.
- Works fully **offline** using locally hosted LLMs.

This project leverages **LLaMA3 via Ollama** for language understanding and **Kokoro TTS** for speech synthesis.

---

## **2. Project Overview**

This application provides two main functionalities:

1. **PDF RAG (Question Answering)**
2. **PDF to Conversational Podcast Generation**

Key characteristics of the project:

- Built using **Streamlit** for an interactive UI.
- Uses **ChromaDB** for vector storage and retrieval.
- Supports **multilingual question answering**.
- Generates **audio podcasts** from document content.

---

## **3. System Components**

The system is organized into modular Python files for better maintainability.

### Project Structure

```text
project/
│
├── app.py                  # Main Streamlit entry point
├── modules/  
│   ├── config.py            # App configuration & constants
│   ├── helpers.py           # Utility/helper functions
│   ├── pdf_rag.py           # PDF RAG logic (ingestion, retrieval, QA)
│   └── pdf_podcast.py       # PDF to podcast (script + audio generation)
│
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```
 ## **4. Functional Modules**

The project is divided into well-structured functional modules to ensure modularity, readability, and ease of maintenance.

### **4.1 PDF RAG Module (`pdf_rag.py`)**

This module implements **Retrieval-Augmented Generation (RAG)** over PDF documents. Its responsibilities include:

- Uploading one or more PDF documents.
- Extracting textual content from PDFs.
- Splitting text into manageable chunks.
- Generating embeddings using the **bge-m3** model.
- Storing embeddings in a **Chroma vector database**.
- Retrieving the most relevant chunks based on user queries.
- Generating accurate answers using **LLaMA3**.
- Displaying source references for transparency.

This module enables users to ask questions in **Nepali or English** and receive contextual answers from the uploaded PDFs.

---

### **4.2 PDF to Podcast Module (`pdf_podcast.py`)**

This module converts PDF content into a **conversational podcast format**. The key functionalities include:

- Reading and extracting text from PDF files.
- Generating a friendly, host–guest style podcast script.
- Assigning distinct voices for different speakers:
  - Host voice: `af_heart`
  - Guest voice: `af_bella`
- Converting the script into speech using **Kokoro TTS**.
- Merging individual audio segments into a single podcast file.
- Saving the final audio output as: podcast.wav

### **4.3 Configuration Module (`config.py`)**

The configuration module centralizes all application-level settings, including:

- Model names and paths.
- Prompt templates for RAG and podcast generation.
- Audio settings and voice configurations.
- Directory paths and file naming conventions.

Centralizing configuration ensures consistency and simplifies future updates.

---

### **4.4 Helper Utilities Module (`helpers.py`)**

This module provides reusable helper functions that support the core modules, such as:

- PDF text extraction utilities.
- Text cleaning and preprocessing functions.
- Audio processing helpers.
- File and directory management utilities.

By isolating helper functions, the project avoids code duplication and improves maintainability.

---

## **5. Technologies Used**

The following tools and libraries are used in this project:

- **Python 3.10+**
- **Streamlit** – Interactive web application framework
- **Ollama** – Local LLM runtime
- **LLaMA3** – Large Language Model for generation
- **bge-m3** – Embedding model for semantic search
- **ChromaDB** – Vector database for RAG
- **Kokoro TTS** – Text-to-speech engine

---

## **6. Usage Workflow**

### **6.1 PDF RAG Workflow**

1. User uploads one or more PDF files.
2. The system extracts and indexes the PDF content.
3. User enters a question in Nepali, Hindi, or English.
4. Relevant document chunks are retrieved from the vector database.
5. The LLM generates an answer using retrieved context.
6. Source references are displayed to the user.

---

### **6.2 PDF to Podcast Workflow**

1. User uploads a PDF document.
2. The document content is transformed into a conversational script.
3. Host and Guest voices are synthesized separately.
4. Audio segments are merged into a single file.
5. The final podcast is generated and made available for listening or download.

---

## **7. Key Features**

- Bilingual PDF question answering
- Conversational podcast-style audio generation
- Fully offline local inference
- Source-aware RAG responses
   
