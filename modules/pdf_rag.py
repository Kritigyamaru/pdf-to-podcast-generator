import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from modules.config import CHROMA_PATH, LLAMA_MODEL, UPLOAD_DIR
from modules.helpers import get_embedding_function



RAG_PROMPT_TEMPLATE = """You are a highly accurate bilingual assistant fluent in Nepali and English. 

RULES (STRICT):
- Always respond ONLY in the same language as the user’s input.
- DO NOT translate unless explicitly asked to do so.
- If the input is in English, respond in English.
- If the input is in Nepali, respond in Nepali.
- Never mix languages in a single response.
- Your answers must rely SOLELY on the provided context. Do not add external information.
- For translation requests, translate the text completely, accurately, and naturally, preserving meaning and tone.


Context:
{context}

Question:
{question}

Answer:
"""

class PDFRAGAssistant:
    def __init__(self):
        self.llm = Ollama(model=LLAMA_MODEL)
        self.db = self.load_vector_db()

    def load_vector_db(self):
        return Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function()
        )

    def process_uploaded_pdfs(self, files):
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        new_docs = []

        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            loader = PyPDFLoader(file_path)
            documents = loader.load()
            chunks = splitter.split_documents(documents)

            for chunk in chunks:
                chunk.metadata["source"] = file.name

            new_docs.extend(chunks)

        if new_docs:
            self.db.add_documents(new_docs)
            self.db.persist()
        return len(new_docs)

    def run(self):
        st.header("PDF Retrieval and QA Assistant")

        uploaded_files = st.file_uploader(
            "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
        )

        if uploaded_files:
            with st.spinner("Indexing uploaded PDFs..."):
                added_chunks = self.process_uploaded_pdfs(uploaded_files)
                st.success(f"Indexed {added_chunks} chunks" if added_chunks else "No chunks were indexed")

        query = st.text_input("Enter your question:", placeholder="Ask a question about the uploaded PDFs...")
        top_k = 5

        if st.button("🔍 Ask"):
            if not query.strip():
                st.warning("Please enter a question.")
                return

            with st.spinner("Searching documents and generating answer..."):
                results = self.db.similarity_search_with_score(query, k=top_k)
                if not results:
                    st.error("No relevant documents found.")
                    return

                context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
                prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE).format(
                    context=context_text, question=query
                )
                response = self.llm.invoke(prompt)

                st.subheader("Answer")
                st.write(response)
                st.subheader("Sources")
                for i, (doc, score) in enumerate(results, start=1):
                    st.markdown(f"**{i}. {doc.metadata.get('source','unknown')}**  \nScore: `{score:.4f}`")


