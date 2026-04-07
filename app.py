import os
import shutil
import time
import streamlit as st
from modules.pdf_rag import PDFRAGAssistant
from modules.pdf_podcast import PDFPodcastGenerator
from modules.config import UPLOAD_DIR, CHROMA_PATH, LLAMA_MODEL
from modules.helpers import remove_chroma_folder


# Initial Setup

os.makedirs(UPLOAD_DIR, exist_ok=True)
remove_chroma_folder(CHROMA_PATH)
os.makedirs(CHROMA_PATH, exist_ok=True)

st.set_page_config(page_title="PDF Assistant and Podcast", layout="wide")
st.title("PDF Assistant and Podcast Generator")


# Streamlit Mode Selection

mode = st.sidebar.radio("Select Mode", ["PDF RAG Assistant", "PDF to Podcast"])

if mode == "PDF RAG Assistant":
    assistant = PDFRAGAssistant()
    assistant.run()

elif mode == "PDF to Podcast":
    podcast_gen = PDFPodcastGenerator()
    podcast_gen.run()
