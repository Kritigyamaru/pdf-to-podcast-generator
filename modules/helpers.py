import shutil
import time
import os

from langchain_community.embeddings.ollama import OllamaEmbeddings


def remove_chroma_folder(path):
   
    for _ in range(5):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            break
        except PermissionError:
            time.sleep(0.2)


def get_embedding_function():
   
    return OllamaEmbeddings(
        model="bge-m3"
    )
