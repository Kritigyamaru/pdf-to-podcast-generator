import re
import os
import numpy as np
import streamlit as st
import fitz  
import soundfile as sf

from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from modules.config import LLAMA_MODEL
from modules.helpers import get_embedding_function


try:
    from kokoro import KPipeline
except ImportError:
    st.error("The 'kokoro' library is required. Please install using 'pip install kokoro'")

class PDFPodcastGenerator:
    def __init__(self):
        self.llm = Ollama(model=LLAMA_MODEL)
        self.pipeline = KPipeline(lang_code='a') if 'KPipeline' in globals() else None

    
    # TEXT CLEANING
   
    @staticmethod
    def clean_pdf_text(text):
        text = re.sub(r"\([^)]*\d{4}[^)]*\)", "", text)
        text = re.sub(r"\bReferences\b.*", "", text, flags=re.I | re.S)
        text = re.sub(r"\n\d+\s*\n", "\n", text)
        text = re.sub(r"[=±∑√≤≥≈]", "", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()

    
    # DIALOGUE EXTRACTION
    
    @staticmethod
    def extract_dialogue(script_text, host, guest):
        lines = []
        for line in script_text.split("\n"):
            line = re.sub(r"[\*_]+", "", line).strip()
            if ":" in line:
                speaker, text = line.split(":", 1)
                if speaker.lower() in [host.lower(), guest.lower()]:
                    lines.append(f"{speaker.strip()}: {text.strip()}")
        return lines

    
    # TTS
    
    def tts(self, text, voice):
        chunks = []
        for _, _, audio in self.pipeline(text + " ...", voice=voice, speed=1):
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            chunks.append(audio)
        return np.concatenate(chunks)

    
    # PODCAST GENERATOR
    
    def run(self):
        st.header("Generate Podcast from PDF")

        uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
        if not uploaded_file:
            return

        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        raw_text = "".join([p.get_text() for p in pdf])
        if not raw_text.strip():
            st.error("No readable text found in PDF.")
            return

        full_text = self.clean_pdf_text(raw_text)
        st.success("PDF text extracted & cleaned")

        # Chunk content for stability
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        chunks = splitter.split_text(full_text)
        content = "\n".join(chunks[:8])

        # Podcast script
        host_name = "Emily"
        guest_name = "James"

        podcast_prompt = f"""
        You are a professional podcast script writer.
        Create a complete podcast episode script that includes:
        1. An engaging INTRO to welcome listeners.
        2. A natural conversation between {host_name} (host) and {guest_name} (guest) 
        explaining the following content in a friendly, clear, and easy-to-understand way.
        Use ONLY the content provided below. Do not add any information or commentary that is not in the content.
        Explain everything in your own words, but strictly based on the given content.
        3. A warm and friendly OUTRO to close the episode.
        Rules:
        - Do not mention "document", "PDF", or "paper"
        - Keep dialogue short (1–2 sentences per turn)
        - Always prefix lines with: {host_name}: or {guest_name}:
        - Avoid equations, tables, citations, or section titles
        - Make it sound like a real podcast, flowing naturally
        Content:
        {{content}}
        """

        chain = ChatPromptTemplate.from_template(podcast_prompt) | self.llm | StrOutputParser()
        st.info("Generating full podcast script...")
        script = chain.invoke({"content": content})

        st.text_area("Full Podcast Script", script, height=400)
        dialogue_lines = self.extract_dialogue(script, host_name, guest_name)

        if not dialogue_lines:
            st.error("Could not extract dialogue lines. Try a different PDF.")
            return

        if self.pipeline:
            self.generate_audio(dialogue_lines, host_name, guest_name)

    
    # AUDIO GENERATION
    
    def generate_audio(self, dialogue_lines, host_name, guest_name):
        SAMPLE_RATE = 24000
        audio_segments = []

        # INTRO music
        intro_file = "intro_music.mp3"
        if os.path.exists(intro_file):
            music, sr = sf.read(intro_file)
            if music.ndim > 1:
                music = music.mean(axis=1)
            audio_segments.append(music[:5 * sr])  

        # Generate main podcast audio
        progress = st.progress(0)
        for i, line in enumerate(dialogue_lines):
            progress.progress(int((i / len(dialogue_lines)) * 100))
            speaker, text = line.split(":", 1)
            voice = "af_heart" if speaker.lower() == host_name.lower() else "am_fenrir"
            audio_segments.append(self.tts(text.strip(), voice))

        # OUTRO music
        outro_file = "outro_music.mp3"
        if os.path.exists(outro_file):
            music, sr = sf.read(outro_file)
            if music.ndim > 1:
                music = music.mean(axis=1)
            audio_segments.append(music[-5 * sr:])  

        # Save final audio
        final_audio = np.concatenate(audio_segments)
        sf.write("podcast.wav", final_audio, SAMPLE_RATE)
        st.audio("podcast.wav")
        st.success("Podcast generated successfully!")

