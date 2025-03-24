import streamlit as st
import requests
import base64
import json
from audio_recorder_streamlit import audio_recorder  # Install via: pip install audio_recorder_streamlit
from dotenv import load_dotenv
import os
load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")  # Your Google Cloud API key
MODEL = "gemini-2.0-flash-exp"
# Endpoint for Gemini (Generative Language API)
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

# Import LangChain Groq for LLM text generation
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

LLM_GROQ = ChatGroq(
    model_name="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

system_prompt = """
You are a helpful assistant that answers questions in the same language as the question.
"""

# --- Function to transcribe audio using Gemini STT ---
def transcribe_audio(audio_bytes):
    encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": "Transcribe the following audio:"},
                    {"inline_data": {
                        "mime_type": "audio/wav",
                        "data": encoded_audio
                    }}
                ]
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()
        # Extract transcript from first candidate's first part
        transcript = (
            result.get("candidates", [{}])[0]
                  .get("content", {})
                  .get("parts", [{}])[0]
                  .get("text", "")
        )
        return transcript
    else:
        return f"Error {response.status_code}: {response.text}"

# --- Function to generate an answer using Gemini LLM (via LangChain Groq) ---
def generate_answer(question):
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    # Invoke your Groq-based LLM for answer generation
    return LLM_GROQ.invoke(messages).content

# --- Function to synthesize text to speech using Google Cloud TTS ---
def text_to_speech(text):
    TTS_API_URL = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={API_KEY}"
    payload = {
        "input": {"text": text},
        "voice": {
            "languageCode": "en-US",   # Change as needed, e.g. "es-ES" for Spanish, etc.
            "ssmlGender": "NEUTRAL"
        },
        "audioConfig": {"audioEncoding": "MP3"}
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(TTS_API_URL, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()
        audio_content = result.get("audioContent", None)
        if audio_content:
            return base64.b64decode(audio_content)
        else:
            st.error("TTS API returned no audio content.")
            return None
    else:
        st.error(f"TTS API error {response.status_code}: {response.text}")
        return None

# --- Streamlit UI ---
def main():
    st.title("Gemini Flash 2.0 STT & TTS Demo")
    st.write("Record your voice question below. Your audio will be transcribed, answered by the LLM, and then the answer will be read aloud.")

    st.info("Click the record button below to start recording and click again to stop.")
    audio_bytes = audio_recorder("Record your question", "Stop recording")
    
    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")
        if st.button("Transcribe, Answer & Speak"):
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio(audio_bytes)
            st.subheader("Transcribed Question")
            st.write(transcript)
            
            if transcript and not transcript.startswith("Error"):
                with st.spinner("Generating answer..."):
                    answer_text = generate_answer(transcript)
                st.subheader("Text Answer")
                st.write(answer_text)
                
                with st.spinner("Synthesizing voice..."):
                    answer_audio = text_to_speech(answer_text)
                if answer_audio:
                    st.subheader("Voice Answer")
                    st.audio(answer_audio, format="audio/mp3")
            else:
                st.error("There was an error transcribing the audio.")

if __name__ == '__main__':
    main()
