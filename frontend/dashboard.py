import os
import sys
import cv2
import numpy as np
import streamlit as st
import requests
import time
import tempfile
import sounddevice as sd
import soundfile as sf

# -------------------------------
# Paths & Backend URL
# -------------------------------
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
scripts_dir = os.path.join(root_dir, "scripts")
sys.path.append(scripts_dir)

from gpt import ask_gpt

BACKEND_URL = "http://localhost:8000"

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("EmotionCV - Multimodal Real-Time Emotion Chat")

# -------------------------------
# Initialize session state
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "emotion_probs" not in st.session_state:
    st.session_state.emotion_probs = {"neutral": 1.0}

if "running_webcam" not in st.session_state:
    st.session_state.running_webcam = False

if "running_mic" not in st.session_state:
    st.session_state.running_mic = False

# -------------------------------
# Chat interface
# -------------------------------
st.markdown("### 💬 Chat")
user_input = st.text_input("Your message:")

if st.button("Send") and user_input.strip():
    reply = ask_gpt(user_input, st.session_state.emotion_probs)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Assistant", reply))

# Display chat history
st.markdown("### 🗨️ Conversation")
for speaker, text in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {text}")

# -------------------------------
# Webcam toggle
# -------------------------------
st.markdown("### 📷 Webcam Emotion Detection")
start_webcam = st.button("Start Webcam") or st.session_state.running_webcam
stop_webcam = st.button("Stop Webcam")

FRAME_WINDOW = st.image([])
EMOTION_BAR = st.bar_chart([])

if stop_webcam:
    st.session_state.running_webcam = False

if start_webcam:
    st.session_state.running_webcam = True
    cap = cv2.VideoCapture(0)

# -------------------------------
# Microphone toggle
# -------------------------------
st.markdown("### 🎤 Microphone Emotion Detection")
start_mic = st.button("Start Microphone") or st.session_state.running_mic
stop_mic = st.button("Stop Microphone")

if stop_mic:
    st.session_state.running_mic = False

if start_mic:
    st.session_state.running_mic = True

# -------------------------------
# Main loop
# -------------------------------
while st.session_state.running_webcam or st.session_state.running_mic:
    # -------------------------------
    # Webcam processing
    # -------------------------------
    if st.session_state.running_webcam:
        ret, frame = cap.read()
        if ret:
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            _, img_bytes = cv2.imencode(".jpg", frame)
            files = {"file": ("frame.jpg", img_bytes.tobytes(), "image/jpeg")}
            try:
                response = requests.post(f"{BACKEND_URL}/analyze_face/", files=files, timeout=5)
                results = response.json()["results"]
                if results:
                    emotion_data = results[0]
                    st.session_state.emotion_probs = emotion_data["probs"]
                    EMOTION_BAR.bar_chart(st.session_state.emotion_probs)
            except Exception as e:
                st.warning(f"Webcam analysis error: {e}")

    # -------------------------------
    # Microphone processing
    # -------------------------------
    if st.session_state.running_mic:
        duration = 3  # seconds
        fs = 16000
        try:
            st.info("Recording audio...")
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, audio, fs)
                files = {"file": (os.path.basename(tmp_file.name), open(tmp_file.name, "rb"), "audio/wav")}
                try:
                    response = requests.post(f"{BACKEND_URL}/analyze_voice/", files=files, timeout=5)
                    voice_results = response.json().get("results")
                    if voice_results:
                        voice_probs = voice_results.get("probs", {})
                        # Merge with webcam emotion probs (average)
                        for k, v in voice_probs.items():
                            st.session_state.emotion_probs[k] = (
                                st.session_state.emotion_probs.get(k, 0.0) + v
                            ) / 2
                        EMOTION_BAR.bar_chart(st.session_state.emotion_probs)
                except Exception as e:
                    st.warning(f"Voice analysis error: {e}")
                finally:
                    tmp_file.close()
                    os.remove(tmp_file.name)
        except Exception as e:
            st.warning(f"Microphone error: {e}")

    time.sleep(0.5)

# Release webcam if stopped
if cap:
    cap.release()
