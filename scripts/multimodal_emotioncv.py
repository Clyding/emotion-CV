"""
Multimodal Emotion + Empathy Streamlit App
- Face analysis: uses a Keras .h5 model if available (default path ./models/fer_model.h5)
- Audio analysis: records short snippets and computes RMS/peaks to detect loud/distress sounds
- Chat UI: rule-based empathetic replies using emotion context
- Emergency detection: text keywords + face+audio heuristics -> shows emergency guidance

Run:
    pip install -r requirements.txt
    streamlit run multimodal_empathy_app.py
"""

import os
import sys
import time
import tempfile
import math
from typing import Dict

import numpy as np
import cv2
from tensorflow.keras.models import load_model
import sounddevice as sd
import soundfile as sf
import streamlit as st

# -----------------------
# Config / Paths
# -----------------------
FER_MODEL_PATH = os.getenv("FER_MODEL_PATH", "./models/fer_model.h5")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Audio settings
AUDIO_SR = 16000
AUDIO_DURATION = 3.0  # seconds for each recording
AUDIO_RMS_THRESHOLD = 0.12  # tuned heuristic; adjust for your mic (0-1)
AUDIO_PEAK_THRESHOLD = 0.6  # for sudden spikes

# Emergency text keywords (lowercase)
EMERGENCY_KEYWORDS = [
    "suicide", "kill myself", "i want to die", "harm myself",
    "bleeding", "bleed", "help me", "emergency", "call 911", "hurt",
    "stabbing", "shoot", "scream", "unresponsive", "not breathing"
]

# Empathy templates keyed by dominant emotion
EMPATHY_TEMPLATES = {
    "happy": [
        "That's great to hear — I’m glad you're feeling good! 😊",
        "Happy to hear that — keep enjoying the positive moment!"
    ],
    "neutral": [
        "Thanks for telling me. Do you want to say more about how you're doing?",
        "I’m here to listen — tell me more if you’d like."
    ],
    "sad": [
        "I’m really sorry you’re feeling this way. I’m here with you.",
        "That sounds really tough. You don’t have to go through it alone."
    ],
    "angry": [
        "Anger can feel overwhelming — it makes sense to feel upset.",
        "I’m sorry that happened. Do you want to talk about what made you angry?"
    ],
    "fear": [
        "That sounds scary. I’m here to support you — are you safe right now?",
        "I’m sorry you’re feeling afraid. If you can, try taking a slow breath with me."
    ],
    "surprise": [
        "That sounds unexpected. Want to tell me more about it?",
        "I can hear that surprised you — what happened next?"
    ],
    "disgust": [
        "That sounds unpleasant — it’s valid to feel that way.",
        "I hear you. Do you want to explain what happened?"
    ],
    # fallback
    "unknown": [
        "I’m here for you. Tell me more about how you're feeling.",
        "Thanks for sharing — I’m listening."
    ]
}

# -----------------------
# Helpers: Load vision model safely
# -----------------------
def load_vision_model(path: str):
    if os.path.exists(path):
        try:
            model = load_model(path, compile=False)
            st.sidebar.success(f"Loaded vision model: {os.path.basename(path)}")
            return model
        except Exception as e:
            st.sidebar.warning(f"Failed to load model at {path}: {e}")
            return None
    else:
        st.sidebar.info(f"No vision model found at {path}. Using face detector only.")
        return None

# -----------------------
# Helpers: Face -> emotion
# -----------------------
def preprocess_face_for_model(face_img, target_size=(48,48), channels=1):
    face = cv2.resize(face_img, target_size)
    if channels == 1:
        face = face.reshape(1, target_size[0], target_size[1], 1).astype('float32') / 255.0
    else:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        face = face.reshape(1, target_size[0], target_size[1], 3).astype('float32') / 255.0
    return face

def predict_emotion_from_frame(model, frame):
    """
    If model is provided, detect faces and run model to return top emotion + probs.
    If model is None, run a simple heuristic: detect face and return neutral.
    Returns: list of {emotion, confidence, probs}
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    results = []
    labels = ['angry','disgust','fear','happy','sad','surprise','neutral']
    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        if model is not None:
            # try to infer number of channels expected
            try:
                channels = model.input_shape[-1]
            except Exception:
                channels = 1
            face_input = preprocess_face_for_model(face, (48,48), channels=channels)
            probs = model.predict(face_input, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            emotion = labels[top_idx] if top_idx < len(labels) else "unknown"
            confidence = float(probs[top_idx])
            probs_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
            results.append({"emotion": emotion, "confidence": confidence, "probs": probs_dict, "box": (x,y,w,h)})
        else:
            # fallback: no model -> neutral with low confidence
            results.append({"emotion": "neutral", "confidence": 0.5, "probs": {"neutral": 0.5}, "box": (x,y,w,h)})
    return results

# -----------------------
# Helpers: Audio emergency detection (loudness & spike heuristics)
# -----------------------
def record_audio(seconds=AUDIO_DURATION, sr=AUDIO_SR):
    # records into a temporary file and returns path
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        recording = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        sf.write(tmp_path, recording, sr)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e
    return tmp_path

def analyze_audio_for_distress(filepath):
    data, sr = sf.read(filepath)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    # Normalize to [-1,1] if necessary
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data**2)))
    # simple metric: big RMS or sudden peak indicates loud distress (scream, cry)
    is_loud = rms > AUDIO_RMS_THRESHOLD
    is_spike = peak > AUDIO_PEAK_THRESHOLD
    return {"rms": rms, "peak": peak, "is_loud": is_loud, "is_spike": is_spike}

# -----------------------
# Helpers: Text emergency detection
# -----------------------
def text_indicates_emergency(text: str) -> bool:
    t = text.lower()
    for kw in EMERGENCY_KEYWORDS:
        if kw in t:
            return True
    return False

# -----------------------
# Helper: Choose empathetic reply
# -----------------------
import random
def empathetic_reply(user_text: str, emotion_probs: Dict[str, float]):
    # if user text triggers emergency, immediate safety-first reply
    if text_indicates_emergency(user_text):
        return (
            "I’m concerned about your safety. If you are in immediate danger, please call your local emergency number right now "
            "(for example, 911 in the U.S.). If you are in the U.S. and are feeling suicidal or in emotional distress, you can call or text 988 to reach the Suicide & Crisis Lifeline. "
            "If you can, please tell me if you are safe right now."
        )
    # pick dominant emotion
    if not emotion_probs:
        dominant = "unknown"
    else:
        dominant = max(emotion_probs, key=emotion_probs.get)
        if dominant not in EMPATHY_TEMPLATES:
            dominant = "unknown"
    template = random.choice(EMPATHY_TEMPLATES.get(dominant, EMPATHY_TEMPLATES["unknown"]))
    # add a follow-up prompt
    follow = " Would you like to tell me more about this?"
    return template + follow

# -----------------------
# Emergency decision logic
# -----------------------
def is_emergency_detected(user_text: str, face_results, audio_metrics) -> Dict:
    """
    Returns a dict with keys:
      - emergency (bool)
      - reasons (list of strings)
    """
    reasons = []
    emergency = False

    # 1) text keywords
    if text_indicates_emergency(user_text):
        emergency = True
        reasons.append("Text contains emergency keywords")

    # 2) face: strong sad/fear
    if face_results:
        top = face_results[0]
        if top["emotion"] in ("sad","fear") and top.get("confidence", 0.0) >= 0.85:
            emergency = True
            reasons.append(f"High-confidence face emotion: {top['emotion']} ({top['confidence']:.2f})")

    # 3) audio loudness/spike
    if audio_metrics:
        if audio_metrics.get("is_loud") or audio_metrics.get("is_spike"):
            emergency = True
            reasons.append(f"Audio spike/rms (peak={audio_metrics.get('peak'):.2f}, rms={audio_metrics.get('rms'):.2f})")

    return {"emergency": emergency, "reasons": reasons}

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="EmotionCV - Local Multimodal Empathy", layout="wide")
st.title("EmotionCV — Local Multimodal Empathetic Assistant")

# Sidebar: load model
st.sidebar.header("Model & Settings")
vision_model = load_vision_model(FER_MODEL_PATH)
st.sidebar.write("Audio RMS threshold (heuristic)")
AUDIO_RMS_THRESHOLD = st.sidebar.slider("RMS threshold", 0.01, 0.5, AUDIO_RMS_THRESHOLD, 0.01)
AUDIO_PEAK_THRESHOLD = st.sidebar.slider("Peak threshold", 0.2, 1.0, AUDIO_PEAK_THRESHOLD, 0.05)
AUDIO_DURATION = st.sidebar.slider("Audio clip duration (s)", 1.0, 6.0, AUDIO_DURATION, 0.5)

# main layout: left = camera + audio, right = chat
col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("Live Camera")
    cam_run = st.checkbox("Enable continuous webcam", value=False, key="cam_run")
    FRAME = st.empty()
    face_status = st.empty()
    last_face = None
    if cam_run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Webcam not available.")
        else:
            ret, frame = cap.read()
            if ret:
                frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME.image(frame_disp, channels="RGB")
                faces = predict_emotion_from_frame(vision_model, frame)
                if faces:
                    f = faces[0]
                    last_face = f
                    face_status.info(f"Face: {f['emotion']} ({f['confidence']:.2f})")
                else:
                    face_status.info("No face detected")
            cap.release()

    st.markdown("---")
    st.header("Microphone (short clip)")
    if st.button("Record audio clip"):
        try:
            st.info("Recording...")
            audio_path = record_audio(seconds=AUDIO_DURATION, sr=AUDIO_SR)
            metrics = analyze_audio_for_distress(audio_path)
            st.success(f"Recorded. RMS={metrics['rms']:.3f}, Peak={metrics['peak']:.3f}")
            os.remove(audio_path)
        except Exception as e:
            st.error(f"Audio recording failed: {e}")
            metrics = None
    else:
        metrics = None

    # Quick summary of sensors
    st.markdown("#### Latest sensors")
    if last_face:
        st.write(f"Face: {last_face['emotion']} ({last_face['confidence']:.2f})")
    else:
        st.write("Face: (none yet)")
    if metrics:
        st.write(f"Audio RMS: {metrics['rms']:.3f}, Peak: {metrics['peak']:.3f}")
    else:
        st.write("Audio: (none yet)")

with col2:
    st.header("Chat — empathetic assistant (local)")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_text = st.text_input("You:", key="user_input")
    send = st.button("Send")

    # Determine current sensor state
    current_face = last_face
    current_audio = metrics
    # Merge simple emotion_probs from face if present
    emotion_probs = {}
    if current_face:
        emotion_probs = current_face.get("probs", {})
    else:
        emotion_probs = {"neutral": 1.0}

    # If user sends message
    if send and user_text.strip():
        # Assess emergency
        emergency_info = is_emergency_detected(user_text.strip(), [current_face] if current_face else [], current_audio)
        if emergency_info["emergency"]:
            # Compose emergency reply (safety-first)
            reasons = emergency_info["reasons"]
            emergency_message = (
                "**EMERGENCY DETECTED**\n\n"
                "I’m concerned about your immediate safety. "
                "If you are in immediate danger, please call your local emergency number right now (for example, 911 in the U.S.).\n\n"
                "Crisis resources (USA):\n"
                "- If you are in emotional distress or thinking about suicide, call or text **988** for the Suicide & Crisis Lifeline.\n\n"
                "If you can, please tell me if you are safe right now. Reasons detected:\n"
                f"- " + "\n- ".join(reasons)
            )
            st.error(emergency_message)
            st.session_state.chat_history.append(("Assistant (EMERGENCY)", emergency_message))
        else:
            # produce an empathetic, non-professional reply using local heuristics
            reply = empathetic_reply(user_text.strip(), emotion_probs)
            st.info("Assistant reply (local):")
            st.write(reply)
            st.session_state.chat_history.append(("You", user_text.strip()))
            st.session_state.chat_history.append(("Assistant", reply))

    # Show history
    st.markdown("### Conversation")
    for who, msg in st.session_state.chat_history[-12:]:
        if who.startswith("Assistant (EMERGENCY)"):
            st.markdown(f"**{who}:** {msg}")
        else:
            st.markdown(f"**{who}:** {msg}")

# End of app
