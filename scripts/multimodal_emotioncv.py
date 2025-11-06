"""
Multimodal Emotion + Empathy Streamlit App (Improved Version)
- Face analysis: uses a Keras .h5 model if available (default path ./models/fer_model.h5)
- Audio analysis: records short snippets and computes RMS/peaks to detect loud/distress sounds
- Chat UI: rule-based empathetic replies using emotion context
- Emergency detection: text keywords + face+audio heuristics -> shows emergency guidance

Improvements:
- Fixed camera management with session state
- Better state persistence for sensors
- Enhanced UI/UX with timestamps and visual feedback
- Improved error handling and resource cleanup
- Added caching for performance
- Clear privacy disclaimers and safety notices

Run:
    pip install -r requirements.txt
    streamlit run multimodal_empathy_app.py
"""

import os
import time
import tempfile
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from tensorflow.keras.models import load_model
import sounddevice as sd
import soundfile as sf
import streamlit as st

# Configuration
FER_MODEL_PATH = os.getenv("FER_MODEL_PATH", "./models/fer_model.h5")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

AUDIO_SR = 16000
AUDIO_DURATION = 3.0  
AUDIO_RMS_THRESHOLD = 0.12  
AUDIO_PEAK_THRESHOLD = 0.6  

EMERGENCY_KEYWORDS = [
    "suicide", "kill myself", "i want to die", "harm myself",
    "bleeding", "bleed", "help me", "emergency", "call 911", "hurt",
    "stabbing", "shoot", "scream", "unresponsive", "not breathing"
]

EMPATHY_TEMPLATES = {
    "happy": [
        "That's great to hear — I'm glad you're feeling good! 😊",
        "Happy to hear that — keep enjoying the positive moment!"
    ],
    "neutral": [
        "Thanks for telling me. Do you want to say more about how you're doing?",
        "I'm here to listen — tell me more if you'd like."
    ],
    "sad": [
        "I'm really sorry you're feeling this way. I'm here with you.",
        "That sounds really tough. You don't have to go through it alone."
    ],
    "angry": [
        "Anger can feel overwhelming — it makes sense to feel upset.",
        "I'm sorry that happened. Do you want to talk about what made you angry?"
    ],
    "fear": [
        "That sounds scary. I'm here to support you — are you safe right now?",
        "I'm sorry you're feeling afraid. If you can, try taking a slow breath with me."
    ],
    "surprise": [
        "That sounds unexpected. Want to tell me more about it?",
        "I can hear that surprised you — what happened next?"
    ],
    "disgust": [
        "That sounds unpleasant — it's valid to feel that way.",
        "I hear you. Do you want to explain what happened?"
    ],
    "unknown": [
        "I'm here for you. Tell me more about how you're feeling.",
        "Thanks for sharing — I'm listening."
    ]
}

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


# ============================================================================
# INITIALIZATION & CACHING
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'last_face' not in st.session_state:
        st.session_state.last_face = None
    if 'last_audio_metrics' not in st.session_state:
        st.session_state.last_audio_metrics = None
    if 'emergency_log' not in st.session_state:
        st.session_state.emergency_log = []
    if 'camera' not in st.session_state:
        st.session_state.camera = None
    if 'camera_enabled' not in st.session_state:
        st.session_state.camera_enabled = False


@st.cache_resource
def load_face_cascade():
    """Load and cache the face cascade classifier."""
    return cv2.CascadeClassifier(CASCADE_PATH)


@st.cache_resource
def load_vision_model(path: str):
    """Load and cache the emotion recognition model."""
    if os.path.exists(path):
        try:
            model = load_model(path, compile=False)
            return model, f"✓ Loaded: {os.path.basename(path)}"
        except Exception as e:
            return None, f"⚠ Failed to load: {e}"
    else:
        return None, f"ℹ No model at {path}"


# ============================================================================
# VISION PROCESSING
# ============================================================================

def preprocess_face_for_model(face_img: np.ndarray, target_size: Tuple[int, int] = (48, 48), 
                               channels: int = 1) -> np.ndarray:
    """Preprocess face image for model input."""
    face = cv2.resize(face_img, target_size)
    if channels == 1:
        face = face.reshape(1, target_size[0], target_size[1], 1).astype('float32') / 255.0
    else:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        face = face.reshape(1, target_size[0], target_size[1], 3).astype('float32') / 255.0
    return face


def predict_emotion_from_frame(model, frame: np.ndarray, face_cascade) -> List[Dict]:
    """
    Detect faces and predict emotions.
    Returns: list of {emotion, confidence, probs, box}
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    results = []
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        if model is not None:
            try:
                channels = model.input_shape[-1]
            except Exception:
                channels = 1
            
            face_input = preprocess_face_for_model(face, (48, 48), channels=channels)
            probs = model.predict(face_input, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            emotion = EMOTION_LABELS[top_idx] if top_idx < len(EMOTION_LABELS) else "unknown"
            confidence = float(probs[top_idx])
            probs_dict = {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))}
            
            results.append({
                "emotion": emotion,
                "confidence": confidence,
                "probs": probs_dict,
                "box": (x, y, w, h)
            })
        else:
            # Fallback when no model available
            results.append({
                "emotion": "neutral",
                "confidence": 0.5,
                "probs": {"neutral": 0.5},
                "box": (x, y, w, h)
            })
    
    return results


def draw_face_boxes(frame: np.ndarray, face_results: List[Dict]) -> np.ndarray:
    """Draw bounding boxes and emotion labels on frame."""
    frame_copy = frame.copy()
    for result in face_results:
        x, y, w, h = result['box']
        emotion = result['emotion']
        confidence = result['confidence']
        
        # Draw rectangle
        color = (0, 255, 0) if confidence > 0.6 else (255, 165, 0)
        cv2.rectangle(frame_copy, (x, y), (x+w, y+h), color, 2)
        
        # Draw label
        label = f"{emotion} ({confidence:.2f})"
        cv2.putText(frame_copy, label, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame_copy


# ============================================================================
# AUDIO PROCESSING
# ============================================================================

def record_audio(seconds: float = AUDIO_DURATION, sr: int = AUDIO_SR) -> str:
    """
    Record audio and save to temporary file.
    Returns: path to temporary audio file
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        # Check if audio device is available
        devices = sd.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No audio devices found")
        
        recording = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        sf.write(tmp_path, recording, sr)
        return tmp_path
        
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Audio recording failed: {str(e)}")


def analyze_audio_for_distress(filepath: str) -> Dict:
    """
    Analyze audio file for distress indicators.
    Returns: dict with rms, peak, is_loud, is_spike
    """
    try:
        data, sr = sf.read(filepath)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        
        # Normalize and compute metrics
        peak = float(np.max(np.abs(data)))
        rms = float(np.sqrt(np.mean(data**2)))
        
        is_loud = rms > AUDIO_RMS_THRESHOLD
        is_spike = peak > AUDIO_PEAK_THRESHOLD
        
        return {
            "rms": rms,
            "peak": peak,
            "is_loud": is_loud,
            "is_spike": is_spike,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        st.error(f"Audio analysis failed: {e}")
        return None


# ============================================================================
# EMERGENCY DETECTION & EMPATHY
# ============================================================================

def text_indicates_emergency(text: str) -> bool:
    """Check if text contains emergency keywords."""
    t = text.lower()
    return any(kw in t for kw in EMERGENCY_KEYWORDS)


def is_emergency_detected(user_text: str, face_results: List[Dict], 
                          audio_metrics: Optional[Dict]) -> Dict:
    """
    Detect emergency situations from multimodal inputs.
    Returns: {emergency: bool, reasons: list, confidence: str}
    """
    reasons = []
    emergency = False
    confidence = "low"
    
    # 1) Text keywords
    if text_indicates_emergency(user_text):
        emergency = True
        confidence = "high"
        reasons.append("Text contains emergency keywords")
    
    # 2) Face: strong sad/fear
    if face_results:
        top = face_results[0]
        if top["emotion"] in ("sad", "fear") and top.get("confidence", 0.0) >= 0.85:
            emergency = True
            if confidence == "low":
                confidence = "medium"
            reasons.append(f"High-confidence face emotion: {top['emotion']} ({top['confidence']:.2f})")
    
    # 3) Audio loudness/spike
    if audio_metrics:
        if audio_metrics.get("is_loud") or audio_metrics.get("is_spike"):
            emergency = True
            if confidence == "low":
                confidence = "medium"
            reasons.append(f"Audio distress detected (peak={audio_metrics.get('peak', 0):.2f}, rms={audio_metrics.get('rms', 0):.2f})")
    
    return {
        "emergency": emergency,
        "reasons": reasons,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat()
    }


def empathetic_reply(user_text: str, emotion_probs: Dict[str, float]) -> str:
    """Generate empathetic reply based on emotion context."""
    if text_indicates_emergency(user_text):
        return (
            "I'm concerned about your safety. If you are in immediate danger, please call your local emergency number right now "
            "(for example, 911 in the U.S.). If you are in the U.S. and are feeling suicidal or in emotional distress, you can call or text 988 to reach the Suicide & Crisis Lifeline. "
            "If you can, please tell me if you are safe right now."
        )
    
    if not emotion_probs:
        dominant = "unknown"
    else:
        dominant = max(emotion_probs, key=emotion_probs.get)
        if dominant not in EMPATHY_TEMPLATES:
            dominant = "unknown"
    
    template = random.choice(EMPATHY_TEMPLATES.get(dominant, EMPATHY_TEMPLATES["unknown"]))
    follow = " Would you like to tell me more about this?"
    return template + follow


# ============================================================================
# CAMERA MANAGEMENT
# ============================================================================

def manage_camera(enable: bool) -> Optional[cv2.VideoCapture]:
    """Manage camera lifecycle with session state."""
    if enable:
        if st.session_state.camera is None or not st.session_state.camera.isOpened():
            st.session_state.camera = cv2.VideoCapture(0)
            time.sleep(0.5)  # Allow camera to initialize
        st.session_state.camera_enabled = True
        return st.session_state.camera
    else:
        if st.session_state.camera is not None:
            st.session_state.camera.release()
            st.session_state.camera = None
        st.session_state.camera_enabled = False
        return None


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_disclaimer():
    """Render safety disclaimer banner."""
    st.warning("""
    ⚠️ **IMPORTANT DISCLAIMER**: This application is a prototype and is NOT a substitute for professional mental health care. 
    If you are experiencing a mental health emergency, please contact emergency services immediately or call the Suicide & Crisis Lifeline at 988 (US).
    
    **Privacy Notice**: All processing happens locally. No data is sent to external servers. Camera and microphone access is optional.
    """)


def render_emergency_resources():
    """Render emergency resources in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🆘 Emergency Resources")
    st.sidebar.markdown("""
    **USA:**
    - 911 (Emergency)
    - 988 (Suicide & Crisis)
    
    **International:**
    - Find your local crisis line at [findahelpline.com](https://findahelpline.com)
    """)


def format_timestamp(dt: datetime = None) -> str:
    """Format timestamp for display."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%H:%M:%S")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(
        page_title="EmotionCV - Local Multimodal Empathy", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("EmotionCV — Local Multimodal Empathetic Assistant")
    
    # Initialize session state
    initialize_session_state()
    
    # Render disclaimer
    render_disclaimer()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Model & Settings")
    
    vision_model, model_status = load_vision_model(FER_MODEL_PATH)
    st.sidebar.info(model_status)
    
    face_cascade = load_face_cascade()
    
    st.sidebar.markdown("### Audio Settings")
    audio_duration = st.sidebar.slider("Audio clip duration (s)", 1.0, 6.0, AUDIO_DURATION, 0.5)
    audio_rms_threshold = st.sidebar.slider("RMS threshold", 0.01, 0.5, AUDIO_RMS_THRESHOLD, 0.01)
    audio_peak_threshold = st.sidebar.slider("Peak threshold", 0.2, 1.0, AUDIO_PEAK_THRESHOLD, 0.05)
    
    render_emergency_resources()
    
    # Main layout
    col1, col2 = st.columns([1, 1.2])
    
    # ========================================================================
    # LEFT COLUMN: Camera & Audio
    # ========================================================================
    with col1:
        st.header("📹 Live Camera")
        cam_run = st.checkbox("Enable continuous webcam", value=st.session_state.camera_enabled, key="cam_run")
        
        FRAME_PLACEHOLDER = st.empty()
        face_status_placeholder = st.empty()
        
        cap = manage_camera(cam_run)
        
        if cam_run and cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                faces = predict_emotion_from_frame(vision_model, frame, face_cascade)
                
                # Draw boxes on frame
                frame_with_boxes = draw_face_boxes(frame, faces)
                frame_disp = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
                FRAME_PLACEHOLDER.image(frame_disp, channels="RGB", use_container_width=True)
                
                if faces:
                    f = faces[0]
                    st.session_state.last_face = f
                    face_status_placeholder.success(f"✓ Face detected: {f['emotion']} ({f['confidence']:.2f})")
                else:
                    face_status_placeholder.info("No face detected")
                
                # CRITICAL: Auto-rerun to keep camera feed live
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                st.rerun()
            else:
                face_status_placeholder.error("Failed to read from camera")
        elif cam_run:
            face_status_placeholder.error("❌ Webcam not available")
        
        st.markdown("---")
        st.header("🎤 Microphone (short clip)")
        
        record_col1, record_col2 = st.columns([1, 1])
        with record_col1:
            record_button = st.button("🔴 Record Audio", use_container_width=True)
        
        if record_button:
            try:
                with st.spinner(f"Recording {audio_duration}s..."):
                    audio_path = record_audio(seconds=audio_duration, sr=AUDIO_SR)
                metrics = analyze_audio_for_distress(audio_path)
                
                if metrics:
                    st.session_state.last_audio_metrics = metrics
                    
                    if metrics['is_loud'] or metrics['is_spike']:
                        st.warning(f"⚠️ Distress detected: RMS={metrics['rms']:.3f}, Peak={metrics['peak']:.3f}")
                    else:
                        st.success(f"✓ Recorded: RMS={metrics['rms']:.3f}, Peak={metrics['peak']:.3f}")
                
                # Cleanup
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    
            except Exception as e:
                st.error(f"❌ Audio recording failed: {e}")
                st.session_state.last_audio_metrics = None
        
        st.markdown("---")
        st.markdown("### 📊 Latest Sensor Data")
        
        sensor_col1, sensor_col2 = st.columns(2)
        with sensor_col1:
            st.markdown("**Face:**")
            if st.session_state.last_face:
                st.write(f"• {st.session_state.last_face['emotion']}")
                st.write(f"• Confidence: {st.session_state.last_face['confidence']:.2f}")
            else:
                st.write("_No data yet_")
        
        with sensor_col2:
            st.markdown("**Audio:**")
            if st.session_state.last_audio_metrics:
                m = st.session_state.last_audio_metrics
                st.write(f"• RMS: {m['rms']:.3f}")
                st.write(f"• Peak: {m['peak']:.3f}")
            else:
                st.write("_No data yet_")
    
    # ========================================================================
    # RIGHT COLUMN: Chat Interface
    # ========================================================================
    with col2:
        st.header("💬 Chat — Empathetic Assistant")
        
        # Chat input
        user_text = st.text_input("You:", key="user_input", placeholder="Type your message here...")
        send_col1, send_col2, send_col3 = st.columns([1, 1, 3])
        
        with send_col1:
            send = st.button("📤 Send", use_container_width=True)
        with send_col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # Process message
        if send and user_text.strip():
            current_face = st.session_state.last_face
            current_audio = st.session_state.last_audio_metrics
            
            # Get emotion probabilities
            emotion_probs = {}
            if current_face:
                emotion_probs = current_face.get("probs", {})
            else:
                emotion_probs = {"neutral": 1.0}
            
            # Check for emergency
            emergency_info = is_emergency_detected(
                user_text.strip(),
                [current_face] if current_face else [],
                current_audio
            )
            
            timestamp = format_timestamp()
            
            if emergency_info["emergency"]:
                # Log emergency
                st.session_state.emergency_log.append(emergency_info)
                
                reasons = emergency_info["reasons"]
                emergency_message = (
                    "**🆘 EMERGENCY DETECTED**\n\n"
                    "I'm concerned about your immediate safety. "
                    "If you are in immediate danger, please call your local emergency number right now (for example, 911 in the U.S.).\n\n"
                    "**Crisis Resources (USA):**\n"
                    "- Emotional distress or suicidal thoughts: Call or text **988** for the Suicide & Crisis Lifeline\n"
                    "- Emergency services: **911**\n\n"
                    "If you can, please tell me if you are safe right now.\n\n"
                    f"_Detection reasons:_\n" + "\n".join([f"• {r}" for r in reasons])
                )
                
                st.error(emergency_message)
                st.session_state.chat_history.append((timestamp, "You", user_text.strip()))
                st.session_state.chat_history.append((timestamp, "Assistant (EMERGENCY)", emergency_message))
            else:
                reply = empathetic_reply(user_text.strip(), emotion_probs)
                st.session_state.chat_history.append((timestamp, "You", user_text.strip()))
                st.session_state.chat_history.append((timestamp, "Assistant", reply))
            
            # Clear input (workaround)
            st.rerun()
        
        # Display conversation
        st.markdown("---")
        st.markdown("### 📝 Conversation History")
        
        if not st.session_state.chat_history:
            st.info("_No messages yet. Start a conversation above._")
        else:
            # Show last 20 messages
            for timestamp, who, msg in st.session_state.chat_history[-20:]:
                if "EMERGENCY" in who:
                    st.markdown(f"**[{timestamp}] {who}:**")
                    st.error(msg)
                elif who == "You":
                    st.markdown(f"**[{timestamp}] {who}:** {msg}")
                else:
                    st.markdown(f"**[{timestamp}] {who}:**")
                    st.info(msg)
        
        # Export conversation
        if st.session_state.chat_history:
            st.markdown("---")
            if st.button("💾 Export Conversation"):
                export_text = "\n\n".join([
                    f"[{ts}] {who}: {msg}" 
                    for ts, who, msg in st.session_state.chat_history
                ])
                st.download_button(
                    label="Download as TXT",
                    data=export_text,
                    file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )


if __name__ == "__main__":
    main()