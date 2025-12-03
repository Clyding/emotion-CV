# multimodal_empathy_app.py
"""
Multimodal Emotion + Empathy Streamlit App (FFmpeg-integrated)
- Face analysis: uses a Keras .h5 model if available (default path ./models/emotion_cnn_best.h5)
- Audio analysis: records short snippets and computes RMS/peaks to detect loud/distress sounds
- Chat UI: rule-based empathetic replies using emotion context
- Emergency detection: text keywords + face+audio heuristics -> shows emergency guidance

Notes:
- This version prefers native OpenCV capture but falls back to FFmpeg frame capture (subprocess) which is WSL-friendly.
- For audio: it tries sounddevice+soufile first, and falls back to ffmpeg ALSA capture if necessary.
"""

import os
import time
import tempfile
import random
import subprocess
import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from tensorflow.keras.models import load_model
import sounddevice as sd
import soundfile as sf
import streamlit as st

# -----------------------
# Configuration (edit as needed)
# -----------------------
FER_MODEL_PATH = os.getenv("FER_MODEL_PATH", "./models/emotion_cnn_best.h5")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

AUDIO_SR = 16000
AUDIO_DURATION = 3.0
AUDIO_RMS_THRESHOLD = 0.12
AUDIO_PEAK_THRESHOLD = 0.6

EMERGENCY_KEYWORDS = [
    "suicide", "kill myself", "i want to die", "harm myself", "cut myself",
    "bleeding", "bleed", "help me", "emergency", "call 911", "hurt", "death",
    "stabbing", "shoot", "scream", "unresponsive", "not breathing", "gun"
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

# -----------------------
# Session initialization
# -----------------------
def initialize_session_state():
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
    if 'camera_backend' not in st.session_state:
        st.session_state.camera_backend = None
    if 'detected_cameras' not in st.session_state:
        st.session_state.detected_cameras = []
    if 'selected_device' not in st.session_state:
        st.session_state.selected_device = "/dev/video0"


# -----------------------
# Model & cascade loaders
# -----------------------
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(CASCADE_PATH)


@st.cache_resource
def load_vision_model(path: str):
    if os.path.exists(path):
        try:
            model = load_model(path, compile=False)
            return model, f"✓ Loaded: {os.path.basename(path)}"
        except Exception as e:
            return None, f"⚠ Failed to load model: {e}"
    else:
        return None, f"ℹ No model at {path}"


# -----------------------
# Face preprocessing & prediction
# -----------------------
def preprocess_face_for_model(face_img: np.ndarray, target_size: Tuple[int, int] = (48, 48),
                               channels: int = 1) -> np.ndarray:
    face = cv2.resize(face_img, target_size)
    if channels == 1:
        face = face.reshape(1, target_size[0], target_size[1], 1).astype('float32') / 255.0
    else:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        face = face.reshape(1, target_size[0], target_size[1], 3).astype('float32') / 255.0
    return face


def predict_emotion_from_frame(model, frame: np.ndarray, face_cascade) -> List[Dict]:
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
            results.append({
                "emotion": "neutral",
                "confidence": 0.5,
                "probs": {"neutral": 0.5},
                "box": (x, y, w, h)
            })
    return results


def draw_face_boxes(frame: np.ndarray, face_results: List[Dict]) -> np.ndarray:
    frame_copy = frame.copy()
    for result in face_results:
        x, y, w, h = result['box']
        emotion = result['emotion']
        confidence = result['confidence']
        color = (0, 255, 0) if confidence > 0.6 else (255, 165, 0)
        cv2.rectangle(frame_copy, (x, y), (x+w, y+h), color, 2)
        label = f"{emotion} ({confidence:.2f})"
        cv2.putText(frame_copy, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame_copy


# -----------------------
# Audio recording & analysis
# -----------------------
def record_audio_sounddevice(seconds: float = AUDIO_DURATION, sr: int = AUDIO_SR) -> Optional[str]:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        devices = sd.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No audio devices found via sounddevice")
        recording = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        sf.write(tmp_path, recording, sr)
        return tmp_path
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return None


def ffmpeg_record_audio(path: str, duration: float = AUDIO_DURATION, sr: int = AUDIO_SR) -> Optional[str]:
    """
    Records audio using ffmpeg via ALSA (WSL or Linux). Returns path if successful, else None.
    Note: ALSA devices must be accessible. This works in Linux/WSL with appropriate audio bridge.
    """
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "alsa",
            "-i", "default",
            "-t", str(duration),
            "-ac", "1",
            "-ar", str(sr),
            path
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=int(duration + 5))
        return path
    except Exception:
        return None


def record_audio(seconds: float = AUDIO_DURATION, sr: int = AUDIO_SR) -> str:
    """
    Try sounddevice first, then ffmpeg fallback. Raises RuntimeError on failure.
    """
    # Try sounddevice
    path = record_audio_sounddevice(seconds=seconds, sr=sr)
    if path:
        return path

    # Try ffmpeg fallback (writes to a temp file)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        fallback_path = tmp.name
    ff_path = ffmpeg_record_audio(fallback_path, duration=seconds, sr=sr)
    if ff_path:
        return ff_path

    # All failed
    if os.path.exists(fallback_path):
        os.remove(fallback_path)
    raise RuntimeError("Audio recording failed: no available backend")


def analyze_audio_for_distress(filepath: str) -> Optional[Dict]:
    try:
        data, sr = sf.read(filepath)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
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


# -----------------------
# Emergency detection & empathy
# -----------------------
def text_indicates_emergency(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in EMERGENCY_KEYWORDS)


def is_emergency_detected(user_text: str, face_results: List[Dict], audio_metrics: Optional[Dict]) -> Dict:
    reasons = []
    emergency = False
    confidence = "low"

    if text_indicates_emergency(user_text):
        emergency = True
        confidence = "high"
        reasons.append("Text contains emergency keywords")

    if face_results:
        top = face_results[0]
        if top["emotion"] in ("sad", "fear") and top.get("confidence", 0.0) >= 0.85:
            emergency = True
            if confidence == "low":
                confidence = "medium"
            reasons.append(f"High-confidence face emotion: {top['emotion']} ({top['confidence']:.2f})")

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


# -----------------------
# FFmpeg helpers (subprocess-based)
# -----------------------
def check_ffmpeg_available() -> Tuple[bool, str]:
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=2)
        version_line = result.stdout.split('\n')[0]
        return True, version_line
    except FileNotFoundError:
        return False, "Not installed"
    except Exception as e:
        return False, str(e)


def detect_cameras_with_ffmpeg() -> List[Dict]:
    cameras = []
    devices = sorted(glob.glob('/dev/video*'))
    for device in devices:
        try:
            cmd = ['ffmpeg', '-f', 'v4l2', '-list_formats', 'all', '-i', device]
            result = subprocess.run(cmd, capture_output=True, text=True, stderr=subprocess.STDOUT, timeout=2)
            if 'video' in result.stdout.lower() or 'Input #0' in result.stdout:
                # Pull index
                try:
                    device_index = int(device.split('video')[-1])
                except:
                    device_index = 0
                cameras.append({
                    'device': device,
                    'index': device_index,
                    'name': f"Camera {device_index} ({device})"
                })
        except Exception:
            continue
    return cameras


def ffmpeg_capture_frame(device_path="/dev/video0", width=640, height=480, timeout=3) -> Optional[np.ndarray]:
    """
    Capture a single frame using ffmpeg and return a BGR numpy array.
    This uses ffmpeg to read from the v4l2 device and outputs one MJPEG frame to stdout.
    """
    try:
        cmd = [
            "ffmpeg",
            "-f", "v4l2",
            "-video_size", f"{width}x{height}",
            "-i", device_path,
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-"
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if proc.returncode != 0 or not proc.stdout:
            return None
        jpg_bytes = proc.stdout
        arr = np.frombuffer(jpg_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


# -----------------------
# Camera management
# -----------------------
def manage_camera(enable: bool, device_path: str = "/dev/video0") -> Optional[cv2.VideoCapture]:
    """
    Try OpenCV capture backends (FFmpeg, V4L2, any). If enabling and we successfully open,
    store in session_state.camera. Do not try to spawn ffmpeg process here — ffmpeg fallback will be used
    for single-frame capture when OpenCV capture fails.
    """
    if enable:
        if st.session_state.camera is None or (hasattr(st.session_state.camera, "isOpened") and not st.session_state.camera.isOpened()):
            cap = None
            backend_used = "none"
            backends = [
                (cv2.CAP_FFMPEG, "OpenCV-FFmpeg"),
                (cv2.CAP_V4L2, "V4L2"),
                (cv2.CAP_ANY, "Auto")
            ]
            device_index = 0
            if 'video' in device_path:
                try:
                    device_index = int(device_path.split('video')[-1])
                except:
                    device_index = 0

            for backend, name in backends:
                try:
                    cap = cv2.VideoCapture(device_index, backend)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        # test read
                        ret, _ = cap.read()
                        if ret:
                            backend_used = name
                            break
                        else:
                            cap.release()
                            cap = None
                    else:
                        cap = None
                except Exception:
                    if cap:
                        cap.release()
                    cap = None
                    continue

            if cap is not None and cap.isOpened():
                st.session_state.camera = cap
                st.session_state.camera_enabled = True
                st.session_state.camera_backend = backend_used
                return cap
            else:
                st.session_state.camera_backend = "Failed"
                return None

        st.session_state.camera_enabled = True
        return st.session_state.camera
    else:
        if st.session_state.camera is not None:
            try:
                st.session_state.camera.release()
            except Exception:
                pass
            st.session_state.camera = None
        st.session_state.camera_enabled = False
        return None


# -----------------------
# UI helpers
# -----------------------
def render_disclaimer():
    st.warning("""
    ⚠️ **IMPORTANT DISCLAIMER**: This application is a prototype and is NOT a substitute for professional mental health care. 
    If you are experiencing a mental health emergency, please contact emergency services immediately or call the Suicide & Crisis Lifeline at 988 (US).
    
    **Privacy Notice**: All processing happens locally. No data is sent to external servers. Camera and microphone access is optional.
    """)


def render_emergency_resources():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🆘 Emergency Resources")
    st.sidebar.markdown("""
    **USA:**
    - 911 (Emergency)
    - 988 (Suicide & Crisis)
    
    **International:**
    - Find your local crisis line at https://findahelpline.com
    """)


def format_timestamp(dt: datetime = None) -> str:
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%H:%M:%S")


# -----------------------
# Sidebar & camera UI
# -----------------------
def render_sidebar_with_ffmpeg():
    st.sidebar.header("⚙️ Model & Settings")
    vision_model, model_status = load_vision_model(FER_MODEL_PATH)
    st.sidebar.info(model_status)

    st.sidebar.markdown("### 🎬 FFmpeg Status")
    ffmpeg_ok, ffmpeg_msg = check_ffmpeg_available()
    if ffmpeg_ok:
        st.sidebar.success(f"✓ {ffmpeg_msg}")
    else:
        st.sidebar.warning(f"⚠ FFmpeg: {ffmpeg_msg}")
        st.sidebar.code("sudo apt-get install ffmpeg")

    st.sidebar.markdown("### 📹 Camera Settings")
    if st.sidebar.button("🔍 Detect Cameras (FFmpeg)"):
        if ffmpeg_ok:
            with st.spinner("Detecting cameras..."):
                cameras = detect_cameras_with_ffmpeg()
                st.session_state.detected_cameras = cameras
                if cameras:
                    st.sidebar.success(f"Found {len(cameras)} camera(s)")
                else:
                    st.sidebar.warning("No cameras detected")
        else:
            st.sidebar.error("FFmpeg required for detection")

    if st.session_state.detected_cameras:
        camera_options = {cam['name']: cam['device'] for cam in st.session_state.detected_cameras}
        selected_name = st.sidebar.selectbox("Camera", list(camera_options.keys()))
        selected_device = camera_options[selected_name]
    else:
        device_index = st.sidebar.number_input("Camera Index", 0, 10, 0)
        selected_device = f"/dev/video{device_index}"

    st.session_state.selected_device = selected_device

    if 'camera_backend' in st.session_state and st.session_state.camera_backend:
        st.sidebar.info(f"Backend: {st.session_state.camera_backend}")

    st.sidebar.markdown("### 🎤 Audio Settings")
    audio_duration = st.sidebar.slider("Audio duration (s)", 1.0, 6.0, AUDIO_DURATION, 0.5)

    return selected_device, audio_duration


def render_camera_section_ffmpeg(vision_model, face_cascade):
    st.header("📹 Live Camera")
    selected_device = st.session_state.get('selected_device', '/dev/video0')

    cam_run = st.checkbox("Enable continuous webcam", value=st.session_state.camera_enabled, key="cam_run")
    FRAME_PLACEHOLDER = st.empty()
    face_status_placeholder = st.empty()

    # OpenCV VideoCapture attempt
    cap = manage_camera(cam_run, selected_device)

    # Determine whether to use ffmpeg fallback
    use_ffmpeg = cam_run and (cap is None or not getattr(cap, "isOpened", lambda: False)())

    if cam_run:
        if use_ffmpeg:
            frame = ffmpeg_capture_frame(selected_device)
            if frame is None:
                face_status_placeholder.error(f"FFmpeg capture failed for {selected_device}")
            else:
                faces = predict_emotion_from_frame(vision_model, frame, face_cascade)
                frame_with_boxes = draw_face_boxes(frame, faces)
                frame_disp = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
                FRAME_PLACEHOLDER.image(frame_disp, channels="RGB", use_container_width=True)
                if faces:
                    f = faces[0]
                    st.session_state.last_face = f
                    face_status_placeholder.success(
                        f"✓ Face: {f['emotion']} ({f['confidence']:.2f}) [FFmpeg]"
                    )
                else:
                    face_status_placeholder.info("No face detected")
        else:
            # Use OpenCV capture
            if cap is None or not cap.isOpened():
                face_status_placeholder.error(f"❌ Cannot open {selected_device} via OpenCV")
            else:
                ret, frame = cap.read()
                if not ret:
                    face_status_placeholder.error("Failed to read frame from camera")
                else:
                    faces = predict_emotion_from_frame(vision_model, frame, face_cascade)
                    frame_with_boxes = draw_face_boxes(frame, faces)
                    frame_disp = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
                    FRAME_PLACEHOLDER.image(frame_disp, channels="RGB", use_container_width=True)
                    if faces:
                        f = faces[0]
                        st.session_state.last_face = f
                        face_status_placeholder.success(
                            f"✓ Face: {f['emotion']} ({f['confidence']:.2f}) [{st.session_state.get('camera_backend','OpenCV')}]"
                        )
                    else:
                        face_status_placeholder.info("No face detected")

    else:
        # Camera not running
        FRAME_PLACEHOLDER.info("Camera is disabled. Enable the webcam above.")

# -----------------------
# Main app
# -----------------------
def main():
    st.set_page_config(page_title="EmotionCV - FFmpeg Enhanced", layout="wide", initial_sidebar_state="expanded")
    st.title("EmotionCV — FFmpeg-Enhanced Multimodal Assistant")

    initialize_session_state()
    render_disclaimer()

    # Sidebar + selection
    selected_device, audio_duration = render_sidebar_with_ffmpeg()

    # Load resources
    face_cascade = load_face_cascade()
    vision_model, _ = load_vision_model(FER_MODEL_PATH)
    render_emergency_resources()

    # Layout
    col1, col2 = st.columns([1, 1.2])

    with col1:
        render_camera_section_ffmpeg(vision_model, face_cascade)

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
                if audio_path and os.path.exists(audio_path):
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
                lf = st.session_state.last_face
                st.write(f"• {lf['emotion']}")
                st.write(f"• Confidence: {lf['confidence']:.2f}")
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

    with col2:
        st.header("💬 Chat — Empathetic Assistant")

        user_text = st.text_input("You:", key="user_input", placeholder="Type your message here...")
        send_col1, send_col2, send_col3 = st.columns([1, 1, 3])
        with send_col1:
            send = st.button("📤 Send", use_container_width=True)
        with send_col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        if send and user_text.strip():
            current_face = st.session_state.last_face
            current_audio = st.session_state.last_audio_metrics
            emotion_probs = {}
            if current_face:
                emotion_probs = current_face.get("probs", {})
            else:
                emotion_probs = {"neutral": 1.0}

            emergency_info = is_emergency_detected(user_text.strip(), [current_face] if current_face else [], current_audio)
            timestamp = format_timestamp()

            if emergency_info["emergency"]:
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

            st.rerun()

        st.markdown("---")
        st.markdown("### 📝 Conversation History")
        if not st.session_state.chat_history:
            st.info("_No messages yet. Start a conversation above._")
        else:
            for timestamp, who, msg in st.session_state.chat_history[-50:]:
                if "EMERGENCY" in who:
                    st.markdown(f"**[{timestamp}] {who}:**")
                    st.error(msg)
                elif who == "You":
                    st.markdown(f"**[{timestamp}] {who}:** {msg}")
                else:
                    st.markdown(f"**[{timestamp}] {who}:**")
                    st.info(msg)

        if st.session_state.chat_history:
            st.markdown("---")
            if st.button("💾 Export Conversation"):
                export_text = "\n\n".join([f"[{ts}] {who}: {msg}" for ts, who, msg in st.session_state.chat_history])
                st.download_button(
                    label="Download as TXT",
                    data=export_text,
                    file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )


if __name__ == "__main__":
    main()
