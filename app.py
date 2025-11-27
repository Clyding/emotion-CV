import streamlit as st
import sqlite3
from datetime import datetime
from pathlib import Path
import sys
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms
import librosa  # for audio loading/resampling


# ---------------------- PATHS & GLOBALS ---------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "emotioncv_history.db"
CKPT_FACE = PROJECT_ROOT / "checkpoints" / "face_emotion_best.pth"
CKPT_AUDIO = PROJECT_ROOT / "checkpoints" / "audio_emotion_best.pth"
HAAR_PATH = PROJECT_ROOT / "haarcascade_frontalface_default.xml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from emotioncv.data.datasets import EMOTION_LABELS
    EMOTIONS = EMOTION_LABELS
except Exception:
    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


# ====================== DB HELPERS ====================== #

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            modality TEXT,
            label TEXT,
            score REAL,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def log_prediction(user_id: int, modality: str, label: str, score: float):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO predictions (user_id, modality, label, score, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (user_id, modality, label, float(score),
         datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()


def get_latest_history(limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT id, modality, label, score, created_at "
        "FROM predictions ORDER BY id DESC LIMIT ?",
        conn,
        params=(limit,),
    )
    conn.close()
    return df.sort_values("id", ascending=False)


# ====================== TEXT EMOTION ====================== #

def simple_text_emotion(text: str):
    text_l = text.lower()
    weights = np.ones(len(EMOTIONS), dtype=np.float32) * 0.1

    def boost(label, amount):
        if label in EMOTIONS:
            i = EMOTIONS.index(label)
            weights[i] += amount

    if any(w in text_l for w in ["happy", "glad", "great", "excited", "awesome", "good"]):
        boost("happy", 0.8)
    if any(w in text_l for w in ["sad", "down", "depressed", "cry", "unhappy"]):
        boost("sad", 0.8)
    if any(w in text_l for w in ["angry", "mad", "furious", "annoyed", "pissed"]):
        boost("angry", 0.8)
    if any(w in text_l for w in ["scared", "afraid", "terrified", "nervous", "anxious"]):
        boost("fear", 0.8)
    if any(w in text_l for w in ["disgusted", "gross", "nasty"]):
        boost("disgust", 0.8)
    if any(w in text_l for w in ["shocked", "surprised", "wow", "omg"]):
        boost("surprise", 0.8)
    if any(w in text_l for w in ["ok", "fine", "normal", "meh", "whatever"]):
        boost("neutral", 0.5)

    probs = weights / weights.sum()
    top_idx = int(np.argmax(probs))
    top_label = EMOTIONS[top_idx]
    top_score = float(probs[top_idx])
    return top_label, top_score, probs


# ====================== FACE MODEL & IMAGE / WEBCAM ====================== #

@st.cache_resource
def load_face_model():
    from emotioncv.models.face_cnn import EmotionNet

    model = EmotionNet()
    state = torch.load(CKPT_FACE, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])

    face_cascade = cv2.CascadeClassifier(str(HAAR_PATH))
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade at {HAAR_PATH}")

    return model, transform, face_cascade


def analyze_image(pil_img, conf_threshold=0.0):
    model, transform, face_cascade = load_face_model()

    rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60),
    )

    results = []
    for i, (x, y, w, h) in enumerate(faces, start=1):
        face_img = gray[y:y + h, x:x + w]
        face_pil = Image.fromarray(face_img)
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(face_tensor)
            probs = F.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        label = EMOTIONS[idx.item()] if idx.item() < len(EMOTIONS) else str(idx.item())
        conf_f = float(conf.item())

        label_disp = label if conf_f >= conf_threshold else "uncertain"
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            rgb,
            f"{label_disp} ({conf_f*100:.1f}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        results.append({
            "Face #": i,
            "label": label,
            "score": round(conf_f, 3),
        })

    return rgb, results


# ====================== AUDIO MODEL (no torchaudio) ====================== #

@st.cache_resource
def load_audio_model():
    from emotioncv.models.audio_cnn import AudioEmotionNet

    model = AudioEmotionNet()
    state = torch.load(CKPT_AUDIO, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


def analyze_audio(file_path_or_buffer):
    """
    Load audio with librosa (no torchaudio / torchcodec),
    resample to 16k mono, pad/trim to 3 seconds, run model.
    """
    model = load_audio_model()

    # librosa can handle both paths and file-like objects (UploadedFile)
    y, sr = librosa.load(file_path_or_buffer, sr=16000, mono=True)

    target_len = 16000 * 3  # 3 seconds at 16k
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # shape [1, 1, T]
    waveform = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(waveform)
        probs = F.softmax(logits, dim=1)[0]

    conf, idx = torch.max(probs, dim=0)
    label = EMOTIONS[idx.item()] if idx.item() < len(EMOTIONS) else str(idx.item())
    return label, float(conf.item()), probs.cpu().numpy()


# ====================== EXTERNAL REAL-TIME SCRIPTS ====================== #

def launch_realtime_face():
    cmd = [sys.executable, "-m", "emotioncv.realtime_face_emotion"]
    subprocess.Popen(cmd)


def launch_realtime_voice():
    cmd = [sys.executable, "-m", "emotioncv.realtime_voice_emotion"]
    subprocess.Popen(cmd)


# ====================== STREAMLIT UI (ONE PAGE) ====================== #

def main():
    st.set_page_config(page_title="EmotionCV – Text & Status", layout="wide")
    init_db()

    st.title("EmotionCV – Text & Status")

    # ----- User ID (used for logging) ----- #
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)

    # ===================== 1. TEXT SECTION ===================== #
    st.markdown("## 📝 Text Emotion")

    text = st.text_area("Say something:")

    if st.button("Analyze Text"):
        if not text.strip():
            st.warning("Please type something first.")
        else:
            label, score, probs = simple_text_emotion(text)
            st.markdown(f"**Top:** {label} ({score:.2f})")

            prob_df = pd.DataFrame(
                {"emotion": EMOTIONS, "score": probs}
            ).set_index("emotion")
            st.bar_chart(prob_df)

            log_prediction(user_id=user_id, modality="text",
                           label=label, score=score)

    st.divider()

    # ===================== 2. IMAGE + WEBCAM SECTION ===================== #
    st.markdown("## 📸 Face Emotion – Upload or Webcam")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Upload an image**")
        uploaded = st.file_uploader(
            "Upload a face image (jpg/png)",
            type=["jpg", "jpeg", "png"],
            key="img_uploader",
        )

    with col_right:
        st.markdown("**Use your webcam (snapshot)**")
        cam_img = st.camera_input("Take a photo")

    conf_th = st.slider(
        "Minimum confidence to display label",
        min_value=0.0, max_value=1.0, value=0.3, step=0.05,
        key="img_conf_slider",
    )

    # Helper to process any PIL image
    def handle_image(pil_img, modality_label: str):
        st.image(pil_img, caption=f"{modality_label} image", use_container_width=True)
        try:
            annotated, results = analyze_image(pil_img, conf_threshold=conf_th)
            if not results:
                st.warning("No faces detected.")
            else:
                st.image(annotated, caption="Detected faces", use_container_width=True)
                df = pd.DataFrame(results)
                st.table(df)

                for row in results:
                    log_prediction(
                        user_id=user_id,
                        modality=modality_label,
                        label=row["label"],
                        score=row["score"],
                    )
        except Exception as e:
            st.error(f"Error while processing image: {e}")

    if uploaded is not None:
        pil_img = Image.open(uploaded)
        handle_image(pil_img, "image_upload")

    if cam_img is not None:
        pil_cam = Image.open(cam_img)
        handle_image(pil_cam, "webcam_snapshot")

    # Button to open full real-time webcam window (your OpenCV script)
    st.markdown("#### Or open full real-time webcam")
    if st.button("Open real-time webcam emotion (separate window)"):
        launch_realtime_face()
        st.info("Real-time webcam window started. Press 'q' in that window to close it.")

    st.divider()

    # ===================== 3. VOICE SECTION (UPLOAD + REAL-TIME SCRIPT) ===================== #
    st.markdown("## 🎤 Voice Emotion")

    col_v1, col_v2 = st.columns(2)

    # Option 1: upload wav and classify via audio model
    with col_v1:
        st.markdown("**Upload .wav file**")
        audio_file = st.file_uploader(
            "Upload a WAV file",
            type=["wav"],
            key="audio_uploader",
        )

        if audio_file is not None:
            try:
                label, score, probs = analyze_audio(audio_file)
                st.markdown(f"**Top:** {label} ({score:.2f})")
                prob_df = pd.DataFrame(
                    {"emotion": EMOTIONS, "score": probs}
                ).set_index("emotion")
                st.bar_chart(prob_df)

                log_prediction(user_id=user_id, modality="voice_upload",
                               label=label, score=score)
            except Exception as e:
                st.error(f"Error while processing audio: {e}")

    # Option 2: button to run your existing real-time mic script
    with col_v2:
        st.markdown("**Open full real-time voice analyzer**")
        st.write(
            "This will launch your `realtime_voice_emotion.py` script in a "
            "separate console window. Follow the prompts there and use 'q' to quit."
        )
        if st.button("Open real-time voice emotion (separate window)"):
            launch_realtime_voice()
            st.info("Real-time voice script started in background.")

    st.divider()

    # ===================== 4. HISTORY & DOWNLOAD CSV ===================== #
    st.markdown("## 📊 History (latest 20)")

    history_df = get_latest_history(limit=20)
    if history_df.empty:
        st.info("No history yet. Run an analysis above to see it here.")
    else:
        st.dataframe(history_df, use_container_width=True)

        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download history as CSV",
            data=csv,
            file_name="emotioncv_history_latest20.csv",
            mime="text/csv",
            key="dl_history",
        )


if __name__ == "__main__":
    main()
