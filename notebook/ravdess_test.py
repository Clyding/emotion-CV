import os
import time
import queue
import sounddevice as sd
import numpy as np
import librosa
import noisereduce as nr
import joblib
from tensorflow.keras.models import load_model

# ---------- CONFIG ----------
MODEL_PATH = "/home/clyde/emotion-CV/notebook/ravdess_final_model.h5"   # path to your trained model
LABEL_ENCODER_PATH = "label_encoder.pkl"  # optional (created during training)
SAMPLE_RATE = 22050
DURATION = 3.0      # seconds per chunk (use same duration as training)
N_MFCC = 40
DEVICE = None       # None -> default device. Set index to choose specific device.
THRESHOLD_SILENCE = 0.01  # optional: skip predictions for near-silent clips
# ----------------------------

# Standard RAVDESS emotion mapping used earlier (fallback)
RAVDESS_EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# ----------------------------
# Utilities
# ----------------------------
def load_label_encoder(path):
    if os.path.exists(path):
        le = joblib.load(path)
        print(f"[INFO] Loaded label encoder from {path}")
        return le
    else:
        print("[WARN] label_encoder.pkl not found — using fallback mapping.")
        # create a fake label encoder-like object with classes_ attribute
        class FakeLE:
            def __init__(self, classes):
                self.classes_ = np.array(classes)
            def inverse_transform(self, idx):
                return np.array(self.classes_)[idx]
        # fallback classes order (ensure this matches how you encoded during training)
        fallback = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']
        return FakeLE(fallback)

def load_keras_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = load_model(path)
    print(f"[INFO] Loaded model: {path}")
    return model

def reduce_noise_and_trim(signal, sr):
    # Basic noise reduction - may be heavy; adjust for latency if needed
    reduced = nr.reduce_noise(y=signal, sr=sr)
    trimmed, _ = librosa.effects.trim(reduced, top_db=20)
    return trimmed

def extract_mfcc(signal, sr, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)  # shape (n_mfcc,)
    return mfcc_mean

# ----------------------------
# Real-time recording + prediction
# ----------------------------
def record_audio(duration, sample_rate, device=None):
    """Record `duration` seconds and return a numpy array."""
    print(f"[REC] Recording {duration:.2f}s...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, device=device, dtype='float32')
    sd.wait()
    return recording.flatten()

def predict_from_signal(model, le, signal, sr):
    # Optionally skip near-silent clips
    if np.mean(np.abs(signal)) < THRESHOLD_SILENCE:
        return None, None

    signal = reduce_noise_and_trim(signal, sr)
    # pad/trim to DURATION seconds so MFCC shape consistent
    target_len = int(DURATION * sr)
    if len(signal) < target_len:
        signal = np.pad(signal, (0, target_len - len(signal)))
    elif len(signal) > target_len:
        signal = signal[:target_len]

    feats = extract_mfcc(signal, sr, n_mfcc=N_MFCC)  # (n_mfcc,)
    # match model input: training used np.expand_dims(mfcc, -1)
    x = np.expand_dims(feats, axis=0)            # (1, n_mfcc)
    # If model expects shape (n_mfcc, 1), expand dims
    if len(model.input_shape) == 3 and model.input_shape[-1] == 1:
        x = np.expand_dims(feats, axis=(0,2))    # (1, n_mfcc, 1)
    preds = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    # Map back to label
    try:
        label = le.inverse_transform([top_idx])[0]
    except Exception:
        # fallback: if le.classes_ exist, index into it
        label = str(le.classes_[top_idx])
    return label, confidence

def realtime_loop(model, le, sample_rate=SAMPLE_RATE, duration=DURATION, device=None):
    print("[INFO] Press Ctrl+C to stop.")
    try:
        while True:
            signal = record_audio(duration, sample_rate, device=device)
            label, conf = predict_from_signal(model, le, signal, sample_rate)
            if label is None:
                print("[SKIP] silence or too quiet to classify.")
            else:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{ts}] Predicted: {label} ({conf*100:.1f}%)")
            # small pause to avoid overlap
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print("[INFO] Loading model and label encoder...")
    model = load_keras_model(MODEL_PATH)
    le = load_label_encoder(LABEL_ENCODER_PATH)

    # Print model input shape for debug
    print(f"[DEBUG] Model input shape: {model.input_shape}")

    # Start realtime loop
    realtime_loop(model, le, sample_rate=SAMPLE_RATE, duration=DURATION, device=DEVICE)
