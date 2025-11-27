
from pathlib import Path

import numpy as np
import sounddevice as sd
import librosa
import torch

from emotioncv.models.audio_cnn import AudioEmotionNet
from emotioncv.data.datasets import EMOTION_LABELS


SAMPLE_RATE = 16000
DURATION = 3.0
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512


def load_model(weights_path: Path, device):
    model = AudioEmotionNet()
    state = torch.load(weights_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def record_clip():
    num_samples = int(DURATION * SAMPLE_RATE)
    print(f"[INFO] Recording {DURATION} seconds. Speak now...")
    audio = sd.rec(num_samples, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze()


def audio_to_logmel(audio: np.ndarray):
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    melspec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    logmelspec = librosa.power_to_db(melspec, ref=np.max)

    mean = np.mean(logmelspec)
    std = np.std(logmelspec) + 1e-8
    logmelspec = (logmelspec - mean) / std

    return logmelspec


def preprocess_audio(audio: np.ndarray, device):
    logmel = audio_to_logmel(audio)
    tensor = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


def loop_voice_emotion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parent.parent
    ckpt_path = project_root / "checkpoints" / "audio_emotion_best.pth"

    if not ckpt_path.exists():
        print(f"[ERROR] Missing checkpoint: {ckpt_path}. Train the audio model first.")
        return

    model = load_model(ckpt_path, device)
    print("[INFO] Model loaded.")
    print("[INFO] Press Enter to record, or 'q' + Enter to quit.")

    while True:
        cmd = input("\n[INPUT] Enter to record, 'q' to quit: ").strip().lower()
        if cmd == "q":
            break

        audio = record_clip()
        try:
            x = preprocess_audio(audio, device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                conf, idx = torch.max(probs, dim=1)
            label = EMOTION_LABELS[idx.item()]
            confidence = conf.item()
            print(f"[RESULT] Emotion: {label} ({confidence*100:.1f}%)")
        except Exception as e:
            print("[ERROR] Prediction failed:", e)


if __name__ == "__main__":
    loop_voice_emotion()
