
import os
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import librosa

# Default emotion labels (can be edited)
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


@dataclass
class FaceConfig:
    root_dir: str
    img_size: int = 48


class FaceEmotionDataset(Dataset):
    def __init__(self, config: FaceConfig, transform=None):
        self.config = config
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        for label_idx, label in enumerate(EMOTION_LABELS):
            emotion_dir = os.path.join(config.root_dir, label)
            if not os.path.isdir(emotion_dir):
                continue
            for fname in os.listdir(emotion_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(emotion_dir, fname), label_idx))

        if not self.samples:
            print(f"[WARN] No images found in {config.root_dir}. "
                  "Make sure your folders match EMOTION_LABELS.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_idx = self.samples[idx]
        img = Image.open(path).convert("L")  # grayscale for simplicity
        if self.transform is not None:
            img = self.transform(img)
        return img, label_idx


@dataclass
class AudioConfig:
    root_dir: str
    sample_rate: int = 16000
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 512
    duration: float = 3.0  # seconds
    mono: bool = True


class AudioEmotionDataset(Dataset):
    def __init__(self, config: AudioConfig):
        self.config = config
        self.samples: List[Tuple[str, int]] = []

        for label_idx, label in enumerate(EMOTION_LABELS):
            emotion_dir = os.path.join(config.root_dir, label)
            if not os.path.isdir(emotion_dir):
                continue
            for fname in os.listdir(emotion_dir):
                if fname.lower().endswith(".wav"):
                    self.samples.append((os.path.join(emotion_dir, fname), label_idx))

        if not self.samples:
            print(f"[WARN] No audio files found in {config.root_dir}. "
                  "Make sure your folders match EMOTION_LABELS.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_idx = self.samples[idx]
        cfg = self.config

        # Load audio
        audio, sr = librosa.load(path, sr=cfg.sample_rate, mono=cfg.mono)

        # Fix length (pad or trim)
        target_len = int(cfg.sample_rate * cfg.duration)
        if len(audio) < target_len:
            pad_width = target_len - len(audio)
            audio = np.pad(audio, (0, pad_width))
        else:
            audio = audio[:target_len]

        # Mel spectrogram
        melspec = librosa.feature.melspectrogram(
            y=audio,
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
        )
        logmelspec = librosa.power_to_db(melspec, ref=np.max)

        # Normalize
        mean = np.mean(logmelspec)
        std = np.std(logmelspec) + 1e-8
        logmelspec = (logmelspec - mean) / std

        # [n_mels, time] -> tensor [1, n_mels, time]
        feat = torch.tensor(logmelspec, dtype=torch.float32).unsqueeze(0)

        return feat, label_idx
