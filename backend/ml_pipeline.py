# backend/ml_pipeline.py
import cv2
import numpy as np
import torch
import librosa
from tensorflow.keras.models import load_model

class EmotionVisionModel:
    def __init__(self, model_path=""):
        self.model = load_model(model_path)
        self.labels = ['angry','disgust','fear','happy','sad','surprise','neutral']
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.input_shape = self.model.input_shape[-1]

    def preprocess_face(self, face):
        face = cv2.resize(face, (48,48))
        if self.input_shape == 1:
            face = face.reshape(1,48,48,1).astype('float32') / 255.0
        else:
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
            face = face.reshape(1,48,48,3).astype('float32') / 255.0
        return face

    def predict(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        results = []

        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face_input = self.preprocess_face(face)
            probs = self.model.predict(face_input, verbose=0)[0]
            top_idx = np.argmax(probs)
            results.append({
                "emotion": self.labels[top_idx],
                "confidence": float(probs[top_idx]),
                "probs": {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
            })
        return results


class EmotionVoiceModel:
    def __init__(self, model_path="", device="cpu"):
        self.device = device
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()
        self.labels = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']

    def extract_features(self, file_path):
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(self.device)

    def predict(self, file_path):
        x = self.extract_features(file_path)
        with torch.no_grad():
            out = self.model(x)
            probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
        top_idx = np.argmax(probs)
        return {
            "emotion": self.labels[top_idx],
            "confidence": float(probs[top_idx]),
            "probs": {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
        }
