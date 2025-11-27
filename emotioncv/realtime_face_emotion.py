import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
import numpy as np
from PIL import Image

# ---------- PATHS ---------- #

BASE_DIR = Path(__file__).resolve().parent.parent  # EmotionCV2_Project
CKPT_FACE = BASE_DIR / "checkpoints" / "face_emotion_best.pth"
HAAR_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from emotioncv.data.datasets import EMOTION_LABELS
    EMOTIONS = EMOTION_LABELS
except Exception:
    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def load_model():
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
        raise RuntimeError(f"Failed to load Haar cascade from {HAAR_PATH}")

    return model, transform, face_cascade


def run_camera():
    print("[INFO] Using device:", device)
    print("[INFO] Press 'q' to quit.")

    model, transform, face_cascade = load_model()

    # Try DirectShow backend (more stable on Windows)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam with CAP_DSHOW. "
              "Is another app using the camera?")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Could not read frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(60, 60),
        )

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            face_pil = Image.fromarray(face_img)

            face_tensor = transform(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(face_tensor)
                probs = F.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)

            label = EMOTIONS[idx.item()] if idx.item() < len(EMOTIONS) else str(idx.item())
            conf_f = float(conf.item())

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} ({conf_f*100:.1f}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Real-Time Face Emotion", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera()

