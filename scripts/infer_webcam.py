# infer_webcam.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model


MODEL_PATH = "/home/clyde/emotion-CV/models/fer_mobilenetv2.h5"   
model = load_model(MODEL_PATH)

input_shape = model.input_shape  
channels = input_shape[-1]

labels = ['angry','disgust','fear','happy','sad','surprise','neutral']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
cap = cv2.VideoCapture(0)
print("🎥 Press 'q' to quit webcam window")

def preprocess_face(face_img):
    face = cv2.resize(face_img, (48,48))
    if channels == 1:  
        face = face.reshape(1,48,48,1).astype('float32') / 255.0
    else:
        # MobileNetV2 → convert to RGB (3 channels)
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        face = face.reshape(1,48,48,3).astype('float32') / 255.0
    return face

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face_input = preprocess_face(face)

        probs = model.predict(face_input, verbose=0)[0]
        top_idx = np.argmax(probs)
        label = labels[top_idx]
        conf = probs[top_idx]

        cv2.putText(frame, f"{label} {conf*100:.1f}%", (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("EmotionCV Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
