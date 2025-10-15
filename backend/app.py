# backend/app.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ml_pipeline import EmotionVisionModel, EmotionVoiceModel
from db import init_db, SessionLocal, User, Log
import shutil
import cv2
import numpy as np
from openai import OpenAI
import os

# Initialize
app = FastAPI(title="EmotionCV API")
init_db()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
vision_model = EmotionVisionModel()
voice_model = EmotionVoiceModel()

client = OpenAI(api_key=os.getenv("lol it's free, get yours"))

# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =============== ROUTES ===================

@app.get("/")
def root():
    return {"message": "EmotionCV backend is running!"}

@app.post("/analyze_face/")
async def analyze_face(file: UploadFile):
    file_bytes = np.frombuffer(await file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    results = vision_model.predict(frame)
    return {"results": results}

@app.post("/analyze_voice/")
async def analyze_voice(file: UploadFile):
    with open("temp_audio.wav", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    results = voice_model.predict("temp_audio.wav")
    os.remove("temp_audio.wav")
    return {"results": results}

@app.post("/chat/")
async def chat(message: str = Form(...), emotion: str = Form("neutral")):
    prompt = f"You are an empathetic AI. The user feels {emotion}. Reply kindly.\nUser: {message}\nAssistant:"
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        max_output_tokens=300
    )
    reply = ""
    for item in resp.output:
        if hasattr(item, "content"):
            for piece in item.content:
                if piece.get("type") == "output_text":
                    reply += piece.get("text", "")
    return {"response": reply}
