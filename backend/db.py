# backend/app.py
from fastapi import FastAPI, UploadFile, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ml_pipeline import EmotionVisionModel, EmotionVoiceModel
from db import init_db, SessionLocal, log_analysis, create_chat_session, add_chat_message
from sqlalchemy.orm import Session
import shutil
import cv2
import numpy as np
from openai import OpenAI
import os
import json

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

# Initialize OpenAI client properly
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
async def analyze_face(file: UploadFile, db: Session = Depends(get_db)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        file_bytes = np.frombuffer(await file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Analyze emotion
        results = vision_model.predict(frame)
        
        # Log to database
        if results:
            primary_result = results[0]  # Take the first face detected
            log_analysis(
                db=db,
                analysis_type="face",
                results=results,
                input_data=file.filename,
                emotion_detected=primary_result["emotion"],
                confidence=primary_result["confidence"]
            )
        
        return {"results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/analyze_voice/")
async def analyze_voice(file: UploadFile, db: Session = Depends(get_db)):
    try:
        # Validate file type
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Save temporary file
        temp_path = "temp_audio.wav"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Analyze emotion
        results = voice_model.predict(temp_path)
        
        # Log to database
        log_analysis(
            db=db,
            analysis_type="voice",
            results=results,
            input_data=file.filename,
            emotion_detected=results["emotion"],
            confidence=results["confidence"]
        )
        
        # Clean up
        os.remove(temp_path)
        
        return {"results": results}
    
    except Exception as e:
        # Clean up temp file if it exists
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/chat/")
async def chat(
    message: str = Form(...), 
    emotion: str = Form("neutral"),
    db: Session = Depends(get_db)
):
    try:
        # Create empathetic prompt
        prompt = f"You are an empathetic AI. The user feels {emotion}. Reply kindly and supportively.\nUser: {message}\nAssistant:"
        
        # Get AI response
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=300
        )
        
        # Extract reply
        reply = ""
        for item in resp.output:
            if hasattr(item, "content"):
                for piece in item.content:
                    if piece.get("type") == "output_text":
                        reply += piece.get("text", "")
        
        # Log chat interaction
        log_analysis(
            db=db,
            analysis_type="chat",
            results={"user_message": message, "ai_response": reply, "emotion_context": emotion},
            input_data=message,
            emotion_detected=emotion
        )
        
        return {"response": reply}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

# Additional endpoints for database queries
@app.get("/logs/")
async def get_logs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    from db import Log
    logs = db.query(Log).order_by(Log.created_at.desc()).offset(skip).limit(limit).all()
    return {"logs": logs}

@app.get("/logs/face/")
async def get_face_logs(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    from db import Log
    logs = db.query(Log).filter(Log.analysis_type == "face").order_by(Log.created_at.desc()).offset(skip).limit(limit).all()
    return {"logs": logs}

@app.get("/stats/emotions/")
async def get_emotion_stats(db: Session = Depends(get_db)):
    from db import Log
    from sqlalchemy import func
    
    stats = db.query(
        Log.emotion_detected, 
        func.count(Log.id).label('count')
    ).filter(
        Log.emotion_detected.isnot(None)
    ).group_by(
        Log.emotion_detected
    ).all()
    
    return {"emotion_stats": [{"emotion": stat.emotion_detected, "count": stat.count} for stat in stats]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
