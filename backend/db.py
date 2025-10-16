# backend/db.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Database URL - you can change this to PostgreSQL/MySQL later
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./emotioncv.db")

# Create engine
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class
Base = declarative_base()

# Define User model
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"

# Define Log model
class Log(Base):
    __tablename__ = "logs"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_type = Column(String(20), nullable=False)  # 'face', 'voice', or 'chat'
    user_id = Column(Integer, nullable=True)  # Optional: link to user if authenticated
    input_data = Column(Text, nullable=True)  # Original input (filename, message, etc.)
    results = Column(Text, nullable=False)  # JSON string of analysis results
    confidence = Column(Float, nullable=True)  # Top confidence score if applicable
    emotion_detected = Column(String(20), nullable=True)  # Primary emotion detected
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Log(id={self.id}, type='{self.analysis_type}', emotion='{self.emotion_detected}')>"

# Define Session model for chat conversations
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)
    initial_emotion = Column(String(20), nullable=True)  # Emotion when session started
    created_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)

# Define ChatMessage model
class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, nullable=False)
    message_type = Column(String(10), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    emotion_context = Column(String(20), nullable=True)  # Emotion context for this message
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    """
    Initialize the database - create all tables
    """
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

def get_db():
    """
    Database dependency for FastAPI routes
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper functions for common operations
def log_analysis(db, analysis_type, results, user_id=None, input_data=None, emotion_detected=None, confidence=None):
    """
    Helper function to log analysis results
    """
    log_entry = Log(
        analysis_type=analysis_type,
        user_id=user_id,
        input_data=input_data,
        results=str(results),
        emotion_detected=emotion_detected,
        confidence=confidence
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
    return log_entry

def create_chat_session(db, user_id=None, initial_emotion=None):
    """
    Create a new chat session
    """
    session = ChatSession(
        user_id=user_id,
        initial_emotion=initial_emotion
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

def add_chat_message(db, session_id, message_type, content, emotion_context=None):
    """
    Add a message to a chat session
    """
    message = ChatMessage(
        session_id=session_id,
        message_type=message_type,
        content=content,
        emotion_context=emotion_context
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message
