# dashboard.py
import os
import sys
import streamlit as st

# ✅ Step 1: Get current directory (frontend/)
curr_dir = os.path.dirname(os.path.abspath(__file__))

# ✅ Step 2: Go up one level (to emotioncv/)
root_dir = os.path.dirname(curr_dir)

# ✅ Step 3: Add scripts/ to sys.path
scripts_dir = os.path.join(root_dir, "scripts")
sys.path.append(scripts_dir)

# ✅ Step 4: Now import GPT helper
from gpt import ask_gpt


# Streamlit UI
st.title("EmotionCV")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box for user message
user_input = st.text_input("What's up with you gAnG!?")

if st.button("Send") and user_input.strip():
    # Temporary fake emotion probabilities — replace later with webcam inference
    fake_emotion_probs = {"happy": 0.2, "sad": 0.7, "neutral": 0.1}

    # Call GPT reply function
    reply = ask_gpt(user_input, fake_emotion_probs)

    # Save chat to session history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Assistant", reply))

# Display chat history
for speaker, text in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {text}")
