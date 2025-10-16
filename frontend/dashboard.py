# dashboard.py
import streamlit as st
from gpt_integration import ask_gpt

st.title("EmotionCV")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Type your message:")

if st.button("Send"):
    # for now, fake emotions — later connect to webcam inference
    fake_emotion_probs = {"happy":0.2, "sad":0.7, "neutral":0.1}
    reply = ask_gpt(user_input, fake_emotion_probs)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Assistant", reply))

for speaker, text in st.session_state.chat_history:
    st.markdown(f"**{speaker}**: {text}")
