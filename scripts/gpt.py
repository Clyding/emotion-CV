# scripts/gpt.py
import os
from openai import OpenAI

# ✅ Expect API key from environment variable — safer and flexible
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are an empathetic mental health support assistant.
Do not provide medical or professional advice.
Encourage healthy coping strategies, validation, and support.
"""

def ask_gpt(user_text, emotion_probs):
    """
    Generate an empathetic assistant response using emotion probabilities.
    """
    context = "Emotions: " + ", ".join([f"{k}:{v:.2f}" for k, v in emotion_probs.items()])
    prompt = f"{SYSTEM_PROMPT}\n{context}\nUser: {user_text}\nAssistant:"

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=400,
        )

        # Extract text output safely
        output_text = ""
        for item in response.output:
            if hasattr(item, "content"):
                for piece in item.content:
                    if piece.get("type") == "output_text":
                        output_text += piece.get("text", "")

        # Fallback if no response text found
        return output_text.strip() or "I'm here for you — tell me more about how you're feeling."

    except Exception as e:
        return f"(Error: {str(e)})"
