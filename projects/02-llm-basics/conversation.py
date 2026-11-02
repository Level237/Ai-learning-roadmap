from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction="Tu es un expert du Titanic. Réponds en 2-3 phrases maximum."
    )
)

questions = [
    "Combien de passagers y avait-il sur le Titanic ?",
    "Pourquoi a-t-il coulé ?",
    "Qui était le capitaine ?"
]

for question in questions:
    print(f"\nQuestion : {question}")
    response = chat.send_message(question)
    print(f"Réponse : {response.text}")