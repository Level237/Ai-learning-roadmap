from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

# Créer le client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Qu'est-ce que Python en une phrase ?"
)

print(response.text)