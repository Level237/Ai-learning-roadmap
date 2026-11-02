from google import genai
from dotenv import load_dotenv
import os
from google.genai import types

load_dotenv()

# Créer le client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


for temp in [0.0, 0.7, 1]:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Explique le machine learning en 2 phrases",
        config=types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=100
        )
    )
    print(f"\n=== Temperature {temp} ===")
    print(response.text)