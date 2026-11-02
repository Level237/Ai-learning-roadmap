from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

from pydantic import BaseModel

class TitanicAnalysis(BaseModel):
    survival_probability: str  # "high", "medium", "low"
    reason: str
    risk_factors: list[str]
    protective_factors: list[str]

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="""Analyse ce passager fictif du Titanic :
- Âge: 25 ans
- Sexe: femme
- Classe: 1ère classe
- Tarif payé: 150 livres""",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=TitanicAnalysis
    )
)

# La réponse est déjà parsée !
data = response.parsed
print(f"Probabilité de survie : {data.survival_probability}")
print(f"Raison : {data.reason}")
print(f"Facteurs de risque : {data.risk_factors}")