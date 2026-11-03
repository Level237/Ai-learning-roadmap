from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import json


load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))



class PassengerAnalysis(BaseModel):
    survival_probability: str = Field(
        description="high, medium, ou low",
        pattern="^(high|medium|low)$"
    )
    confidence_score: float = Field(
        description="Score de confiance entre 0 et 1",
        ge=0,
        le=1
    )
    reasoning: str = Field(
        description="Explication détaillée en 2-3 phrases"
    )
    risk_factors: list[str] = Field(
        description="Liste des facteurs de risque"
    )
    protective_factors: list[str] = Field(
        description="Liste des facteurs protecteurs"
    )
    historical_context: str = Field(
        description="Contexte historique pertinent"
    )