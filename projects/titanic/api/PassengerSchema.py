from pydantic import BaseModel, Field

class PassengerSchema(BaseModel):
    prediction : int
    probability : float
    message:str
    
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