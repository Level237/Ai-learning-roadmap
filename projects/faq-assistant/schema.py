from pydantic import BaseModel, Field

class FAQResponse(BaseModel):
    question: str = Field(description="La question posée par l'utilisateur")
    answer: str = Field(description="La réponse trouvée dans la FAQ")
    confidence: float = Field(ge=0, le=1, description="Score de confiance entre 0 et 1")
    category: str = Field(description="Catégorie de la question")
    source: str = Field(description="Source de la réponse")
    
class QuestionRequest(BaseModel):
    question: str = Field(description="La question posée par l'utilisateur")