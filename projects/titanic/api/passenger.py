from pydantic import BaseModel,Field,EmailStr

class Passenger(BaseModel):
    age: int = Field(ge=0, le=100, description="Âge entre 0 et 100")
    fare: float = Field(ge=0, le=600, description="Tarif entre 0 et 600 livres")
    p_class: int = Field(ge=1, le=3, description="Classe 1, 2 ou 3")
    sex_encoded: int = Field(ge=0, le=1, description="0=male, 1=female")
    family_size: int = Field(ge=1, le=11, description="Taille de famille entre 1 et 11")
    
    def validate_fare_class(fare: float, p_class: int) -> bool:
        """Vérifie que le tarif est cohérent avec la classe."""
        if p_class == 1 and fare < 25:
            return False  # 1ère classe commence à 25 livres
        if p_class == 2 and (fare < 10 or fare > 75):
            return False  # 2ème classe entre 10 et 75 livres
        if p_class == 3 and fare > 50:
            return False  # 3ème classe rarement au-dessus de 50 livres
        return True
    
class PassengerAnalized(BaseModel):
    age:int
    sex_encoded:bool
    p_class:bool
    fare:float