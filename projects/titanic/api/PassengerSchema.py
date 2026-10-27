from pydantic import BaseModel

class PassengerSchema(BaseModel):
    prediction : int
    probability : float
    message:str