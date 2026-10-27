from pydantic import BaseModel,Field,EmailStr

class Passenger(BaseModel):
    age:int
    fare:float
    sex_encoded:bool
    p_class:bool
    family_size:int