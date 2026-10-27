from fastapi import FastAPI,HTTPException
from passenger import Passenger
from PassengerSchema import PassengerSchema
import numpy as np
import joblib
from contextlib import asynccontextmanager

app_state = {}

@asynccontextmanager
async def lifespan(app:FastAPI):
   print("Chargement du model")
   
   try:
      
      app_state['model'] = joblib.load("../model/titanic_logistic.pkl")
      print("Model chargé avant succès")
      
   except FileNotFoundError:
      app_state["model"] = None
      print("Erreur : Fichier model/titanic_logistic.pkl introuvable.")
   except Exception as e:
        app_state["model"] = None
        print(f"Erreur lors du chargement : {e}")
    
   yield
   
   print("Nettoyage des ressources...")
   app_state.clear()
      
app = FastAPI(lifespan=lifespan)

@app.get("/")

async def root():
   return {"message": "Hello World"}

@app.post("/predict")

async def predict_passenger(passenger:Passenger):
   model = app_state.get("model")
    
   passenger_data =[
       passenger.age,
       passenger.fare,
       passenger.sex_encoded,
       passenger.p_class,
       passenger.family_size
   ]
    
   passenger_array = np.array([passenger_data]).reshape(1, -1)
   print(passenger_array)
   if model is None:
      raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    # Logique de prédiction ici
   prediction = model.predict(passenger_array)
   prediction_value =int(prediction[0])
   probability = float(model.predict_proba(passenger_array)[0, 1])
    
   if probability < 0.5:
      message = f"Le passager a {probability * 100 } % de ne pas survivre"
   else:
      message = f"Le passager a {probability * 100 } % de  survivre"
      
   return {"probabilité": probability,
           "message" : message,
           "prediction":prediction_value
           }