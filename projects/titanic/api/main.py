import os
import joblib
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from dotenv import load_dotenv

from passenger import Passenger, PassengerAnalized
from PassengerSchema import PassengerAnalysis, PassengerSchema

app_state = {}

load_dotenv()
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Chargement du modèle ML...")
    try:
        app_state['model'] = joblib.load("../model/titanic_logistic.pkl")
        print("Modèle chargé avec succès.")
    except FileNotFoundError:
        app_state["model"] = None
        print("Erreur : Fichier model/titanic_logistic.pkl introuvable.")
    except Exception as e:
        app_state["model"] = None
        print(f"Erreur lors du chargement : {e}")
    
    print("Initialisation du client Gemini...")
    try:
        app_state['gemini_client'] = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        print("Client Gemini initialisé.")
    except Exception as e:
        app_state["gemini_client"] = None
        print(f"Erreur Gemini : {e}")
    
    yield

    print("Nettoyage des ressources...")
    # Au lieu d'un app_state.clear() brutal qui casse le GC de httpx/genai
    client = app_state.pop("gemini_client", None)
    if client is not None:
        try:
            # Réinitialisation propre sans déclencher de crash sur l'attribut manquant
            del client
        except Exception:
            pass
    app_state.pop("model", None)


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict_passenger(passenger: Passenger):
    model = app_state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    # Attention à l'ordre des colonnes : Age, Fare, Sex_encoded, Pclass, Family_size
    passenger_data = [
        passenger.age,
        passenger.fare,
        passenger.sex_encoded,
        passenger.p_class,
        passenger.family_size
    ]
    
    passenger_array = np.array([passenger_data]).reshape(1, -1)
    
    prediction = model.predict(passenger_array)
    prediction_value = int(prediction[0])
    probability = float(model.predict_proba(passenger_array)[0, 1])
    
    if probability < 0.5:
        message = f"Le passager a {(1 - probability) * 100:.1f} % de chances de ne pas survivre"
    else:
        message = f"Le passager a {probability * 100:.1f} % de chances de survivre"
      
    return {
        "probabilité": probability,
        "message": message,
        "prediction": prediction_value
    }


@app.post("/analyze-passenger")
async def analyze_passenger(passenger: Passenger):
    model = app_state.get("model")
    gemini_client = app_state.get("gemini_client")
   
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle ML non disponible")
    if gemini_client is None:
        raise HTTPException(status_code=503, detail="Client Gemini non disponible")
    
    # Correction de l'ordre pour correspondre exactement à celui de /predict
    passenger_data = [
        passenger.age,
        passenger.fare,
        passenger.sex_encoded,  # Inversé avec p_class dans votre version initiale
        passenger.p_class,
        passenger.family_size
    ]
   
    passenger_array = np.array([passenger_data]).reshape(1, -1)

    # 1. Prédiction ML
    ml_probability = float(model.predict_proba(passenger_array)[0, 1])
    ml_prediction = int(ml_probability >= 0.5)
   
    # 2. Analyse LLM
    sex_label = "female" if passenger.sex_encoded == 1 else "male"
    
    
   
    prompt = f"""Analyse ce passager du Titanic :
- Âge: {passenger.age} ans
- Sexe: {sex_label}
- Classe: {passenger.p_class} (IMPORTANT: utilise cette valeur, pas le tarif pour déduire la classe)
- Tarif payé: {passenger.fare} livres
- Taille de la famille: {passenger.family_size}

Évalue sa probabilité de survie en te basant sur les facteurs historiques connus."""
   
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PassengerAnalysis,
                temperature=0.3
            )
        )
        llm_analysis = response.parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Gemini: {str(e)}")
     
     
    
    # 3. Combiner les résultats
    combined_insight = f"Le modèle ML prédit {ml_probability:.1%} de survie. "
    
    ml_high = ml_probability > 0.5
    llm_high = llm_analysis.survival_probability == "high"
    llm_low = llm_analysis.survival_probability == "low"
    
    divergence = (ml_high and llm_low) or (not ml_high and llm_high)
    if hasattr(llm_analysis, 'survival_probability'):
        prob_str = str(llm_analysis.survival_probability).lower()
        if prob_str == "high" and ml_prediction == 1:
            combined_insight += "L'analyse contextuelle confirme une haute probabilité de survie."
        elif prob_str == "low" and ml_prediction == 0:
            combined_insight += "L'analyse contextuelle confirme une faible probabilité de survie."
        else:
            combined_insight += "Il y a une divergence entre le modèle ML et l'analyse contextuelle."
    
    # 4. Retourner la réponse combinée
    return {
        "ml_prediction": {
            "probability": ml_probability,
            "survived": ml_prediction
        },
        "llm_analysis": llm_analysis,
        "combined_insight": combined_insight,
        "divergence_detected": divergence,
      "recommendation": "Vérifier la cohérence des données d'entrée" if divergence else "Prédictions concordantes"
    }