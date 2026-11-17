from fastapi import FastAPI,HTTPException
from google import genai
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import os
from schema import FAQResponse,QuestionRequest
import json 


app_state = {}

load_dotenv()

@asynccontextmanager

async def lifespan(app:FastAPI):
    print("Chargement de la faq....")
    
    
    try:
        with open("faq_data.json",'r',encoding="utf-8") as f:
            app_state['faq'] = json.load(f)
            print(f"FAQ chargée : {len(app_state['faq'])} questions")
            
            print(f"FAQ chargée : {len(app_state['faq'])} questions")
            
    except FileNotFoundError:
        app_state["faq"] = []
        print("Erreur : Fichier faq_data.json introuvable.")
    
    
    try:
        app_state['gemini_client'] = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        print("Client gemini initialisé")
        
    except Exception as e:
        app_state["gemini_client"] = None
        print(f"Erreur Gemini : {e}")
        
    yield
    
    print("Nettoyage des ressources...")
    
    app_state.clear()
    
        
    
app=FastAPI(lifespan=lifespan)


@app.post("/ask", response_model=FAQResponse)
async def ask_question(question_request: QuestionRequest):
    gemini_client = app_state.get("gemini_client")
    faq = app_state.get("faq")
    
    # Vérifications
    if gemini_client is None:
        raise HTTPException(status_code=503, detail="Client Gemini non disponible")
    
    if not faq:
        raise HTTPException(status_code=503, detail="FAQ non chargée")
    
    # Étape 1 : Recherche simple dans la FAQ (par mots-clés)
    question_lower = question_request.question.lower()
    best_match = None
    best_score = 0
    
    for item in faq:
        # Compter les mots en commun
        item_words = set(item["question"].lower().split())
        question_words = set(question_lower.split())
        score = len(item_words & question_words)
        
        if score > best_score:
            best_score = score
            best_match = item
    
    # Si aucun match trouvé, on donne un contexte vide
    if best_match is None or best_score == 0:
        context = "Aucune information trouvée dans la FAQ."
    else:
        context = f"Question de référence : {best_match['question']}\nRéponse : {best_match['answer']}"
    
    # Étape 2 : Construire le prompt pour Gemini
    prompt = f"""Tu es un assistant FAQ pour une PME.
Voici les informations de référence :
{context}

Question de l'utilisateur : {question_request.question}

Réponds UNIQUEMENT en JSON avec cette structure :
{{
  "question": "la question posée",
  "answer": "ta réponse basée sur le contexte",
  "confidence": 0.85,
  "category": "catégorie de la question",
  "source": "faq_data.json"
}}"""
    
    # Étape 3 : Appeler Gemini avec sortie JSON forcée
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FAQResponse,
                temperature=0.2
            )
        )
        
        return response.parsed
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Gemini : {str(e)}")

