import pandas as pd
import joblib

# 1. Charger le modèle v2 (avec is_child)
model = joblib.load("../model/titanic_tree_v2.pkl")

# 2. Définir les 10 cas de test fixes
# Format : (description, features, réponse attendue)
# Réponse attendue : 1 = survit, 0 = meurt
test_cases = [
    {
        "name": "Enfant 8 ans, 2e classe",
        "features": {"Age": 8, "Fare": 29.0, "Sex_encoded": 0, "Pclass": 2, "Family_size": 3},
        "expected": 1  # Enfant prioritaire
    },
    {
        "name": "Femme 1re classe, 25 ans",
        "features": {"Age": 25, "Fare": 150.0, "Sex_encoded": 1, "Pclass": 1, "Family_size": 2},
        "expected": 1  # Femme + 1re classe
    },
    {
        "name": "Homme 3e classe, 35 ans",
        "features": {"Age": 35, "Fare": 7.5, "Sex_encoded": 0, "Pclass": 3, "Family_size": 1},
        "expected": 0  # Homme + 3e classe
    },
    {
        "name": "Enfant 5 ans, 3e classe",
        "features": {"Age": 5, "Fare": 15.0, "Sex_encoded": 1, "Pclass": 3, "Family_size": 4},
        "expected": 1  # Enfant, même en 3e classe
    },
    {
        "name": "Personne âgée 70 ans, 1re classe",
        "features": {"Age": 70, "Fare": 200.0, "Sex_encoded": 0, "Pclass": 1, "Family_size": 1},
        "expected": 0  # Homme âgé, même en 1re classe
    },
    {
        "name": "Femme 3e classe, 30 ans",
        "features": {"Age": 30, "Fare": 8.0, "Sex_encoded": 1, "Pclass": 3, "Family_size": 5},
        "expected": 0  # Femme mais 3e classe + grande famille
    },
    {
        "name": "Homme 1re classe, 40 ans",
        "features": {"Age": 40, "Fare": 100.0, "Sex_encoded": 0, "Pclass": 1, "Family_size": 1},
        "expected": 0  # Homme, même en 1re classe
    },
    {
        "name": "Femme 2e classe, 20 ans",
        "features": {"Age": 20, "Fare": 30.0, "Sex_encoded": 1, "Pclass": 2, "Family_size": 1},
        "expected": 1  # Femme + 2e classe
    },
    {
        "name": "Enfant 10 ans, 1re classe",
        "features": {"Age": 10, "Fare": 120.0, "Sex_encoded": 1, "Pclass": 1, "Family_size": 2},
        "expected": 1  # Enfant + 1re classe
    },
    {
        "name": "Homme 3e classe, 50 ans, famille nombreuse",
        "features": {"Age": 50, "Fare": 20.0, "Sex_encoded": 0, "Pclass": 3, "Family_size": 8},
        "expected": 0  # Homme + 3e classe + famille nombreuse
    }
]

# 3. Lancer les tests
score = 0
results = []

for case in test_cases:
    # Ajouter is_child
    features = case["features"].copy()
    features["is_child"] = 1 if features["Age"] < 12 else 0
    
    # Créer le DataFrame avec les bonnes colonnes
    input_df = pd.DataFrame([features])
    
    # Prédire
    prediction = int(model.predict(input_df)[0])
    
    # Comparer
    is_correct = prediction == case["expected"]
    if is_correct:
        score += 1
    
    status = "✅" if is_correct else "❌"
    results.append({
        "name": case["name"],
        "expected": case["expected"],
        "predicted": prediction,
        "status": status
    })

# 4. Afficher les résultats
print("=" * 60)
print("RÉSULTATS DU JEU DE TEST DE RÉGRESSION")
print("=" * 60)

for r in results:
    expected_label = "Survit" if r["expected"] == 1 else "Meurt"
    predicted_label = "Survit" if r["predicted"] == 1 else "Meurt"
    print(f"{r['status']} {r['name']}")
    print(f"   Attendu: {expected_label} | Prédit: {predicted_label}")
    print()

print("=" * 60)
print(f"SCORE FINAL : {score}/10")
print("=" * 60)