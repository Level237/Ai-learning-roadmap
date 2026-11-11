import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

print("=== ANALYSE DES ERREURS DU MODÈLE ===\n")

# 1. Charger le modèle (Adapte le chemin si ton fichier .pkl s'appelle autrement)
try:
    model = joblib.load("../model/titanic_logistic.pkl")
    print("✅ Modèle chargé avec succès.")
except FileNotFoundError:
    print("❌ Erreur: Fichier modèle introuvable. Vérifie le chemin.")
    exit()

# 2. Charger le dataset BRUT (celui avec les NaN et les majuscules)
df = pd.read_csv("../../../data/titanic.csv") 

# 3. Recréer les features exactement comme dans ton notebook
df['Family_size'] = df['SibSp'] + df['Parch'] + 1
df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})

# 4. Nettoyer les NaN (Indispensable pour LogisticRegression)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 5. Séparer X et y (Avec les MAJUSCULES comme dans ton notebook)
X = df[["Age", "Fare", "Sex_encoded", "Pclass", "Family_size"]]
# Note : On utilise df['Survived'] (1D) et non df[['Survived']] (2D) pour éviter les warnings sklearn
y = df['Survived'] 

# 6. Refaire exactement le même split que dans ton notebook (test_size=0.6)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.6, random_state=42
)

# 7. Faire les prédictions
y_pred = model.predict(X_test)

# 8. Identifier les erreurs
# Faux Négatifs : La réalité est 1 (survit) mais le modèle prédit 0 (meurt)
fn_mask = (y_test == 1) & (y_pred == 0)
false_negatives = X_test[fn_mask].copy()

# Faux Positifs : La réalité est 0 (meurt) mais le modèle prédit 1 (survit)
fp_mask = (y_test == 0) & (y_pred == 1)
false_positives = X_test[fp_mask].copy()

# 9. Afficher les résultats
print(f"\n📊 RÉSULTATS SUR {len(y_test)} PASSAGERS (JEU DE TEST) :")
print(f"Nombre de Faux Négatifs (survivants ratés) : {fn_mask.sum()}")
print(f"Nombre de Faux Positifs (morts imaginaires) : {fp_mask.sum()}")

print("\n--- 🔍 ANALYSE DES FAUX NÉGATIFS (Ceux qui ont survécu mais que le modèle a condamnés) ---")
if len(false_negatives) > 0:
    print(f"Âge moyen : {false_negatives['Age'].mean():.1f} ans")
    print(f"Classe moyenne : {false_negatives['Pclass'].mean():.2f}")
    print(f"% de femmes : {(false_negatives['Sex_encoded'].mean() * 100):.1f}%")
else:
    print("Aucun faux négatif.")

print("\n--- 🔍 ANALYSE DES FAUX POSITIFS (Ceux qui sont morts mais que le modèle a sauvés) ---")
if len(false_positives) > 0:
    print(f"Âge moyen : {false_positives['Age'].mean():.1f} ans")
    print(f"Classe moyenne : {false_positives['Pclass'].mean():.2f}")
    print(f"% de femmes : {(false_positives['Sex_encoded'].mean() * 100):.1f}%")
else:
    print("Aucun faux positif.")

print("\n✅ Analyse terminée.")