import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# 1. Charger le fichier brut
df = pd.read_csv('../../data/titanic.csv')

# 2. Créer les features
df['Family_size'] = df['SibSp'] + df['Parch'] + 1
df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})
df['Age'] = df['Age'].fillna(df['Age'].median())
df['is_child'] = (df['Age'] < 12).astype(int)

# 3. Séparer X et y (AVEC is_child, en MAJUSCULES comme le notebook)
X = df[["Age", "Fare", "Sex_encoded", "Pclass", "Family_size", "is_child"]]
y = df["Survived"]

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.6, random_state=42
)

for depth in [3, 4, 5]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(tree, X, y, cv=5, scoring='accuracy')
    print(f"max_depth={depth}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# 5. Réentraîner le modèle AVEC is_child
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 6. Tester l'enfant (mêmes noms de colonnes en MAJUSCULES)
test_child = pd.DataFrame([{
    "Age": 8, "Fare": 29.0, "Sex_encoded": 0,
    "Pclass": 2, "Family_size": 3, "is_child": 1
}])

print("Probabilité de survie de l'enfant :")
print(model.predict_proba(test_child))

# 7. Sauvegarder le NOUVEAU modèle
joblib.dump(model, "model/titanic_tree_v2.pkl")
print("✅ Modèle v2 sauvegardé")