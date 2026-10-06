import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

df = pd.read_csv("Crop_recommendation.csv")
X, y = df[FEATURES], df["label"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {"n_estimators": [100, 200], "max_depth": [None, 15], "min_samples_split": [2, 5]},
    cv=5, n_jobs=-1,
)
grid.fit(X_tr, y_tr)
model = grid.best_estimator_

pred = model.predict(X_te)
acc = accuracy_score(y_te, pred)
print("Best params:", grid.best_params_)
print("Test accuracy:", round(acc, 4))
print(classification_report(y_te, pred))

joblib.dump(model, "model.pkl")
json.dump({
    "accuracy": acc,
    "best_params": grid.best_params_,
    "features": FEATURES,
    "importances": dict(zip(FEATURES, model.feature_importances_.tolist())),
    "classes": model.classes_.tolist(),
    "confusion_matrix": confusion_matrix(y_te, pred, labels=model.classes_).tolist(),
    "ranges": {f: [float(df[f].min()), float(df[f].max()), float(df[f].median())] for f in FEATURES},
}, open("metrics.json", "w"))
print("Saved model.pkl and metrics.json")
