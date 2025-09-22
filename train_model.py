# train_model.py
# Ce script sert à préparer les données, entraîner un modèle XGBoost pour prédire le churn
# et sauvegarder le modèle + scaler + imputer pour usage ultérieur dans Streamlit.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

# Charger le dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Nettoyage des données
df = df.replace(" ", np.nan)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df = df.drop("customerID", axis=1)

# Encodage des variables catégorielles
df_encoded = pd.get_dummies(df, drop_first=True)

# Séparation features/cible
X = df_encoded.drop("Churn_Yes", axis=1)
y = df_encoded["Churn_Yes"]

# Split en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Imputer les valeurs manquantes
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sauvegarder les noms de colonnes pour l'app Streamlit
model_columns = X_train.columns.tolist()
joblib.dump(model_columns, "model_columns.pkl")

# Entraîner XGBoost
xgb_model = XGBClassifier(eval_metric="logloss", random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Évaluation du modèle
y_pred_xgb = xgb_model.predict(X_test_scaled)
print("XGBoost")
print(classification_report(y_test, y_pred_xgb))

# Sauvegarder le modèle et les outils de preprocessing
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")
