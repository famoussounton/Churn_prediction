# app.py
# Application Streamlit pour prédire le churn client.
# L'utilisateur renseigne quelques informations et obtient une prédiction.

import streamlit as st
import pandas as pd
import joblib

# Charger modèle, scaler, imputer et colonnes utilisées lors de l'entraînement
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
model_columns = joblib.load("model_columns.pkl")

# Titre de l'app
st.title("Prédiction du churn client")

# Inputs utilisateur
tenure = st.number_input("Durée d'abonnement (mois)", min_value=0)
monthly_charges = st.number_input("Facturation mensuelle", min_value=0.0)

# Bouton de prédiction
if st.button("Prédire"):
    # Créer un DataFrame avec toutes les colonnes attendues par le modèle
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Remplir les colonnes renseignées par l'utilisateur
    input_df.loc[0, "tenure"] = tenure
    input_df.loc[0, "MonthlyCharges"] = monthly_charges
    
    # Appliquer l'imputer pour combler les valeurs manquantes
    input_df = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)
    
    # Standardiser
    input_scaled = scaler.transform(input_df)
    
    # Prédiction
    prediction = model.predict(input_scaled)
    
    # Affichage résultat
    st.write("Churn prévu" if prediction[0]==1 else "Pas de churn prévu")
