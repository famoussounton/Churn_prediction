# 📊 Prédiction du churn client

Ce projet a pour but de prédire le **churn** (le départ des clients) dans le secteur télécom à partir d’un dataset public.  
L’idée est d’entraîner un modèle de machine learning puis de le rendre accessible via une petite application web avec **Streamlit**.  

---

## 🚀 Ce que contient le projet

- **`train_model.py`** → script qui nettoie les données, entraîne le modèle et sauvegarde tout le nécessaire (modèle, scaler, imputer, colonnes).  
- **`app.py`** → une app Streamlit simple pour tester des prédictions en direct.  
- **`notebook.ipynb`** → quelques explorations et visualisations pour mieux comprendre les données.  
- **`.pkl`** → fichiers sauvegardés (modèle et preprocessing) pour réutiliser l’entraînement sans tout relancer.  
- **dataset** → `WA_Fn-UseC_-Telco-Customer-Churn.csv` (fourni par IBM).  

---

## Installation

1. **Cloner le repo** :
git clone https://github.com/famoussounton/Churn_prediction.git
cd churn-prediction

2. **Créer un environnement virtuel et l’activer** :
python3 -m venv venv
source venv/bin/activate   # sur Mac/Linux
venv\Scripts\activate      # sur Windows

3. **Installer les dépendances** :
pip install -r requirements.txt

## Utilisation

1. **Réentraîner le modèle** :
python train_model.py

2. **Lancer l’app Streamlit** :
streamlit run app.py

Une page web s’ouvrira automatiquement (ou dispo sur http://localhost:8501).

---


## Fonctionnement de l’app

Tu renseignes la durée d’abonnement et la facturation mensuelle).
Le modèle applique le même preprocessing que lors de l’entraînement (imputation + scaling).
Résultat : l’app te dit si le client a de fortes chances de partir (churn) ou de rester (pas de churn).
<img width="1467" height="615" alt="Screenshot 2025-09-22 at 21 40 32" src="https://github.com/user-attachments/assets/3deeb6fe-bd56-4607-aaa6-4ed0aa9c774b" />
<img width="1464" height="757" alt="Screenshot 2025-09-22 at 21 40 18" src="https://github.com/user-attachments/assets/65155f41-8e77-445c-b53b-f968d322fb9b" />
