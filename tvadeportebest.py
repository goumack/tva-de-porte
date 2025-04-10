import streamlit as st
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Fonction pour charger et préparer les données
def load_and_prepare_data(file_path):
    # Chargement des données
    df = pd.read_csv(file_path)
    
    # Normalisation des mois
    df["MONTH"] = df["MONTH"].map({
        "janv": "01", "févr": "02", "mars": "03", "avr": "04", "mai": "05", "juin": "06",
        "juil": "07", "août": "08", "sept": "09", "oct": "10", "nov": "11", "déc": "12"
    })
    
    # Création de la colonne Date
    df["Date"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"] + "-01") + pd.offsets.MonthEnd(0)
    
    # Garder seulement les colonnes nécessaires
    df = df[["Date", "YEAR", "TOTAL_TAXE"]].rename(columns={"TOTAL_TAXE": "Impot_Direct"})
    
    # Conversion des valeurs en milliards
    df["Impot_Direct"] /= 1e9  # En milliards
    return df

# Prétraitement des données + mise à l’échelle
def scale_data(df, seq_length):
    scaler = MinMaxScaler()
    year_scaler = MinMaxScaler()
    
    # Normalisation des valeurs
    values = scaler.fit_transform(df["Impot_Direct"].values.reshape(-1, 1))
    
    # Normalisation des années
    years = year_scaler.fit_transform(df["YEAR"].values.reshape(-1, 1))
    
    return values, years, scaler, year_scaler

# Fonction pour prédire via l'API d'inférence distante
def predict_years_remote(scaled_values, scaled_years, scaler, year_scaler, seq_length, api_url, start_year, end_year=None):
    if end_year is None:
        end_year = start_year  # Prédire une seule année

    # Initialiser la séquence de prédiction
    input_seq = np.stack([scaled_values[-seq_length:].flatten(), scaled_years[-seq_length:].flatten()], axis=0).reshape(1, 2, seq_length)

    predictions = []
    
    # Faire la prédiction année par année
    for year in range(start_year, end_year + 1):
        for month in range(12):
            payload = {
                "inputs": [
                    {
                        "name": "input",  # Nom de l'entrée, peut varier selon le modèle
                        "shape": list(input_seq.shape),
                        "datatype": "FP32",
                        "data": input_seq.flatten().tolist()
                    }
                ]
            }
            
            # Effectuer la requête POST vers l'API d'inférence
            response = requests.post(api_url, json=payload)
            if response.status_code != 200:
                raise Exception(f"Erreur d'inférence : {response.status_code} - {response.text}")
            
            result = response.json()
            pred_norm = result["outputs"][0]["data"][0]  # Première valeur prédite
            pred = scaler.inverse_transform([[pred_norm]])[0][0]
            predictions.append((year, pred))
            
            # Mettre à jour la séquence pour la prédiction suivante
            input_seq = np.roll(input_seq, -1, axis=2)
            input_seq[0, 0, -1] = pred_norm
            input_seq[0, 1, -1] = year_scaler.transform([[year]])[0][0]

    months = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
    df_result = pd.DataFrame(predictions, columns=["Année", "Prédiction (Mds)"])
    df_result["Mois"] = months * (end_year - start_year + 1)
    return df_result

# === CONFIGURATION ===
SEQ_LENGTH = 36
API_URL = "https://tvadeportebestdernier-working.apps.origins.heritage.africa/v2/models/tvadeportebestdernier/infer"

# === UTILISATION STREAMLIT ===

# Titre de l'application
st.title("Prédictions mensuelles de la tva de porte")

# Charge et prépare les données
DATA_PATH = "tva_data.csv"  # Indiquer ici le chemin du fichier CSV
df = load_and_prepare_data(DATA_PATH)

# Affichage des données dans Streamlit
#st.subheader("Aperçu des données")
#st.write(df)

# Mise à l'échelle des données
scaled_values, scaled_years, scaler, year_scaler = scale_data(df, SEQ_LENGTH)

# Choisir l'année à prédire
year_to_predict = st.number_input("Choisir l'année de début", min_value=2024, max_value=2030, value=2024)
end_year = st.number_input("Choisir l'année de fin", min_value=year_to_predict, max_value=2030, value=2025)

# Appeler la fonction pour effectuer les prédictions
df_pred = predict_years_remote(
    scaled_values, scaled_years, scaler, year_scaler,
    seq_length=SEQ_LENGTH,
    api_url=API_URL,
    start_year=year_to_predict,
    end_year=end_year
)

# Affichage du DataFrame des prédictions
st.subheader("Prédictions")
st.write(df_pred)

# Affichage du graphique
st.subheader("Graphique des prévisions")
pivot_df = df_pred.pivot(index="Mois", columns="Année", values="Prédiction (Mds)")
fig, ax = plt.subplots(figsize=(12, 6))
pivot_df.plot(kind="bar", ax=ax)
ax.set_title(f"Prévisions mensuelles de la TVA de Porte ({year_to_predict} à {df_pred['Année'].max()})")
ax.set_ylabel("Milliards")
ax.set_xlabel("Mois")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()

st.pyplot(fig)
