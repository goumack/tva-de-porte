import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib as plt

# === MinMaxScaler maison ===
def manual_minmax_scale(arr):
    min_val = arr.min()
    max_val = arr.max()
    scaled = (arr - min_val) / (max_val - min_val)
    return scaled, min_val, max_val

def manual_inverse_scale(scaled_val, min_val, max_val):
    return scaled_val * (max_val - min_val) + min_val

# Fonction pour charger et préparer les données
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    df["MONTH"] = df["MONTH"].map({
        "janv": "01", "févr": "02", "mars": "03", "avr": "04", "mai": "05", "juin": "06",
        "juil": "07", "août": "08", "sept": "09", "oct": "10", "nov": "11", "déc": "12"
    })
    
    df["Date"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"] + "-01") + pd.offsets.MonthEnd(0)
    df = df[["Date", "YEAR", "TOTAL_TAXE"]].rename(columns={"TOTAL_TAXE": "Impot_Direct"})
    df["Impot_Direct"] /= 1e9  # en milliards
    return df

# Prétraitement des données
def scale_data(df, seq_length):
    values = df["Impot_Direct"].values
    years = df["YEAR"].values

    values_scaled, val_min, val_max = manual_minmax_scale(values)
    years_scaled, year_min, year_max = manual_minmax_scale(years)

    return (
        values_scaled.reshape(-1, 1),
        years_scaled.reshape(-1, 1),
        (val_min, val_max),
        (year_min, year_max)
    )

# Prédiction via API
def predict_years_remote(scaled_values, scaled_years, scaler, year_scaler, seq_length, api_url, start_year, end_year=None):
    if end_year is None:
        end_year = start_year

    input_seq = np.stack([scaled_values[-seq_length:].flatten(), scaled_years[-seq_length:].flatten()], axis=0).reshape(1, 2, seq_length)
    predictions = []

    for year in range(start_year, end_year + 1):
        for month in range(12):
            payload = {
                "inputs": [
                    {
                        "name": "input",
                        "shape": list(input_seq.shape),
                        "datatype": "FP32",
                        "data": input_seq.flatten().tolist()
                    }
                ]
            }

            response = requests.post(api_url, json=payload)
            if response.status_code != 200:
                raise Exception(f"Erreur d'inférence : {response.status_code} - {response.text}")

            result = response.json()
            pred_norm = result["outputs"][0]["data"][0]
            pred = manual_inverse_scale(pred_norm, *scaler)
            predictions.append((year, pred))

            input_seq = np.roll(input_seq, -1, axis=2)
            input_seq[0, 0, -1] = pred_norm
            input_seq[0, 1, -1] = (year - year_scaler[0]) / (year_scaler[1] - year_scaler[0])

    months = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
    df_result = pd.DataFrame(predictions, columns=["Année", "Prédiction (Mds)"])
    df_result["Mois"] = months * (end_year - start_year + 1)
    return df_result

# === CONFIGURATION ===
SEQ_LENGTH = 36
API_URL = "https://tvadeportebest-working.apps.origins.heritage.africa/v2/models/tvadeportebest/infer"

# === STREAMLIT APP ===
st.title("Prédictions mensuelles de la tva de porte")

DATA_PATH = "tva_data.csv"
df = load_and_prepare_data(DATA_PATH)

scaled_values, scaled_years, scaler, year_scaler = scale_data(df, SEQ_LENGTH)

year_to_predict = st.number_input("Choisir l'année de début", min_value=2024, max_value=2030, value=2024)
end_year = st.number_input("Choisir l'année de fin", min_value=year_to_predict, max_value=2030, value=2025)

df_pred = predict_years_remote(
    scaled_values, scaled_years, scaler, year_scaler,
    seq_length=SEQ_LENGTH,
    api_url=API_URL,
    start_year=year_to_predict,
    end_year=end_year
)

st.subheader("Prédictions")
st.write(df_pred)

st.subheader("Graphique des prévisions")
pivot_df = df_pred.pivot(index="Mois", columns="Année", values="Prédiction (Mds)")
fig, ax = plt.subplots(figsize=(12, 6))
pivot_df.plot(kind="bar", ax=ax)
ax.set_title(f"Prévisions mensuelles de l'impôt direct ({year_to_predict} à {df_pred['Année'].max()})")
ax.set_ylabel("Milliards")
ax.set_xlabel("Mois")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
st.pyplot(fig)
