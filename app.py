import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Cashier Forecast App", layout="wide")

st.title("📊 Cashier Sales & Resource Forecast")
st.write("Lade deine Daten hoch, um die Vorhersagen zu starten.")

# 1. Datei-Upload im Sidebar
st.sidebar.header("1. Daten hochladen")
uploaded_file = st.sidebar.file_uploader("Wähle die Datei 'CashierData.csv'", type=["csv"])

if uploaded_file is None:
    st.info("Bitte lade deine `CashierData.csv` in der linken Seitenleiste hoch, um das Dashboard zu aktivieren.")
    st.stop()

# Daten einlesen (Semicolon getrennt)
df = pd.read_csv(uploaded_file, sep=";")

# Spalten bereinigen
df.columns = df.columns.str.strip()

# Datum konvertieren
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek

st.write("### 📋 Daten-Vorschau (Erste 5 Zeilen)")
st.dataframe(df.head())

# Feature-Kandidaten
candidate_features = ["mean_temp", "mean_humid", "public_holiday", "school_holiday"]
features = [f for f in candidate_features if f in df.columns]

if len(features) == 0:
    st.error("Die erforderlichen Wetter- und Kalenderspalten wurden in der Datei nicht gefunden.")
    st.stop()

st.write("### ⚙️ Modell-Training")
target_col = st.selectbox(
    "Welche Kategorie möchtest du vorhersagen?",
    options=["CutFlowers", "PotOwn", "PotPurchased", "Wholesale", "FruitsVegs", "Commodity"],
)

if target_col not in df.columns:
    st.error(f"Die Zielspalte '{target_col}' wurde in der Datei nicht gefunden.")
    st.stop()

# Optional: numerisch erzwingen (wichtig für ML)
cols_to_numeric = list(set(features + [target_col]))
for c in cols_to_numeric:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Training vorbereiten (Zeilen mit NaN im Target oder Features löschen)
train_df = df.dropna(subset=[target_col] + features)

if train_df.empty:
    st.error("Keine gültigen Trainingsdaten vorhanden (zu viele NaNs nach dem Bereinigen).")
    st.stop()

X = train_df[features]
y = train_df[target_col]

# Modell trainieren
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

st.success(f"Modell für **{target_col}** erfolgreich trainiert!")

# 2. Prognose-Eingabe
st.write("### 🔮 Neue Prognose erstellen")
st.write("Gib die Wetter- und Kalenderdaten für den Tag ein, den du vorhersagen willst:")

input_data = {}
col1, col2 = st.columns(2)

with col1:
    if "mean_temp" in features:
        input_data["mean_temp"] = st.slider(
            "Durchschnittstemperatur (°C)",
            float(df["mean_temp"].min()),
            float(df["mean_temp"].max()),
            float(df["mean_temp"].mean()),
        )

    if "mean_humid" in features:
        input_data["mean_humid"] = st.slider(
            "Luftfeuchtigkeit (%)",
            float(df["mean_humid"].min()),
            float(df["mean_humid"].max()),
            float(df["mean_humid"].mean()),
        )

with col2:
    if "public_holiday" in features:
        input_data["public_holiday"] = st.checkbox("Feiertag (Public Holiday)?")

    if "school_holiday" in features:
        input_data["school_holiday"] = st.checkbox("Schulferien (School Holiday)?")

# Prediction Button (verhindert Re-Computing bei jeder UI-Änderung)
if st.button("Vorhersagen"):
    input_df = pd.DataFrame([input_data])

    # Checkboxen -> 0/1
    if "public_holiday" in input_df.columns:
        input_df["public_holiday"] = input_df["public_holiday"].astype(int)

    if "school_holiday" in input_df.columns:
        input_df["school_holiday"] = input_df["school_holiday"].astype(int)

    # Prediction ausführen
    prediction = model.predict(input_df[features])[0]
    st.metric(label=f"Prognostizierter Wert für {target_col}", value=f"{prediction:.2f}")
