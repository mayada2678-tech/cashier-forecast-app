import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Cashier Forecast App", layout="wide")

st.title("📊 Cashier Sales & Resource Forecast")
st.write("Lade deine Daten hoch, um die Vorhersagen zu starten.")

# 1. Datei-Upload im Sidebar
st.sidebar.header("1. Daten hochladen")
uploaded_file = st.sidebar.file_uploader("Wähle die Datei 'CashierData.csv'", type=["csv"])

if uploaded_file is not None:
    # Daten einlesen (Semicolon getrennt)
    df = pd.read_csv(uploaded_file, sep=";")
    
    # Spalten bereinigen
    df.columns = df.columns.str.strip()
    
    # Datum konvertieren
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    st.write("### 📋 Daten-Vorschau (Erste 5 Zeilen)")
    st.dataframe(df.head())
    
    # Spaltenauswahl für die Prognose
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Features (Einflussfaktoren) und Target (Vorhersageziel) definieren
    features = ['mean_temp', 'mean_humid', 'public_holiday', 'school_holiday']
    # Sicherstellen, dass diese Features existieren
    features = [f for f in features if f in df.columns]
    
    if len(features) > 0:
        st.write("### ⚙️ Modell-Training")
        target_col = st.selectbox("Welche Kategorie möchtest du vorhersagen?", 
                                  options=['CutFlowers', 'PotOwn', 'PotPurchased', 'Wholesale', 'FruitsVegs', 'Commodity'])
        
        if target_col in df.columns:
            # Daten für Training vorbereiten (Zeilen mit NaN im Target löschen)
            train_df = df.dropna(subset=[target_col] + features)
            
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
                if 'mean_temp' in features:
                    input_data['mean_temp'] = st.slider("Durchschnittstemperatur (°C)", float(df['mean_temp'].min()), float(df['mean_temp'].max()), float(df['mean_temp'].mean()))
                if 'mean_humid' in features:
                    input_data['mean_humid'] = st.slider("Luftfeuchtigkeit (%)", float(df['mean_humid'].min()), float(df['mean_humid'].max()), float(df['mean_humid'].mean()))
            
            with col2:
                if 'public_holiday' in features:
                    input_data['public_holiday'] = st.checkbox("Feiertag (Public Holiday)?")
                if 'school_holiday' in features:
                    input_data['school_holiday'] = st.checkbox("Schulferien (School Holiday)?")
            
            # Prediction ausführen
            input_df = pd.DataFrame([input_data])
            # Werte für Checkboxen in 0 und 1 umwandeln
            if 'public_holiday' in input_df.columns:
                input_df['public_holiday'] = input_df['public_holiday'].astype(int)
            if 'school_holiday' in input_df.columns:
                input_df['school_holiday'] = input_df['school_holiday'].astype(int)
                
            prediction = model.predict(input_df[features])[0]
            
            st.metric(label=f"Prognostizierter Wert für {target_col}", value=f"{prediction:.2f}")
            
    else:
        st.error("Die erforderlichen Wetter- und Kalenderspalten wurden in der Datei nicht gefunden.")
else:
    st.info("Bitte lade deine `CashierData.csv` in der linken Seitenleiste hoch, um das Dashboard zu aktivieren.")
