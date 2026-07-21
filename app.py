# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 21:07:20 2026

@author: mayad
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Seiteneinstellungen für das Dashboard
st.set_page_config(page_title="LSTM Forecasting Dashboard", layout="wide")

# Titel der App
st.title("📈 Interaktives LSTM-Prognose-Dashboard")
st.markdown("Dieses Dashboard nutzt ein optimiertes LSTM-Modell mit integrierten Kalender-Features (Wochentage, Monate, Feiertage) und einem 14-tägigen Lookback-Fenster.")

# -------------------------------------------------------------------------
# Daten & Modell-Training (gecached, damit die App extrem schnell reagiert)
# -------------------------------------------------------------------------
@st.cache_resource
def train_lstm_model():
    # 1. Realistische Daten mit Datumsbezug generieren
    np.random.seed(42)
    datums_bereich = pd.date_range(start="2025-01-01", periods=365, freq="D")
    t = np.arange(len(datums_bereich))

    # Erzeuge synthetische Verkäufe: Wochentagsmuster + Saisonalität + Rauschen
    wochentags_effekt = 15 * np.sin(2 * np.pi * datums_bereich.dayofweek / 7)
    monats_effekt = 10 * np.sin(2 * np.pi * datums_bereich.month / 12)
    rauschen = np.random.normal(0, 5, len(datums_bereich))

    verkaeufe = 50 + wochentags_effekt + monats_effekt + rauschen
    verkaeufe = np.clip(verkaeufe, 0, None)  # Keine negativen Verkäufe

    df = pd.DataFrame(index=datums_bereich)
    df['Verkäufe'] = verkaeufe

    # 2. Kalender- & Feiertags-Features (Manuelle Logik)
    df['Wochentag'] = df.index.dayofweek
    df['Monat'] = df.index.month

    # Zyklische Transformationen
    df['weekday_sin'] = np.sin(2 * np.pi * df['Wochentag'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['Wochentag'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['Monat'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Monat'] / 12)

    # Feiertage simulieren (z.B. Neujahr, Ostern, Weihnachten etc.)
    df['Feiertag'] = 0
    feiertage = ['2025-01-01', '2025-04-18', '2025-04-21', '2025-05-01', '2025-10-03', '2025-12-25', '2025-12-26']
    df.loc[df.index.isin(pd.to_datetime(feiertage)), 'Feiertag'] = 1

    # Feature-Liste & Skalierung
    feature_cols = ['Verkäufe', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos', 'Feiertag']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])

    # Datenset für LSTM vorbereiten
    lookback = 14
    forecast_length = 5
    X, y = [], []

    for i in range(len(scaled_data) - lookback - forecast_length + 1):
        X.append(scaled_data[i:(i + lookback)])
        y.append(scaled_data[(i + lookback):(i + lookback + forecast_length), 0])

    X, y = np.array(X), np.array(y)

    # Train-Test Split (80% / 20%)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # LSTM Modell
    model = Sequential([
        LSTM(128, activation='tanh', return_sequences=False, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.05),
        Dense(64, activation='relu'),
        Dense(forecast_length, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='huber', metrics=['mae'])
    
    # Training
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=0)
    
    return model, history, X_test, y_test, lookback, forecast_length

# Modell & Daten laden
with st.spinner("Modell wird im Hintergrund trainiert... Bitte kurz warten... ⏳"):
    model, history, X_test, y_test, lookback, forecast_length = train_lstm_model()
st.success("Modell erfolgreich geladen!")

# -------------------------------------------------------------------------
# Web-Oberfläche Layout
# -------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🔮 Interaktive Prognose", "📊 Modell-Lernkurve"])

with tab1:
    st.subheader("Wähle ein Test-Szenario aus")
    
    # Sidebar oder Slider für den Test-Index
    max_index = len(X_test) - 1
    test_index = st.slider("Wähle den Startzeitpunkt (Test-Beispiel Index):", 0, max_index, 15)

    # Daten für das ausgewählte Beispiel vorbereiten
    historische_werte = X_test[test_index, :, 0]
    tatsaechliche_zukunft = y_test[test_index]

    # Vorhersage generieren
    vorhersage_input = X_test[test_index].reshape(1, X_test.shape[1], X_test.shape[2])
    prognose = model.predict(vorhersage_input)[0]

    # KPI Metriken berechnen
    fehler = np.mean(np.abs(tatsaechliche_zukunft - prognose))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Ausgewähltes Test-Szenario", f"Index {test_index}")
    col2.metric("Durchschnittlicher Vorhersagefehler (MAE)", f"{fehler:.4f}")
    col3.metric("Lookback-Fenster", f"{lookback} Tage")

    # Plot erstellen mit Matplotlib
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(lookback), historische_werte, color='black', marker='o', label='Historie (letzte 14 Tage mit Kalender-Features)')
    ax.plot(range(lookback, lookback + forecast_length), tatsaechliche_zukunft, color='blue', marker='o', label='Tatsächliche Verkäufe (Zukunft)')
    ax.plot(range(lookback, lookback + forecast_length), prognose, color='red', linestyle='--', marker='x', label='Modell-Prognose (inkl. Feiertage)')

    ax.axvline(x=lookback-0.5, color='gray', linestyle=':', label='Start der Prognose')
    ax.set_title(f'Gesamtprognose für Test-Beispiel {test_index}')
    ax.set_xlabel('Zeitschritte (Tage)')
    ax.set_ylabel('Verkäufe (skaliert)')
    ax.legend()
    ax.grid(True)
    
    # Plot in Streamlit anzeigen
    st.pyplot(fig)

with tab2:
    st.subheader("Trainingsverlauf & Lernkurve des Modells")
    
    # Lernkurven-Plot
    fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
    ax_loss.plot(history.history['loss'], label='Trainings-Loss (Huber)', color='blue')
    ax_loss.plot(history.history['val_loss'], label='Validierungs-Loss (Huber)', color='green')
    ax_loss.set_title('Lernkurve des feiertags-optimierten Modells')
    ax_loss.set_xlabel('Epochen')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(True)
    
    st.pyplot(fig_loss)
    
    st.info("💡 **Interpretation:** Solange die grüne Linie (Validierung) nahe bei der blauen Linie (Training) liegt, generalisiert das Modell hervorragend und neigt nicht zu Overfitting.")