from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Blueprint per il rilevamento di anomalie
autoencoder_anomaly_bp = Blueprint('autoencoder_anomaly', __name__)

# Simuliamo un dataset di esempio
# Questo dovrebbe essere caricato da un database o da un file in un'applicazione reale
data = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'feature3': np.random.rand(100),
})

# Prepara i dati per l'addestramento
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Costruzione del modello Autoencoder
def build_autoencoder(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(input_shape[0], activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Inizializzazione del modello
autoencoder_model = build_autoencoder((data_scaled.shape[1],))
autoencoder_model.fit(data_scaled, data_scaled, epochs=50, batch_size=10, verbose=0)

@autoencoder_anomaly_bp.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    """
    Route per il rilevamento di anomalie nei dati.
    """
    data_input = request.json.get('data')

    if not data_input:
        return jsonify({"error": "Nessun dato fornito per il rilevamento delle anomalie."}), 400

    # Normalizza i dati in ingresso
    input_data = np.array(data_input)
    input_data_scaled = scaler.transform(input_data)

    # Previsione con l'autoencoder
    reconstructed_data = autoencoder_model.predict(input_data_scaled)

    # Calcola l'errore quadratico medio
    mse = np.mean(np.power(input_data_scaled - reconstructed_data, 2), axis=1)

    # Soglia per determinare le anomalie (può essere personalizzata)
    threshold = np.percentile(mse, 95)
    anomalies = mse > threshold

    return jsonify({
        "anomalies": anomalies.tolist(),
        "threshold": threshold
    })
