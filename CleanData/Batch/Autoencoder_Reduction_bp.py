from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Blueprint per la riduzione della dimensionalità
autoencoder_reduction_bp = Blueprint('autoencoder_reduction', __name__)

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

@autoencoder_reduction_bp.route('/reduce_dimensionality', methods=['POST'])
def reduce_dimensionality():
    """
    Route per la riduzione della dimensionalità dei dati.
    """
    data_input = request.json.get('data')

    if not data_input:
        return jsonify({"error": "Nessun dato fornito per la riduzione della dimensionalità."}), 400

    # Normalizza i dati in ingresso
    input_data = np.array(data_input)
    input_data_scaled = scaler.transform(input_data)

    # Riduci la dimensionalità
    encoded_data = autoencoder_model.predict(input_data_scaled)

    # Restituisci i dati ridotti
    return jsonify({"reduced_data": encoded_data.tolist()})
