from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf

# Definisci il Blueprint per Autoencoder
autoencoder_fraud_bp = Blueprint('autoencoder_fraud_detection', __name__)

# Variabile globale per il modello
autoencoder_model = None

def create_autoencoder_model(input_shape):
    """
    Crea un modello Autoencoder per il rilevamento di anomalie.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(input_shape[0], activation='sigmoid')  # Output con la stessa forma dell'input
    ])
    
    model.compile(optimizer='adam', loss='mse')  # Utilizza MSE come perdita per il rilevamento di anomalie
    return model

def detect_anomalies(model, data, threshold=0.01):
    """
    Calcola la ricostruzione e il MSE per rilevare anomalie.
    """
    reconstruction = model.predict(data)
    mse = np.mean(np.power(data - reconstruction, 2), axis=1)
    anomalies = mse > threshold
    return anomalies

@autoencoder_fraud_bp.route('/train_model', methods=['POST'])
def train_autoencoder():
    """
    Addestra il modello Autoencoder sui dati forniti.
    """
    global autoencoder_model
    data = request.get_json()
    
    # Supponiamo che i dati siano una lista di transazioni/azioni (feature già normalizzate)
    input_data = np.array(data['transactions'])
    
    # Crea e addestra il modello
    autoencoder_model = create_autoencoder_model(input_shape=(input_data.shape[1],))
    autoencoder_model.fit(input_data, input_data, epochs=50, batch_size=32, verbose=0)  # Addestramento dell'Autoencoder

    return jsonify({"message": "Modello addestrato con successo!"})

@autoencoder_fraud_bp.route('/detect_fraud', methods=['POST'])
def detect_fraud():
    """
    Rileva frodi basate su dati di input utilizzando l'Autoencoder.
    """
    global autoencoder_model
    if autoencoder_model is None:
        return jsonify({"error": "Il modello non è stato addestrato. Effettua prima l'addestramento."}), 400

    data = request.get_json()
    
    # Supponiamo che i dati siano una lista di transazioni/azioni (feature già normalizzate)
    input_data = np.array(data['transactions'])
    
    # Rileva le anomalie
    anomalies = detect_anomalies(autoencoder_model, input_data)
    return jsonify({"anomalies": anomalies.tolist()})
