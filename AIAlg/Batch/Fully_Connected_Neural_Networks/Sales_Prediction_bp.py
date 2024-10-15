from flask import Blueprint, request, jsonify
import tensorflow as tf
import numpy as np

sales_prediction_bp = Blueprint('sales_prediction', __name__)

# Variabile globale per il modello
sales_model = None

def create_sales_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),  # Input: 10 feature
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output: previsione di una singola vendita
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def train_sales_model(features, labels):
    """
    Funzione per addestrare il modello con le features e le labels fornite.
    """
    global sales_model
    sales_model = create_sales_model()
    # Addestra il modello con i dati forniti
    sales_model.fit(features, labels, epochs=5, batch_size=32)

@sales_prediction_bp.route('/train_model', methods=['POST'])
def update_sales_model():
    """
    Route per aggiornare il modello con nuovi dati.
    """
    data = request.get_json()

    if not data or 'features' not in data or 'labels' not in data:
        return jsonify({"error": "Dati non forniti o mancanti."}), 400

    features = np.array(data['features'])
    labels = np.array(data['labels'])  # Label numeriche per la previsione delle vendite

    # Addestra il modello con i nuovi dati
    train_sales_model(features, labels)

    return jsonify({"message": "Modello aggiornato con successo."})

@sales_prediction_bp.route('/predict_sales', methods=['POST'])
def predict_sales():
    data = request.get_json()

    if not data or 'features' not in data:
        return jsonify({"error": "Dati non forniti o mancanti."}), 400

    features = np.array(data['features'])

    if sales_model is None:
        return jsonify({"error": "Il modello non è stato addestrato."}), 400

    # Predizione delle vendite
    predictions = sales_model.predict(features)

    return jsonify({"predictions": predictions.flatten().tolist()})
