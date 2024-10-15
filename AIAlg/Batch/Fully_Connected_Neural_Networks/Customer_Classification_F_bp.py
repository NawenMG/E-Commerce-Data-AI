from flask import Blueprint, request, jsonify
import tensorflow as tf
import numpy as np

customer_classification_bp = Blueprint('customer_classification', __name__)

# Variabili globali per il modello
customer_classification_model = None

def create_customer_classification_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),  # Input: 10 feature
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # Output: 3 classi di clienti
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(features, labels):
    """
    Funzione per addestrare il modello con le features e le labels fornite.
    """
    global customer_classification_model
    customer_classification_model = create_customer_classification_model()
    # Simula un addestramento con i dati forniti
    customer_classification_model.fit(features, labels, epochs=5, batch_size=32)

@customer_classification_bp.route('/train_model', methods=['POST'])
def update_model():
    """
    Route per aggiornare il modello con nuovi dati.
    """
    data = request.get_json()

    if not data or 'features' not in data or 'labels' not in data:
        return jsonify({"error": "Dati non forniti o mancanti."}), 400

    features = np.array(data['features'])
    labels = np.array(data['labels'])  # Label numeriche per classificazione

    # Addestra il modello con i nuovi dati
    train_model(features, labels)

    return jsonify({"message": "Modello aggiornato con successo."})

@customer_classification_bp.route('/classify_customers', methods=['POST'])
def classify_customers():
    data = request.get_json()

    if not data or 'features' not in data:
        return jsonify({"error": "Dati non forniti o mancanti."}), 400

    features = np.array(data['features'])

    if customer_classification_model is None:
        return jsonify({"error": "Il modello non è stato addestrato."}), 400

    # Predizione delle classi di clienti
    predictions = customer_classification_model.predict(features)
    predicted_classes = np.argmax(predictions, axis=1)

    return jsonify({"predicted_classes": predicted_classes.tolist()})
