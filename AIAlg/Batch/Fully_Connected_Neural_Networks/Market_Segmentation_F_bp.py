from flask import Blueprint, request, jsonify
import tensorflow as tf
import numpy as np

market_segmentation_bp = Blueprint('market_segmentation', __name__)

# Variabile globale per il modello
segmentation_model = None

def create_segmentation_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),  # Input: 10 feature
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')  # Output: 5 segmenti di mercato
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(features, labels):
    """
    Funzione per addestrare il modello con le features e le labels fornite.
    """
    global segmentation_model
    segmentation_model = create_segmentation_model()
    # Simula un addestramento con i dati forniti
    segmentation_model.fit(features, labels, epochs=5, batch_size=32)

@market_segmentation_bp.route('/train_model', methods=['POST'])
def update_model():
    """
    Route per aggiornare il modello con nuovi dati.
    """
    data = request.get_json()

    if not data or 'features' not in data or 'labels' not in data:
        return jsonify({"error": "Dati non forniti o mancanti."}), 400

    features = np.array(data['features'])
    labels = np.array(data['labels'])  # Label numeriche per segmentazione

    # Addestra il modello con i nuovi dati
    train_model(features, labels)

    return jsonify({"message": "Modello aggiornato con successo."})

@market_segmentation_bp.route('/segment_market', methods=['POST'])
def segment_market():
    data = request.get_json()

    if not data or 'features' not in data:
        return jsonify({"error": "Dati non forniti o mancanti."}), 400

    features = np.array(data['features'])

    if segmentation_model is None:
        return jsonify({"error": "Il modello non è stato addestrato."}), 400

    # Predizione dei segmenti di mercato
    predictions = segmentation_model.predict(features)
    predicted_segments = np.argmax(predictions, axis=1)

    return jsonify({"predicted_segments": predicted_segments.tolist()})
