from flask import Blueprint, request, jsonify
import tensorflow as tf
import numpy as np

market_segmentation_bp = Blueprint('market_segmentation', __name__)

def create_segmentation_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),  # Input: 10 feature
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')  # Output: 5 segmenti di mercato
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

@market_segmentation_bp.route('/segment_market', methods=['POST'])
def segment_market():
    data = request.get_json()

    if not data or 'features' not in data or 'labels' not in data:
        return jsonify({"error": "Dati non forniti o mancanti."}), 400

    features = np.array(data['features'])
    labels = np.array(data['labels'])  # Label numeriche per segmentazione

    model = create_segmentation_model()

    # Simula un addestramento rapido con i dati forniti
    model.fit(features, labels, epochs=5, batch_size=32)
    
    predictions = model.predict(features)
    predicted_segments = np.argmax(predictions, axis=1)

    return jsonify({"predicted_segments": predicted_segments.tolist()})
