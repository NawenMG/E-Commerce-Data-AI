from flask import Blueprint, request, jsonify
import tensorflow as tf
import numpy as np

customer_classification_bp = Blueprint('customer_classification', __name__)

def create_customer_classification_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),  # Input: 10 feature
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # Output: 3 classi di clienti
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

@customer_classification_bp.route('/classify_customers', methods=['POST'])
def classify_customers():
    data = request.get_json()

    if not data or 'features' not in data or 'labels' not in data:
        return jsonify({"error": "Dati non forniti o mancanti."}), 400

    features = np.array(data['features'])
    labels = np.array(data['labels'])  # Label numeriche per classificazione

    model = create_customer_classification_model()

    # Simula un addestramento rapido con i dati forniti
    model.fit(features, labels, epochs=5, batch_size=32)
    
    predictions = model.predict(features)
    predicted_classes = np.argmax(predictions, axis=1)

    return jsonify({"predicted_classes": predicted_classes.tolist()})
