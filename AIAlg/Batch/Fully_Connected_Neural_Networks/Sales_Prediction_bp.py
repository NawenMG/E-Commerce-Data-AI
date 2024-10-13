from flask import Blueprint, request, jsonify
import tensorflow as tf
import numpy as np

sales_prediction_bp = Blueprint('sales_prediction', __name__)

def create_sales_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),  # Input: 10 feature
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output: previsione di una singola vendita
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

@sales_prediction_bp.route('/predict_sales', methods=['POST'])
def predict_sales():
    data = request.get_json()

    if not data or 'features' not in data:
        return jsonify({"error": "Dati non forniti o mancanti."}), 400

    features = np.array(data['features'])
    model = create_sales_model()

    # Simula un addestramento rapido con dati sintetici per esempio
    model.fit(features, np.random.rand(len(features)), epochs=5, batch_size=32)
    
    predictions = model.predict(features)

    return jsonify({"predictions": predictions.tolist()})
