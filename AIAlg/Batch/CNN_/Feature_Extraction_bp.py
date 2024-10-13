from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Blueprint per l'estrazione di feature
feature_extraction_bp = Blueprint('feature_extraction', __name__)

def create_feature_extraction_model(input_shape):
    """
    Crea un modello di rete neurale per l'estrazione di feature da dati strutturati.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='sigmoid')  # Output: 10 feature estratte
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

@feature_extraction_bp.route('/extract_features', methods=['POST'])
def extract_features():
    """
    Route per l'estrazione di feature da dati strutturati.
    """
    data = request.json

    if 'structured_data' not in data:
        return jsonify({"error": "Dati strutturati non forniti."}), 400
    
    # Supponiamo che i dati strutturati siano una lista di liste (simile a un DataFrame)
    structured_data = np.array(data['structured_data'])
    
    # Standardizzazione dei dati
    scaler = StandardScaler()
    structured_data_scaled = scaler.fit_transform(structured_data)

    # Crea il modello per l'estrazione di feature
    model = create_feature_extraction_model((structured_data_scaled.shape[1],))

    # Addestramento simulato (puoi sostituirlo con un modello pre-addestrato)
    x_train = np.random.rand(100, structured_data_scaled.shape[1])  # Dati di esempio
    y_train = np.random.rand(100, 10)  # Etichette di esempio
    model.fit(x_train, y_train, epochs=10, batch_size=32)  # Addestra il modello

    # Estrai le feature dai dati strutturati
    extracted_features = model.predict(structured_data_scaled).tolist()
    
    return jsonify({"extracted_features": extracted_features})
