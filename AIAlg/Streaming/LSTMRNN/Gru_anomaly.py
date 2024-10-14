from flask import Blueprint, request, jsonify
import tensorflow as tf
import numpy as np

# Definizione del Blueprint
gru_anomaly_bp = Blueprint('gru_anomaly_bp', __name__)

# ============================
# GRU Model Definition
# ============================
def create_gru_model(input_shape):
    """
    Crea un modello GRU per il rilevamento delle anomalie
    """
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(64, activation='relu', input_shape=input_shape, return_sequences=True),
        tf.keras.layers.GRU(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output: una singola previsione (ad esempio, se è un'anomalia o meno)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Inizializza il modello GRU
input_shape = (10, 1)  # Ad esempio: 10 feature di input
gru_model = create_gru_model(input_shape)

# ============================
# Route per il rilevamento delle anomalie
# ============================
@gru_anomaly_bp.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    """
    Rilevamento di anomalie in tempo reale basato su input come vendite o comportamento cliente
    """
    try:
        # Riceve i dati di input (ad esempio, dati di vendite o transazioni)
        data = request.json['data']
        input_data = np.array(data).reshape(-1, 10, 1)  # Reshape per adattarsi all'input del modello
        
        # Esegue la previsione
        prediction = gru_model.predict(input_data)
        
        # Rileva anomalie (ad esempio, se la previsione supera una soglia)
        anomalies = [1 if pred > 0.8 else 0 for pred in prediction]  # Threshold 0.8 come esempio
        
        return jsonify({'anomalies': anomalies})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ============================
# Route per addestrare il modello GRU
# ============================
@gru_anomaly_bp.route('/train_gru', methods=['POST'])
def train_gru():
    """
    Addestra il modello GRU sui dati forniti per rilevare anomalie
    """
    try:
        # Riceve i dati di addestramento
        x_train = np.array(request.json['x_train']).reshape(-1, 10, 1)
        y_train = np.array(request.json['y_train'])  # Ad esempio: 0 per normale, 1 per anomalia
        
        # Addestra il modello GRU
        gru_model.fit(x_train, y_train, epochs=10, batch_size=32)
        
        return jsonify({'message': 'Modello addestrato con successo!'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
