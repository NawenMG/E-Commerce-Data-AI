from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf

# Definisci il Blueprint per LSTM
lstm_anomaly_bp = Blueprint('lstm_anomaly_detection', __name__)

def create_lstm_model(input_shape):
    # Crea un modello LSTM per il rilevamento di anomalie
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
        tf.keras.layers.LSTM(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output con una previsione singola (es. valore anomalo)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def detect_lstm_anomalies(model, data, threshold=0.01):
    # Utilizza il modello LSTM per fare previsioni e rilevare anomalie
    predictions = model.predict(data)
    mse = np.mean(np.power(data - predictions, 2), axis=1)
    anomalies = mse > threshold
    return anomalies

@lstm_anomaly_bp.route('/detect_traffic_anomalies', methods=['POST'])
def detect_traffic_anomalies():
    data = request.get_json()
    
    # Supponiamo che i dati siano una lista di log transazioni o traffico
    input_data = np.array(data['traffic_logs'])
    
    # Preprocessing: Dividi i dati in sequenze temporali (ad esempio finestre temporali di 30 punti)
    sequence_length = 30
    sequences = []
    for i in range(len(input_data) - sequence_length):
        sequences.append(input_data[i: i + sequence_length])
    
    sequences = np.array(sequences)
    
    # Crea e addestra il modello
    model = create_lstm_model(input_shape=(sequences.shape[1], sequences.shape[2]))
    model.fit(sequences, sequences, epochs=50, batch_size=32, verbose=0)  # Addestramento

    # Rileva le anomalie
    anomalies = detect_lstm_anomalies(model, sequences)
    return jsonify({"anomalies": anomalies.tolist()})
