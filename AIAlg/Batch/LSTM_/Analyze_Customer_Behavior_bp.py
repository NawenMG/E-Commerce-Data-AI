from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf

# Blueprint per RNN e LSTM
analyze_customer_behavior_bp = Blueprint('lstm', __name__)

# Variabile per tenere il modello LSTM
lstm_model = None

def create_lstm_model(input_shape):
    """
    Crea un modello LSTM per la previsione delle serie temporali.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
        tf.keras.layers.LSTM(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output: una singola previsione
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

@analyze_customer_behavior_bp.route('/train_model', methods=['POST'])
def train_model():
    """
    Route per l'addestramento del modello LSTM.
    """
    global lstm_model

    data = request.json

    if 'customer_sequences' not in data:
        return jsonify({"error": "Sequenze di clienti non fornite."}), 400
    
    customer_sequences = np.array(data['customer_sequences'])
    
    # Reshape per l'input della LSTM (samples, timesteps, features)
    customer_sequences = customer_sequences.reshape((customer_sequences.shape[0], customer_sequences.shape[1], 1))

    # Crea il modello LSTM
    lstm_model = create_lstm_model((customer_sequences.shape[1], 1))

    # Simula l'addestramento del modello con dati casuali (puoi sostituirlo con dati reali)
    x_train = np.random.rand(100, customer_sequences.shape[1], 1)  # Dati di esempio
    y_train = np.random.rand(100, 1)  # Etichette di esempio
    lstm_model.fit(x_train, y_train, epochs=10, batch_size=32)  # Addestra il modello

    return jsonify({"message": "Modello LSTM addestrato con successo."})

@analyze_customer_behavior_bp.route('/predict', methods=['POST'])
def predict():
    """
    Route per la previsione utilizzando il modello LSTM addestrato.
    """
    global lstm_model

    if lstm_model is None:
        return jsonify({"error": "Il modello non è stato addestrato. Effettua prima l'addestramento."}), 400

    data = request.json

    if 'customer_sequences' not in data:
        return jsonify({"error": "Sequenze di clienti non fornite."}), 400
    
    customer_sequences = np.array(data['customer_sequences'])
    
    # Reshape per l'input della LSTM (samples, timesteps, features)
    customer_sequences = customer_sequences.reshape((customer_sequences.shape[0], customer_sequences.shape[1], 1))

    # Previsione
    predictions = lstm_model.predict(customer_sequences).tolist()

    return jsonify({"predictions": predictions})
