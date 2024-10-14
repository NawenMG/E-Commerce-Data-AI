from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf

# Blueprint per RNN e LSTM
analyze_customer_behavior_bp = Blueprint('lstm', __name__)

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


@analyze_customer_behavior_bp.route('/analyze_customer_behavior', methods=['POST'])
def analyze_customer_behavior():
    """
    Route per l'analisi del comportamento sequenziale dei clienti utilizzando un modello LSTM.
    """
    data = request.json

    if 'customer_sequences' not in data:
        return jsonify({"error": "Sequenze di clienti non fornite."}), 400
    
    customer_sequences = np.array(data['customer_sequences'])
    
    # Reshape per l'input della LSTM (samples, timesteps, features)
    customer_sequences = customer_sequences.reshape((customer_sequences.shape[0], customer_sequences.shape[1], 1))

    # Crea il modello LSTM
    model = create_lstm_model((customer_sequences.shape[1], 1))

    # Addestramento simulato (puoi sostituirlo con un modello pre-addestrato)
    x_train = np.random.rand(100, customer_sequences.shape[1], 1)  # Dati di esempio
    y_train = np.random.rand(100, 1)  # Etichette di esempio
    model.fit(x_train, y_train, epochs=10, batch_size=32)  # Addestra il modello

    # Previsione
    predictions = model.predict(customer_sequences).tolist()

    return jsonify({"predictions": predictions})