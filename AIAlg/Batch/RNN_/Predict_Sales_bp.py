from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf

# Blueprint per RNN e LSTM
predicate_Sales_RNNLSTM_bp = Blueprint('rnn', __name__)

# Variabili globali per il modello
rnn_model = None

def create_rnn_model(input_shape):
    """
    Crea un modello RNN per la previsione delle serie temporali.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(64, activation='relu', input_shape=input_shape, return_sequences=True),
        tf.keras.layers.SimpleRNN(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output: una singola previsione
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

@predicate_Sales_RNNLSTM_bp.route('/train_model', methods=['POST'])
def train_model():
    """
    Route per l'addestramento del modello RNN.
    """
    global rnn_model

    data = request.json

    if 'sales_data' not in data or 'labels' not in data:
        return jsonify({"error": "Dati di vendita o etichette non forniti."}), 400

    sales_data = np.array(data['sales_data'])
    labels = np.array(data['labels'])

    # Reshape per l'input della RNN (samples, timesteps, features)
    sales_data = sales_data.reshape((sales_data.shape[0], sales_data.shape[1], 1))

    # Crea il modello RNN
    rnn_model = create_rnn_model((sales_data.shape[1], 1))

    # Addestramento del modello
    rnn_model.fit(sales_data, labels, epochs=10, batch_size=32)

    return jsonify({"message": "Modello RNN addestrato con successo."})

@predicate_Sales_RNNLSTM_bp.route('/predict_sales', methods=['POST'])
def predict_sales():
    """
    Route per la previsione delle vendite utilizzando un modello RNN.
    """
    global rnn_model

    if rnn_model is None:
        return jsonify({"error": "Il modello non è stato addestrato. Effettua prima l'addestramento."}), 400
    
    data = request.json

    if 'sales_data' not in data:
        return jsonify({"error": "Dati di vendita non forniti."}), 400
    
    sales_data = np.array(data['sales_data'])
    
    # Reshape per l'input della RNN (samples, timesteps, features)
    sales_data = sales_data.reshape((sales_data.shape[0], sales_data.shape[1], 1))

    # Previsione
    predictions = rnn_model.predict(sales_data).tolist()

    return jsonify({"predictions": predictions})
