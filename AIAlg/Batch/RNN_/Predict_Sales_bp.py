from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf

# Blueprint per RNN e LSTM
rnn_bp = Blueprint('rnn', __name__)

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

@rnn_bp.route('/predict_sales', methods=['POST'])
def predict_sales():
    """
    Route per la previsione delle vendite utilizzando un modello RNN.
    """
    data = request.json

    if 'sales_data' not in data:
        return jsonify({"error": "Dati di vendita non forniti."}), 400
    
    sales_data = np.array(data['sales_data'])
    
    # Reshape per l'input della RNN (samples, timesteps, features)
    sales_data = sales_data.reshape((sales_data.shape[0], sales_data.shape[1], 1))

    # Crea il modello RNN
    model = create_rnn_model((sales_data.shape[1], 1))

    # Addestramento simulato (puoi sostituirlo con un modello pre-addestrato)
    x_train = np.random.rand(100, sales_data.shape[1], 1)  # Dati di esempio
    y_train = np.random.rand(100, 1)  # Etichette di esempio
    model.fit(x_train, y_train, epochs=10, batch_size=32)  # Addestra il modello

    # Previsione
    predictions = model.predict(sales_data).tolist()

    return jsonify({"predictions": predictions})