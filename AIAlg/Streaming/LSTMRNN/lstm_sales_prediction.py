from flask import Blueprint, request, jsonify
import tensorflow as tf
import numpy as np

# Definizione del Blueprint
lstm_sales_bp = Blueprint('rnn_sales_bp', __name__)

# Variabile globale per il modello LSTM
lstm_model = None

# ============================
# LSTM Model Definition
# ============================
def create_lstm_model(input_shape):
    """
    Crea un modello LSTM per la previsione del comportamento o delle vendite.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
        tf.keras.layers.LSTM(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output: una singola previsione (ad esempio, vendite o comportamento futuro)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

@lstm_sales_bp.route('/train_model', methods=['POST'])
def train_lstm():
    """
    Addestra il modello LSTM sui dati forniti per migliorare le previsioni.
    """
    global lstm_model
    try:
        # Riceve i dati di addestramento
        x_train = np.array(request.json['x_train']).reshape(-1, 10, 1)  # Reshape per il modello
        y_train = np.array(request.json['y_train'])
        
        # Crea e addestra il modello LSTM
        lstm_model = create_lstm_model(input_shape=(10, 1))
        lstm_model.fit(x_train, y_train, epochs=10, batch_size=32)
        
        return jsonify({'message': 'Modello addestrato con successo!'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ============================
# Route per la previsione delle vendite
# ============================
@lstm_sales_bp.route('/predict_sales', methods=['POST'])
def predict_sales():
    """
    Previsione delle vendite in tempo reale basata sul comportamento del cliente.
    """
    global lstm_model
    try:
        if lstm_model is None:
            return jsonify({"error": "Il modello non è stato addestrato. Effettua prima l'addestramento."}), 400
        
        # Riceve i dati di input (comportamento utente o storico vendite)
        data = request.json['data']
        input_data = np.array(data).reshape(-1, 10, 1)  # Reshape per adattarsi all'input del modello
        
        # Esegue la previsione
        prediction = lstm_model.predict(input_data)
        
        # Restituisce la previsione come JSON
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
