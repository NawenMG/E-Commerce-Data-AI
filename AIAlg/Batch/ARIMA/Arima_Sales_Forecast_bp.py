from flask import Blueprint, request, jsonify
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Definizione del Blueprint
sales_forecast_bp = Blueprint('sales_forecast_bp', __name__)

# ============================
# LSTM Model Definition
# ============================
def create_lstm_model(input_shape):
    """
    Crea un modello LSTM per la previsione delle vendite
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)  # Previsione di un valore continuo (es. vendite future)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Impostiamo la forma dell'input del modello LSTM
input_shape = (10, 1)  # Supponiamo di usare una sequenza di 10 dati storici
lstm_model = create_lstm_model(input_shape)

# Scaler per normalizzare i dati
scaler = MinMaxScaler(feature_range=(0, 1))

# ============================
# Route per la previsione delle vendite
# ============================
@sales_forecast_bp.route('/predict_sales', methods=['POST'])
def predict_sales():
    """
    Prevedi le vendite future utilizzando un modello LSTM addestrato.
    """
    try:
        # Ricevi i dati in input (ad esempio, gli ultimi 10 dati di vendita)
        data = request.json['data']
        input_data = np.array(data).reshape(-1, 10, 1)  # Reshape per adattarsi all'input del modello
        
        # Pre-processare i dati di input (normalizzazione)
        input_data = scaler.transform(input_data.reshape(-1, 1)).reshape(-1, 10, 1)

        # Esegue la previsione
        prediction = lstm_model.predict(input_data)

        # Riporta il valore predetto al suo spazio originale (de-normalizzazione)
        predicted_sales = scaler.inverse_transform(prediction)

        return jsonify({'predicted_sales': predicted_sales.flatten().tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ============================
# Route per addestrare il modello LSTM
# ============================
@sales_forecast_bp.route('/train_lstm', methods=['POST'])
def train_lstm():
    """
    Addestra il modello LSTM sui dati forniti per prevedere le vendite future.
    """
    try:
        # Ricevi i dati di addestramento
        x_train = np.array(request.json['x_train']).reshape(-1, 10, 1)
        y_train = np.array(request.json['y_train']).reshape(-1, 1)  # Vendite storiche come target
        
        # Normalizzare i dati
        x_train = scaler.fit_transform(x_train.reshape(-1, 1)).reshape(-1, 10, 1)
        y_train = scaler.fit_transform(y_train)

        # Addestra il modello LSTM
        lstm_model.fit(x_train, y_train, epochs=10, batch_size=32)
        
        return jsonify({'message': 'Modello addestrato con successo!'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
