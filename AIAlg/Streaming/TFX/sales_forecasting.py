import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Blueprint, jsonify, request

forecasting_bp = Blueprint('forecasting', __name__)

# Carica i dati
data = pd.read_csv('sales_data.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Funzione per creare il modello LSTM
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Previsione della domanda
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Funzione per preparare i dati per il modello
def prepare_data(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Route per la previsione
@forecasting_bp.route('/forecast', methods=['POST'])
def forecast():
    """Previsione delle vendite in base ai dati storici."""
    time_steps = request.json.get('time_steps', 30)  # Numero di giorni da utilizzare per la previsione
    recent_data = data['sales'].values[-time_steps:]  # Prendi i dati recenti

    # Prepara i dati per il modello
    X, y = prepare_data(recent_data, time_steps=1)

    # Reshape per il modello LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Crea e addestra il modello
    model = create_model((X.shape[1], 1))
    model.fit(X, y, epochs=10, verbose=0)

    # Effettua la previsione
    prediction = model.predict(X[-1].reshape((1, 1, 1)))

    return jsonify({'forecast': prediction[0][0]})
