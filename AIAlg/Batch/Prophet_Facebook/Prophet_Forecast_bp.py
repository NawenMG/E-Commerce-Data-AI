from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
from prophet import Prophet

# Blueprint per le previsioni con Prophet
prophet_forecast_bp = Blueprint('prophet_forecast', __name__)

# Variabile globale per il modello Prophet
prophet_model = None

@prophet_forecast_bp.route('/train_model', methods=['POST'])
def train_model():
    """
    Route per l'addestramento del modello Prophet.
    """
    global prophet_model

    # Simuliamo un dataset di esempio
    # Genera date e vendite casuali
    dates = pd.date_range(start='2022-01-01', periods=100)
    sales_data = [round(x) for x in (100 + 10 * np.sin(np.linspace(0, 3.14, 100)) + np.random.normal(0, 5, 100))]  # Dati di esempio
    sales_df = pd.DataFrame({'ds': dates, 'y': sales_data})  # Prophet si aspetta 'ds' e 'y'

    # Inizializziamo il modello Prophet e lo addestriamo
    prophet_model = Prophet()
    prophet_model.fit(sales_df)

    return jsonify({"message": "Modello Prophet addestrato con successo."})

@prophet_forecast_bp.route('/forecast', methods=['POST'])
def forecast():
    """
    Route per le previsioni di serie temporali.
    """
    global prophet_model

    if prophet_model is None:
        return jsonify({"error": "Il modello non è stato addestrato. Effettua prima l'addestramento."}), 400

    data_input = request.json.get('periods')

    if not data_input or not isinstance(data_input, int):
        return jsonify({"error": "Numero di periodi non fornito o non valido."}), 400

    # Crea un dataframe per le previsioni future
    future = prophet_model.make_future_dataframe(periods=data_input)
    
    # Previsione
    forecast = prophet_model.predict(future)

    # Restituisce le previsioni
    return jsonify({
        "forecast": forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(data_input).to_dict(orient='records')
    })
