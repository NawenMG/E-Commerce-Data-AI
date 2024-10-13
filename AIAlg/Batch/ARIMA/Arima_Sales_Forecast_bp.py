from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Blueprint per la previsione delle vendite
arima_sales_forecast_bp = Blueprint('sales_forecast', __name__)

# Simuliamo un dataset di esempio
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', periods=100)
sales_data = np.random.randint(100, 200, size=(100,))
sales_df = pd.DataFrame({'date': dates, 'sales': sales_data})

# Impostiamo il modello ARIMA
model = ARIMA(sales_df['sales'], order=(5, 1, 0))
model_fit = model.fit()

@arima_sales_forecast_bp.route('/forecast_sales', methods=['POST'])
def forecast_sales():
    """
    Route per la previsione delle vendite.
    """
    data_input = request.json.get('periods')

    if not data_input or not isinstance(data_input, int):
        return jsonify({"error": "Numero di periodi non fornito o non valido."}), 400

    # Previsione delle vendite
    forecast = model_fit.forecast(steps=data_input)

    return jsonify({"forecasted_sales": forecast.tolist()})
