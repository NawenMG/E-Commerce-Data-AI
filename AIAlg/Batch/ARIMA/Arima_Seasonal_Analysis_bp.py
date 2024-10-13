from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

# Blueprint per l'analisi delle tendenze stagionali
arima_seasonal_analysis_bp = Blueprint('seasonal_analysis', __name__)

# Simuliamo un dataset di esempio
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', periods=100)
sales_data = np.random.randint(100, 200, size=(100,))
sales_df = pd.DataFrame({'date': dates, 'sales': sales_data})

# Assicuriamoci che la colonna 'date' sia l'indice
sales_df.set_index('date', inplace=True)

@arima_seasonal_analysis_bp.route('/analyze_seasonality', methods=['POST'])
def analyze_seasonality():
    """
    Route per l'analisi delle tendenze stagionali.
    """
    decomposition = seasonal_decompose(sales_df['sales'], model='additive', period=12)
    
    # Estrai le componenti
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    return jsonify({
        "trend": trend.dropna().tolist(),         # Rimuove i NaN
        "seasonal": seasonal.tolist(),
        "residual": residual.dropna().tolist()    # Rimuove i NaN
    })
