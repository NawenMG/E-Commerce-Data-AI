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
sales_df.set_index('date', inplace=True)

# ============================
# Route per analizzare la stagionalità
# ============================
@arima_seasonal_analysis_bp.route('/analyze_seasonality', methods=['POST'])
def analyze_seasonality():
    """
    Route per l'analisi delle tendenze stagionali.
    """
    try:
        # Estrai il periodo dal JSON, default a 12 se non fornito
        period = request.json.get('period', 12)
        
        # Decomponi la serie temporale
        decomposition = seasonal_decompose(sales_df['sales'], model='additive', period=period)
        
        # Estrai le componenti
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        return jsonify({
            "trend": trend.dropna().tolist(),         # Rimuove i NaN
            "seasonal": seasonal.tolist(),
            "residual": residual.dropna().tolist()    # Rimuove i NaN
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ============================
# Route per aggiornare i dati di vendita
# ============================
@arima_seasonal_analysis_bp.route('/update_sales_data', methods=['POST'])
def update_sales_data():
    """
    Route per aggiornare il dataset delle vendite.
    """
    try:
        # Ricevi i nuovi dati di vendita
        new_sales_data = request.json.get('sales_data')

        if not new_sales_data or not isinstance(new_sales_data, list):
            return jsonify({"error": "Dati di vendita non validi."}), 400
        
        # Aggiungi nuovi dati al dataframe
        new_dates = pd.date_range(start=sales_df.index[-1] + pd.Timedelta(days=1), periods=len(new_sales_data))
        new_sales_df = pd.DataFrame({'date': new_dates, 'sales': new_sales_data})
        new_sales_df.set_index('date', inplace=True)

        global sales_df  # Assicura di aggiornare il dataframe globale
        sales_df = pd.concat([sales_df, new_sales_df])
        
        return jsonify({"message": "Dati di vendita aggiornati con successo!"})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
