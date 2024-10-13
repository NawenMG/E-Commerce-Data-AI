from flask import Blueprint, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Blueprint per le stime di vendite
random_forest_sales_estimation_bp = Blueprint('sales_estimation', __name__)

# Simuliamo un dataset di esempio
data = pd.DataFrame({
    'advertising_budget': np.random.randint(1000, 50000, size=100),
    'season': np.random.choice(['low', 'high'], size=100),
    'previous_sales': np.random.randint(100, 1000, size=100),
    'sales': np.random.randint(200, 1500, size=100)  # Vendite
})

# Prepara i dati per l'addestramento
X = data[['advertising_budget', 'season', 'previous_sales']]
y = data['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizzazione dei dati
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Costruzione del modello Random Forest
random_forest_model = RandomForestRegressor()
random_forest_model.fit(X_train, y_train)

@random_forest_sales_estimation_bp.route('/estimate_sales', methods=['POST'])
def estimate_sales():
    """
    Route per le stime di vendite.
    """
    data_input = request.json.get('data')

    if not data_input:
        return jsonify({"error": "Nessun dato fornito per le stime di vendite."}), 400

    # Normalizza i dati in ingresso
    input_data = np.array(data_input)
    input_data_scaled = scaler.transform(input_data)

    # Predizione delle vendite
    sales_estimations = random_forest_model.predict(input_data_scaled)

    return jsonify({"sales_estimations": sales_estimations.tolist()})
