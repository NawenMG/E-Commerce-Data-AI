from flask import Blueprint, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Blueprint per le stime di vendite
random_forest_sales_estimation_bp = Blueprint('sales_estimation', __name__)

# Variabili globali per il modello e lo scaler
random_forest_model = None
scaler = None

@random_forest_sales_estimation_bp.route('/train_model', methods=['POST'])
def train_model():
    """
    Route per l'addestramento del modello Random Forest.
    """
    global random_forest_model, scaler

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

    # Codifica la variabile categorica 'season'
    X = pd.get_dummies(X, columns=['season'], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardizzazione dei dati
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Costruzione del modello Random Forest
    random_forest_model = RandomForestRegressor()
    random_forest_model.fit(X_train, y_train)

    return jsonify({"message": "Modello Random Forest addestrato con successo."})

@random_forest_sales_estimation_bp.route('/estimate_sales', methods=['POST'])
def estimate_sales():
    """
    Route per le stime di vendite.
    """
    global random_forest_model, scaler

    if random_forest_model is None or scaler is None:
        return jsonify({"error": "Il modello non è stato addestrato. Effettua prima l'addestramento."}), 400

    data_input = request.json.get('data')

    if not data_input:
        return jsonify({"error": "Nessun dato fornito per le stime di vendite."}), 400

    # Normalizza i dati in ingresso
    input_data = pd.DataFrame(data_input, columns=['advertising_budget', 'previous_sales', 'season'])
    input_data = pd.get_dummies(input_data, columns=['season'], drop_first=True)  # Codifica 'season'
    input_data = input_data.reindex(columns=['advertising_budget', 'previous_sales', 'season_high'], fill_value=0)  # Assicurati che tutte le colonne siano presenti
    input_data_scaled = scaler.transform(input_data)

    # Predizione delle vendite
    sales_estimations = random_forest_model.predict(input_data_scaled)

    return jsonify({"sales_estimations": sales_estimations.tolist()})
