from flask import Blueprint, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Blueprint per la previsione del comportamento dei clienti
random_forest_customer_behavior_bp = Blueprint('customer_behavior', __name__)

# Simuliamo un dataset di esempio
data = pd.DataFrame({
    'age': np.random.randint(18, 70, size=100),
    'income': np.random.randint(30000, 100000, size=100),
    'previous_purchases': np.random.randint(0, 10, size=100),
    'behavior': np.random.choice(['active', 'inactive'], size=100)  # Comportamento del cliente
})

# Prepara i dati per l'addestramento
X = data[['age', 'income', 'previous_purchases']]
y = data['behavior']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizzazione dei dati
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Costruzione del modello Random Forest
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)

@random_forest_customer_behavior_bp.route('/predict_customer_behavior', methods=['POST'])
def predict_customer_behavior():
    """
    Route per la previsione del comportamento dei clienti.
    """
    data_input = request.json.get('data')

    if not data_input:
        return jsonify({"error": "Nessun dato fornito per la previsione del comportamento dei clienti."}), 400

    # Normalizza i dati in ingresso
    input_data = np.array(data_input)
    input_data_scaled = scaler.transform(input_data)

    # Predizione del comportamento
    behaviors = random_forest_model.predict(input_data_scaled)

    return jsonify({"behaviors": behaviors.tolist()})
