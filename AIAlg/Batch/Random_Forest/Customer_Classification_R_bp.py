from flask import Blueprint, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Blueprint per la classificazione di categorie di clienti
random_forest_customer_classification_bp = Blueprint('customer_classification', __name__)

# Variabili globali per il modello e lo scaler
random_forest_model = None
scaler = None

@random_forest_customer_classification_bp.route('/train_model', methods=['POST'])
def train_model():
    """
    Route per l'addestramento del modello Random Forest.
    """
    global random_forest_model, scaler

    # Simuliamo un dataset di esempio
    data = pd.DataFrame({
        'age': np.random.randint(18, 70, size=100),
        'income': np.random.randint(30000, 100000, size=100),
        'spending_score': np.random.randint(1, 100, size=100),
        'customer_category': np.random.choice(['high_value', 'medium_value', 'low_value'], size=100)  # Categorie di clienti
    })

    # Prepara i dati per l'addestramento
    X = data[['age', 'income', 'spending_score']]
    y = data['customer_category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardizzazione dei dati
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Costruzione del modello Random Forest
    random_forest_model = RandomForestClassifier()
    random_forest_model.fit(X_train, y_train)

    return jsonify({"message": "Modello Random Forest addestrato con successo."})

@random_forest_customer_classification_bp.route('/classify_customer', methods=['POST'])
def classify_customer():
    """
    Route per la classificazione di categorie di clienti.
    """
    global random_forest_model, scaler

    if random_forest_model is None or scaler is None:
        return jsonify({"error": "Il modello non è stato addestrato. Effettua prima l'addestramento."}), 400

    data_input = request.json.get('data')

    if not data_input:
        return jsonify({"error": "Nessun dato fornito per la classificazione di categorie di clienti."}), 400

    # Normalizza i dati in ingresso
    input_data = np.array(data_input)
    input_data_scaled = scaler.transform(input_data)

    # Predizione delle categorie di clienti
    categories = random_forest_model.predict(input_data_scaled)

    return jsonify({"customer_categories": categories.tolist()})
