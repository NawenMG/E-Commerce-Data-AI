from flask import Blueprint, request, jsonify
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Blueprint per le decisioni basate su regole
decision_tree_rule_based_decision_bp = Blueprint('rule_based_decision', __name__)

# Simuliamo un dataset di esempio
data = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'decision': np.random.choice(['approve', 'reject'], size=100)  # Decisioni
})

# Variabili globali per il modello e lo scaler
decision_tree_model = None
scaler = None

def train_model():
    """
    Funzione per addestrare il modello con il dataset iniziale.
    """
    global decision_tree_model, scaler

    # Prepara i dati per l'addestramento
    X = data[['feature1', 'feature2']]
    y = data['decision']
    
    # Divisione in set di addestramento e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardizzazione dei dati
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Costruzione del modello Albero di Decisione
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(X_train, y_train)

# Addestra il modello inizialmente
train_model()

# ============================
# Route per aggiornare il modello
# ============================
@decision_tree_rule_based_decision_bp.route('/train_model', methods=['POST'])
def update_model():
    """
    Route per aggiornare il modello con nuovi dati.
    """
    global data
    new_data = request.json.get('data')

    if not new_data or not isinstance(new_data, list):
        return jsonify({"error": "Nessun dato fornito o formato errato per l'addestramento."}), 400

    # Converti i nuovi dati in DataFrame
    new_data_df = pd.DataFrame(new_data)
    
    # Aggiorna il dataset esistente
    data = pd.concat([data, new_data_df], ignore_index=True)

    # Riaddestra il modello
    train_model()
    
    return jsonify({"message": "Modello aggiornato con successo."})

# ============================
# Route per le decisioni basate su regole
# ============================
@decision_tree_rule_based_decision_bp.route('/make_decision', methods=['POST'])
def make_decision():
    """
    Route per decisioni basate su regole.
    """
    data_input = request.json.get('data')

    if not data_input:
        return jsonify({"error": "Nessun dato fornito per la decisione basata su regole."}), 400

    # Normalizza i dati in ingresso
    input_data = np.array(data_input)
    input_data_scaled = scaler.transform(input_data)

    # Predizione delle decisioni
    decisions = decision_tree_model.predict(input_data_scaled)

    return jsonify({"decisions": decisions.tolist()})
