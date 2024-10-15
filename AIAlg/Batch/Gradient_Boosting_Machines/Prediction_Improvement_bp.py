from flask import Blueprint, request, jsonify
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Blueprint per il miglioramento delle previsioni
gradient_boosting_prediction_improvement_bp = Blueprint('prediction_improvement', __name__)

# Variabile globale per il modello
gradient_boosting_model = None

def create_and_train_model(X, y):
    """
    Funzione per creare e addestrare il modello Gradient Boosting Regressor.
    """
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model

@gradient_boosting_prediction_improvement_bp.route('/train_model', methods=['POST'])
def train_model():
    """
    Route per addestrare il modello con nuovi dati.
    """
    data = request.get_json()

    if not data or 'features' not in data or 'target' not in data:
        return jsonify({"error": "Dati non forniti o mancanti."}), 400

    features = np.array(data['features'])
    target = np.array(data['target'])  # Label per la regressione

    # Addestra il modello con i nuovi dati
    global gradient_boosting_model
    gradient_boosting_model = create_and_train_model(features, target)

    return jsonify({"message": "Modello addestrato con successo."})

@gradient_boosting_prediction_improvement_bp.route('/improve_predictions', methods=['POST'])
def improve_predictions():
    """
    Route per il miglioramento delle previsioni basate su modelli deboli.
    """
    if gradient_boosting_model is None:
        return jsonify({"error": "Il modello non è stato addestrato."}), 400

    data_input = request.json.get('data')

    if not data_input:
        return jsonify({"error": "Nessun dato fornito per il miglioramento delle previsioni."}), 400

    # Predizione dei target
    predictions = gradient_boosting_model.predict(np.array(data_input))

    return jsonify({"predictions": predictions.tolist()})
