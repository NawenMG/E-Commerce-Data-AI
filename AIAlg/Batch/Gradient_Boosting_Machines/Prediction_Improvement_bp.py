from flask import Blueprint, request, jsonify
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Blueprint per il miglioramento delle previsioni
gradient_boosting_prediction_improvement_bp = Blueprint('prediction_improvement', __name__)

# Simuliamo un dataset di esempio
data = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'feature3': np.random.rand(100),
    'target': np.random.rand(100)  # Target per miglioramento previsioni
})

# Prepara i dati per l'addestramento
X = data[['feature1', 'feature2', 'feature3']]
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Costruzione del modello Gradient Boosting
gradient_boosting_model = GradientBoostingRegressor()
gradient_boosting_model.fit(X_train, y_train)

@gradient_boosting_prediction_improvement_bp.route('/improve_predictions', methods=['POST'])
def improve_predictions():
    """
    Route per il miglioramento delle previsioni basate su modelli deboli.
    """
    data_input = request.json.get('data')

    if not data_input:
        return jsonify({"error": "Nessun dato fornito per il miglioramento delle previsioni."}), 400

    # Predizione dei target
    predictions = gradient_boosting_model.predict(np.array(data_input))

    return jsonify({"predictions": predictions.tolist()})
