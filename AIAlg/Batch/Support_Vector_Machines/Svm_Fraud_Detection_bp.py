from flask import Blueprint, request, jsonify
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Blueprint per il rilevamento delle frodi
svm_fraud_detection_bp = Blueprint('fraud_detection', __name__)

# Simuliamo un dataset di esempio
data = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'feature3': np.random.rand(100),
    'target': np.random.choice([0, 1], size=100)  # 0 = non frode, 1 = frode
})

# Prepara i dati per l'addestramento
X = data[['feature1', 'feature2', 'feature3']]
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Costruzione del modello SVM
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)

@svm_fraud_detection_bp.route('/detect_fraud', methods=['POST'])
def detect_fraud():
    """
    Route per il rilevamento delle frodi.
    """
    data_input = request.json.get('data')

    if not data_input:
        return jsonify({"error": "Nessun dato fornito per il rilevamento delle frodi."}), 400

    # Predizione delle frodi
    predictions = svm_model.predict(np.array(data_input))

    return jsonify({"predictions": predictions.tolist()})
