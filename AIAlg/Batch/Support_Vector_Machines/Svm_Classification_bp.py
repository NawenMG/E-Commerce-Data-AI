from flask import Blueprint, request, jsonify
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Blueprint per la classificazione
svm_classification_bp = Blueprint('classification', __name__)

# Simuliamo un dataset di esempio
data = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'feature3': np.random.rand(100),
    'target': np.random.choice(['class1', 'class2'], size=100)  # Target per classificazione
})

# Prepara i dati per l'addestramento
X = data[['feature1', 'feature2', 'feature3']]
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Costruzione del modello SVM
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)

@svm_classification_bp.route('/classify_customers', methods=['POST'])
def classify_customers():
    """
    Route per la classificazione dei clienti.
    """
    data_input = request.json.get('data')

    if not data_input:
        return jsonify({"error": "Nessun dato fornito per la classificazione."}), 400

    # Predizione delle classi
    predictions = svm_model.predict(np.array(data_input))

    return jsonify({"predictions": predictions.tolist()})
