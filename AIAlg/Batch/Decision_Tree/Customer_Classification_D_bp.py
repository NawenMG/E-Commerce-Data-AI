from flask import Blueprint, request, jsonify
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Blueprint per la classificazione dei clienti
decision_tree_customer_classification_bp = Blueprint('customer_classification', __name__)

# Simuliamo un dataset di esempio
data = pd.DataFrame({
    'age': np.random.randint(18, 70, size=100),
    'income': np.random.randint(30000, 100000, size=100),
    'previous_purchases': np.random.randint(0, 10, size=100),
    'customer_class': np.random.choice(['new', 'returning'], size=100)  # Classi di clienti
})

# Prepara i dati per l'addestramento
X = data[['age', 'income', 'previous_purchases']]
y = data['customer_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizzazione dei dati
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Costruzione del modello Albero di Decisione
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)

@decision_tree_customer_classification_bp.route('/classify_customer', methods=['POST'])
def classify_customer():
    """
    Route per la classificazione dei clienti.
    """
    data_input = request.json.get('data')

    if not data_input:
        return jsonify({"error": "Nessun dato fornito per la classificazione dei clienti."}), 400

    # Normalizza i dati in ingresso
    input_data = np.array(data_input)
    input_data_scaled = scaler.transform(input_data)

    # Predizione delle classi di clienti
    classes = decision_tree_model.predict(input_data_scaled)

    return jsonify({"customer_classes": classes.tolist()})
