from flask import Blueprint, request, jsonify
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Definisci il Blueprint per Online Learning
online_learning_bp = Blueprint('online_learning', __name__)

class OnlineLearningModel:
    def __init__(self):
        self.model = SGDClassifier(loss='log')  # Modello di classificazione
        self.vectorizer = TfidfVectorizer()  # Per la trasformazione del testo in feature
        self.X = np.empty((0, 0))  # Feature matrix
        self.y = np.array([])  # Labels

    def increment(self, new_data, labels):
        # Trasformare i nuovi dati in feature
        new_features = self.vectorizer.fit_transform(new_data).toarray()

        # Concatenare le nuove feature e le etichette con i dati esistenti
        if self.X.size == 0:
            self.X = new_features
            self.y = labels
        else:
            self.X = np.vstack((self.X, new_features))
            self.y = np.hstack((self.y, labels))

        # Addestrare o aggiornare il modello con i nuovi dati
        self.model.partial_fit(new_features, labels)

    def predict(self, new_data):
        # Previsione per i nuovi dati
        new_features = self.vectorizer.transform(new_data).toarray()
        return self.model.predict(new_features)

# Inizializza il modello di apprendimento online
online_model = OnlineLearningModel()

@online_learning_bp.route('/increment', methods=['POST'])
def add_new_data():
    data = request.get_json()
    new_data = data['new_data']  # Nuovi dati in arrivo
    labels = data['labels']  # Etichette corrispondenti

    # Aggiungi nuovi dati e etichette al modello
    online_model.increment(new_data, labels)
    return jsonify({"message": "Nuovi dati aggiunti e modello aggiornato"})

@online_learning_bp.route('/predict', methods=['POST'])
def predict_new_users():
    data = request.get_json()
    new_data = data['new_data']  # Dati di nuovi utenti o prodotti

    # Fai previsioni sui nuovi dati
    predictions = online_model.predict(new_data)
    return jsonify({"predictions": predictions.tolist()})
