from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf

# Blueprint per il Collaborative Filtering
suggest_products_bp = Blueprint('collaborative_filtering', __name__)

class CollaborativeFilteringModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_size=50):
        super(CollaborativeFilteringModel, self).__init__()
        # Crea embedding per utenti e prodotti
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_size)
    
    def call(self, inputs):
        # Estrae embedding di utente e prodotto
        user_vector = self.user_embedding(inputs[:, 0])  # Primo input è l'utente
        item_vector = self.item_embedding(inputs[:, 1])  # Secondo input è il prodotto

        # Calcola il prodotto scalare tra utente e prodotto
        dot_product = tf.reduce_sum(user_vector * item_vector, axis=1)
        return dot_product

# Inizializza il modello (ma non addestrarlo qui)
model = None

# ============================
# Route per suggerire prodotti
# ============================
@suggest_products_bp.route('/suggest_products', methods=['POST'])
def suggest_products():
    """
    Route per suggerire prodotti basati su un utente specifico.
    """
    global model
    data = request.json

    if 'user_id' not in data:
        return jsonify({"error": "ID utente non fornito."}), 400
    
    user_id = data['user_id']

    # Supponiamo di avere accesso a una lista di articoli predefiniti
    item_ids = np.arange(10)  # Esempio: ID di articoli disponibili

    if model is None:
        return jsonify({"error": "Il modello non è stato addestrato."}), 400

    # Suggerire articoli
    pairs_to_predict = np.array([[user_id, item_id] for item_id in item_ids])
    predictions = model.predict(pairs_to_predict).flatten().tolist()

    # Restituisce i suggerimenti ordinati per punteggio
    suggested_products = sorted(zip(item_ids, predictions), key=lambda x: x[1], reverse=True)[:5]

    return jsonify({"suggested_products": suggested_products})

# ============================
# Route per addestrare il modello
# ============================
@suggest_products_bp.route('/train_model', methods=['POST'])
def train_model():
    """
    Route per addestrare il modello di Collaborative Filtering con i dati forniti.
    """
    global model
    data = request.json

    if 'user_item_pairs' not in data or 'ratings' not in data:
        return jsonify({"error": "Dati non forniti."}), 400
    
    user_item_pairs = np.array(data['user_item_pairs'])  # Pair di utenti e articoli
    ratings = np.array(data['ratings'])  # Valutazioni corrispondenti
    
    # Numero di utenti e articoli
    num_users = np.max(user_item_pairs[:, 0]) + 1
    num_items = np.max(user_item_pairs[:, 1]) + 1

    # Crea il modello
    model = CollaborativeFilteringModel(num_users, num_items)

    # Addestramento del modello
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(user_item_pairs, ratings, epochs=10, batch_size=32)

    return jsonify({"message": "Modello di Collaborative Filtering addestrato con successo!"})
