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

@suggest_products_bp.route('/suggest_products', methods=['POST'])
def suggest_products():
    """
    Route per suggerire prodotti basati su un utente specifico.
    """
    data = request.json

    if 'user_id' not in data:
        return jsonify({"error": "ID utente non fornito."}), 400
    
    user_id = data['user_id']

    # Supponiamo di avere accesso a una lista di articoli predefiniti
    item_ids = np.arange(10)  # Esempio di ID di articoli disponibili

    # Crea il modello (questo dovrebbe essere lo stesso modello addestrato prima)
    num_users = 10  # Esempio: numero totale di utenti
    num_items = 10  # Esempio: numero totale di articoli
    model = CollaborativeFilteringModel(num_users, num_items)

    # Addestramento simulato (puoi sostituirlo con un modello pre-addestrato)
    user_item_pairs = np.array([[i, j] for i in range(num_users) for j in range(num_items)])
    ratings = np.random.rand(num_users * num_items)  # Genera valutazioni casuali
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(user_item_pairs, ratings, epochs=10, batch_size=32)

    # Suggerire articoli
    pairs_to_predict = np.array([[user_id, item_id] for item_id in item_ids])
    predictions = model.predict(pairs_to_predict).flatten().tolist()

    # Restituisce i suggerimenti ordinati per punteggio
    suggested_products = sorted(zip(item_ids, predictions), key=lambda x: x[1], reverse=True)[:5]

    return jsonify({"suggested_products": suggested_products})