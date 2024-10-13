from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf

# Blueprint per il Collaborative Filtering
collaborative_filtering_bp = Blueprint('collaborative_filtering', __name__)

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

@collaborative_filtering_bp.route('/recommendations', methods=['POST'])
def get_recommendations():
    """
    Route per ottenere raccomandazioni basate sul comportamento passato degli utenti.
    """
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

    # Previsione delle valutazioni per i prodotti non ancora valutati da un utente
    user_id = 0  # Esempio di utente per cui ottenere raccomandazioni
    item_ids = np.arange(num_items)  # ID di tutti gli articoli
    pairs_to_predict = np.array([[user_id, item_id] for item_id in item_ids])

    predictions = model.predict(pairs_to_predict).flatten().tolist()

    # Restituisce le raccomandazioni ordinate per punteggio
    recommended_items = sorted(zip(item_ids, predictions), key=lambda x: x[1], reverse=True)[:5]

    return jsonify({"recommended_items": recommended_items})
