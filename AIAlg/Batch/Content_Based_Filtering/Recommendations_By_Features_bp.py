from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Blueprint per il Content-Based Filtering
recommendations_by_features_bp = Blueprint('content_based_filtering', __name__)

# Simuliamo un dataset di prodotti
# Questo dovrebbe essere caricato da un database o da un file in un'applicazione reale
products_data = pd.DataFrame({
    'product_id': [0, 1, 2, 3, 4],
    'title': [
        'Smartphone con fotocamera 108MP',
        'Laptop da gioco con scheda grafica RTX 3060',
        'Cuffie Bluetooth con cancellazione del rumore',
        'Smartwatch con monitoraggio della salute',
        'Tablet con schermo Retina'
    ],
    'description': [
        'Ultimo smartphone con una potente fotocamera per fotografie eccezionali.',
        'Laptop ideale per i videogiocatori con prestazioni elevate.',
        'Cuffie di alta qualità con suono surround e cancellazione del rumore.',
        'Smartwatch elegante che tiene traccia della tua salute e attività.',
        'Tablet leggero e potente con uno schermo ad alta risoluzione.'
    ]
})

# Inizializza il TF-IDF Vectorizer e la matrice di similarità
vectorizer = TfidfVectorizer()
similarity_matrix = None

def fit_tfidf():
    """
    Calcola il TF-IDF e la matrice di similarità per i prodotti.
    """
    global similarity_matrix
    tfidf_matrix = vectorizer.fit_transform(products_data['description'])
    similarity_matrix = cosine_similarity(tfidf_matrix)

# Addestra il modello inizialmente
fit_tfidf()

# ============================
# Route per aggiornare il modello
# ============================
@recommendations_by_features_bp.route('/train_model', methods=['POST'])
def train_model():
    """
    Route per aggiornare il modello con nuovi dati.
    """
    global products_data
    
    # Ottieni nuovi dati
    new_data = request.json.get('products')

    if not new_data:
        return jsonify({"error": "Nessun dato fornito per l'addestramento."}), 400

    # Converti in DataFrame e aggiorna il dataset dei prodotti
    new_products_df = pd.DataFrame(new_data)
    products_data = pd.concat([products_data, new_products_df], ignore_index=True)

    # Riaddestra il modello
    fit_tfidf()
    
    return jsonify({"message": "Modello aggiornato con successo."})

# ============================
# Route per ottenere raccomandazioni
# ============================
@recommendations_by_features_bp.route('/recommendations_by_features', methods=['POST'])
def get_recommendations_by_features():
    """
    Route per ottenere raccomandazioni basate sulle caratteristiche dei prodotti.
    """
    data = request.json

    if 'product_id' not in data:
        return jsonify({"error": "ID prodotto non fornito."}), 400
    
    product_id = data['product_id']

    # Trova l'indice del prodotto
    product_idx = products_data[products_data['product_id'] == product_id].index

    if product_idx.empty:
        return jsonify({"error": "ID prodotto non trovato."}), 404

    # Ottieni le raccomandazioni
    similar_indices = similarity_matrix[product_idx].argsort().flatten()[-6:-1][::-1]
    recommended_products = products_data.iloc[similar_indices]['product_id'].tolist()

    return jsonify({"recommended_products": recommended_products})
