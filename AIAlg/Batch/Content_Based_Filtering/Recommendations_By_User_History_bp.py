from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Blueprint per il Content-Based Filtering
recommendations_by_user_history_bp = Blueprint('content_based_filtering', __name__)

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

# Inizializza il TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Calcola il TF-IDF e la matrice di similarità
def fit_tfidf():
    tfidf_matrix = vectorizer.fit_transform(products_data['description'])
    return cosine_similarity(tfidf_matrix)

similarity_matrix = fit_tfidf()

@recommendations_by_user_history_bp.route('/recommendations_by_user_history', methods=['POST'])
def get_recommendations_by_user_history():
    """
    Route per ottenere raccomandazioni basate sulle preferenze storiche degli utenti.
    """
    data = request.json

    if 'user_id' not in data or 'user_history' not in data:
        return jsonify({"error": "ID utente o cronologia non forniti."}), 400
    
    user_id = data['user_id']
    user_history = data['user_history']  # Esempio: lista di ID di prodotti visti/acquistati dall'utente

    # Calcola la media delle caratteristiche di prodotti già visti
    seen_products = products_data[products_data['product_id'].isin(user_history)]

    if seen_products.empty:
        return jsonify({"error": "Nessun prodotto trovato nella cronologia."}), 404

    # Ottieni la media del TF-IDF delle descrizioni dei prodotti visti
    seen_tfidf = vectorizer.transform(seen_products['description'])
    average_tfidf = seen_tfidf.mean(axis=0)

    # Calcola la similarità con tutti i prodotti
    similarity_scores = cosine_similarity(average_tfidf, vectorizer.transform(products_data['description']))

    # Ottieni i prodotti consigliati in base alla media delle preferenze
    recommended_indices = similarity_scores.flatten().argsort()[-6:-1][::-1]
    recommended_products = products_data.iloc[recommended_indices]['product_id'].tolist()

    return jsonify({"recommended_products": recommended_products})