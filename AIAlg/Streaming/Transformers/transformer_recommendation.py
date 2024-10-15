from flask import Blueprint, request, jsonify
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout, Embedding
import numpy as np

# Definizione del Blueprint
transformer_recommendation_bp = Blueprint('transformer_recommendation_bp', __name__)

# ============================
# Transformer Model Definition
# ============================
def create_transformer_model(input_shape, num_heads=4, dff=128, d_model=64, num_layers=2):
    """
    Crea un modello Transformer per le raccomandazioni personalizzate.
    """
    inputs = Input(shape=input_shape)

    # Embedding per la rappresentazione del comportamento utente o recensioni prodotti
    embedding_layer = Embedding(input_dim=10000, output_dim=d_model)(inputs)

    # Aggiungi strati Transformer
    x = embedding_layer
    for _ in range(num_layers):
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output + x)
        
        ff_output = Dense(dff, activation='relu')(attention_output)
        ff_output = Dense(d_model)(ff_output)
        x = LayerNormalization(epsilon=1e-6)(ff_output + attention_output)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    
    # Output layer for recommendation score
    outputs = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Inizializza il modello Transformer
input_shape = (100,)  # Ad esempio, 100 sequenze del comportamento utente o recensioni
transformer_model = create_transformer_model(input_shape)

# ============================
# Route per la raccomandazione personalizzata
# ============================
@transformer_recommendation_bp.route('/recommend', methods=['POST'])
def recommend():
    """
    Prevedi raccomandazioni personalizzate basate sul comportamento degli utenti o recensioni dei prodotti.
    """
    try:
        # Riceve i dati di input (ad esempio, comportamento utente o recensioni)
        data = request.json['data']
        input_data = np.array(data).reshape(-1, 100)  # Reshape per adattarsi all'input del modello
        
        # Esegue la previsione
        recommendation_scores = transformer_model.predict(input_data)
        
        # Interpreta la raccomandazione come 1 (raccomandato) o 0 (non raccomandato)
        recommendations = [1 if score > 0.5 else 0 for score in recommendation_scores]
        
        return jsonify({'recommendations': recommendations})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ============================
# Route per addestrare il modello Transformer
# ============================
@transformer_recommendation_bp.route('/train_model', methods=['POST'])
def train_transformer():
    """
    Addestra il modello Transformer sui dati di comportamento o recensioni.
    """
    try:
        # Riceve i dati di addestramento
        x_train = np.array(request.json['x_train']).reshape(-1, 100)
        y_train = np.array(request.json['y_train'])  # Ad esempio: 1 per raccomandato, 0 per non raccomandato
        
        # Addestra il modello Transformer
        transformer_model.fit(x_train, y_train, epochs=10, batch_size=32)
        
        return jsonify({'message': 'Modello addestrato con successo!'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
