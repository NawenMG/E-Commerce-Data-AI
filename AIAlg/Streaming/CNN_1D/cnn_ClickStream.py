from flask import Blueprint, request, jsonify
import tensorflow as tf
import numpy as np

# Definizione del Blueprint
cnn_clickstream_bp = Blueprint('cnn_clickstream_bp', __name__)

# ============================
# CNN 1D Model Definition
# ============================
def create_cnn_1d_model(input_shape):
    """
    Crea un modello CNN 1D per la previsione di eventi clickstream
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output per la previsione binaria (ad es. clic o non clic)
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Inizializza il modello CNN 1D
input_shape = (100, 1)  # Ad esempio: 100 eventi clickstream come feature
cnn_model = create_cnn_1d_model(input_shape)

# ============================
# Route per la previsione di eventi clickstream
# ============================
@cnn_clickstream_bp.route('/predict_clickstream', methods=['POST'])
def predict_clickstream():
    """
    Prevedi eventi clickstream (ad esempio, se un utente cliccherà su un prodotto)
    """
    try:
        # Riceve i dati di input (ad esempio, sequenza di eventi clickstream)
        data = request.json['data']
        input_data = np.array(data).reshape(-1, 100, 1)  # Reshape per adattarsi all'input del modello
        
        # Esegue la previsione
        prediction = cnn_model.predict(input_data)
        
        # Interpreta la previsione come click (1) o non click (0)
        clicks = [1 if pred > 0.5 else 0 for pred in prediction]  # Threshold 0.5 come esempio
        
        return jsonify({'click_predictions': clicks})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ============================
# Route per addestrare il modello CNN 1D
# ============================
@cnn_clickstream_bp.route('/train_cnn', methods=['POST'])
def train_cnn():
    """
    Addestra il modello CNN 1D sui dati forniti per prevedere eventi clickstream
    """
    try:
        # Riceve i dati di addestramento
        x_train = np.array(request.json['x_train']).reshape(-1, 100, 1)
        y_train = np.array(request.json['y_train'])  # Ad esempio: 1 per clic, 0 per non clic
        
        # Addestra il modello CNN 1D
        cnn_model.fit(x_train, y_train, epochs=10, batch_size=32)
        
        return jsonify({'message': 'Modello addestrato con successo!'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
