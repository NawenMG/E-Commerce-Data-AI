from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf

# Blueprint per l'analisi delle immagini
image_analysis_bp = Blueprint('image_analysis', __name__)

def create_cnn_model(input_shape):
    """
    Crea un modello CNN per analizzare le immagini di prodotti.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # 10 classi di output per classificazione
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Inizializza il modello CNN
cnn_model = create_cnn_model((64, 64, 3))

# Addestramento simulato (puoi sostituirlo con un modello pre-addestrato)
x_train = np.random.rand(100, 64, 64, 3)  # Dati di esempio
y_train = np.random.rand(100, 10)         # Etichette di esempio
cnn_model.fit(x_train, y_train, epochs=10, batch_size=32)  # Addestra il modello

# ============================
# Route per l'analisi delle immagini
# ============================
@image_analysis_bp.route('/analyze_image', methods=['POST'])
def analyze_image():
    """
    Route per l'analisi delle immagini di prodotti.
    """
    data = request.json

    if 'image_data' not in data:
        return jsonify({"error": "Dati immagine non forniti."}), 400
    
    # Supponiamo che l'immagine sia rappresentata come una lista piatta di pixel
    image_data = np.array(data['image_data']).reshape((64, 64, 3))  # Forma esempio (64x64x3)

    # Prevedi l'immagine caricata
    prediction = cnn_model.predict(np.expand_dims(image_data, axis=0))
    
    return jsonify({"prediction": prediction.tolist()})

# ============================
# Route per addestrare il modello CNN
# ============================
@image_analysis_bp.route('/train_cnn', methods=['POST'])
def train_cnn():
    """
    Route per addestrare il modello CNN con i nuovi dati forniti.
    """
    try:
        # Ricevi i nuovi dati per l'addestramento
        training_data = request.json.get('training_data')
        training_labels = request.json.get('training_labels')

        if not training_data or not training_labels:
            return jsonify({"error": "Dati di addestramento o etichette non forniti."}), 400

        # Supponiamo che i dati di addestramento siano una lista di liste di pixel
        x_train = np.array(training_data).reshape(-1, 64, 64, 3)  # Forma (n_samples, 64, 64, 3)
        y_train = np.array(training_labels)  # Etichette di esempio

        # Addestra il modello con i nuovi dati
        cnn_model.fit(x_train, y_train, epochs=10, batch_size=32)

        return jsonify({"message": "Modello CNN addestrato con successo!"})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
