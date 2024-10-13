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

    # Crea il modello CNN
    model = create_cnn_model((64, 64, 3))

    # Addestramento simulato (puoi sostituirlo con un modello pre-addestrato)
    x_train = np.random.rand(100, 64, 64, 3)  # Dati di esempio
    y_train = np.random.rand(100, 10)         # Etichette di esempio
    model.fit(x_train, y_train, epochs=10, batch_size=32)  # Addestra il modello

    # Prevedi l'immagine caricata
    prediction = model.predict(np.expand_dims(image_data, axis=0))
    
    return jsonify({"prediction": prediction.tolist()})
