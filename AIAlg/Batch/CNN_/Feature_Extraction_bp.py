from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Blueprint per l'estrazione di feature
feature_extraction_bp = Blueprint('feature_extraction', __name__)

def create_feature_extraction_model(input_shape):
    """
    Crea un modello di rete neurale per l'estrazione di feature da dati strutturati.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='sigmoid')  # Output: 10 feature estratte
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Inizializza il modello di estrazione delle feature
input_shape = (3,)  # Supponiamo che ci siano 3 feature nei dati strutturati
feature_model = create_feature_extraction_model(input_shape)

# Addestramento simulato (puoi sostituirlo con un modello pre-addestrato)
x_train = np.random.rand(100, input_shape[0])  # Dati di esempio
y_train = np.random.rand(100, 10)  # Etichette di esempio
feature_model.fit(x_train, y_train, epochs=10, batch_size=32)  # Addestra il modello

# ============================
# Route per l'estrazione di feature
# ============================
@feature_extraction_bp.route('/extract_features', methods=['POST'])
def extract_features():
    """
    Route per l'estrazione di feature da dati strutturati.
    """
    data = request.json

    if 'structured_data' not in data:
        return jsonify({"error": "Dati strutturati non forniti."}), 400
    
    # Supponiamo che i dati strutturati siano una lista di liste (simile a un DataFrame)
    structured_data = np.array(data['structured_data'])
    
    # Standardizzazione dei dati
    scaler = StandardScaler()
    structured_data_scaled = scaler.fit_transform(structured_data)

    # Estrai le feature dai dati strutturati
    extracted_features = feature_model.predict(structured_data_scaled).tolist()
    
    return jsonify({"extracted_features": extracted_features})

# ============================
# Route per addestrare il modello di estrazione di feature
# ============================
@feature_extraction_bp.route('/train_feature_extraction', methods=['POST'])
def train_feature_extraction():
    """
    Route per addestrare il modello di estrazione di feature sui dati forniti.
    """
    try:
        # Ricevi i nuovi dati per l'addestramento
        new_data = request.json.get('structured_data')

        if not new_data:
            return jsonify({"error": "Nessun dato fornito per l'addestramento."}), 400

        # Supponiamo che i dati strutturati siano una lista di liste (simile a un DataFrame)
        new_data_array = np.array(new_data)

        # Standardizzazione dei nuovi dati
        scaler = StandardScaler()
        new_data_scaled = scaler.fit_transform(new_data_array)

        # Genera le etichette di esempio per l'addestramento
        # In un caso reale, dovresti avere le etichette corrette per l'addestramento
        y_train = np.random.rand(new_data_scaled.shape[0], 10)  # 10 feature estratte

        # Addestra il modello con i nuovi dati
        feature_model.fit(new_data_scaled, y_train, epochs=10, batch_size=32)

        return jsonify({"message": "Modello di estrazione di feature addestrato con successo!"})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
