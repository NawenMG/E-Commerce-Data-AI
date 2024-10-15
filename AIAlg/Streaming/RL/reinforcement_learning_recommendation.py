from flask import Blueprint, request, jsonify
import numpy as np
import random

# Definisci il Blueprint per RL
rl_recommendation_bp = Blueprint('reinforcement_learning_recommendation', __name__)

class SimpleReinforcementLearning:
    def __init__(self, num_products):
        self.num_products = num_products
        self.q_table = np.zeros(num_products)  # Q-value table per i prodotti
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995

    def recommend(self):
        """
        Raccomanda un prodotto utilizzando la strategia epsilon-greedy.
        """
        if random.random() < self.exploration_rate:
            return random.randint(0, self.num_products - 1)  # Esplora
        else:
            return np.argmax(self.q_table)  # Sfrutta la conoscenza attuale

    def update_q_table(self, product_index, reward):
        """
        Aggiorna la Q-table basata sul feedback ricevuto.
        """
        self.q_table[product_index] += self.learning_rate * (reward + self.discount_factor * np.max(self.q_table) - self.q_table[product_index])
        # Riduci il tasso di esplorazione
        self.exploration_rate *= self.exploration_decay

# Inizializza il modello di apprendimento per rinforzo
num_products = 10  # Esempio: 10 prodotti diversi
rl_model = SimpleReinforcementLearning(num_products)

@rl_recommendation_bp.route('/recommend', methods=['GET'])
def recommend_product():
    """
    Endpoint per raccomandare un prodotto.
    """
    product_index = rl_model.recommend()
    return jsonify({"recommended_product": product_index})

@rl_recommendation_bp.route('/feedback', methods=['POST'])
def provide_feedback():
    """
    Endpoint per ricevere feedback sulle raccomandazioni.
    """
    data = request.get_json()
    product_index = data['product_index']
    reward = data['reward']  # Ricevi un feedback in forma di ricompensa (0 o 1)

    # Aggiorna la Q-table con il feedback ricevuto
    rl_model.update_q_table(product_index, reward)
    return jsonify({"message": "Feedback received and Q-table updated"})
