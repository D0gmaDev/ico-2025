print("✅ QLearning.py démarré")
import  numpy as np

def init_q_table(num_states, num_actions):
    """
    Initialise la table Q avec des zéros.
    """
    return np.zeros((num_states, num_actions))

def epsilon_greedy(state, epsilon, Q):
    """
    Epsilon-greedy pour choisir une action.
    """
    p = np.random.rand()
    if p < epsilon:
        return np.random.choice(Q.shape[1])  # Action aléatoire (exploration)
    else:
        return np.argmax(Q[state])  # Meilleure action connue (exploitation)
            
def update_q(Q, state, action, reward, next_state, alpha, gamma):
    """
    Met à jour la Q-table selon la règle du Q-learning.
    """
    best_next = np.max(Q[next_state])
    Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

# Paramètres
num_states = 100   # dépend de comment tu encodes les états
num_actions = 4    # nombre de mutations
alpha = 0.1
gamma = 0.9
epsilon = 0.1

Q = init_q_table(num_states, num_actions)

# Supposons une itération
state = 10
action = epsilon_greedy(state, epsilon, Q)
reward = 2.5
next_state = 11
update_q(Q, state, action, reward, next_state, alpha, gamma)

