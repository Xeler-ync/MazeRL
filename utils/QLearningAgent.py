import numpy as np


class QLearningAgent:
    """Tabular Q-learning with ε-greedy policy."""

    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        # Q-table: rows x cols x actions
        self.q_table = np.zeros((env.rows, env.cols, len(env.actions)))

    def get_action(self, state, greedy=False):
        """
        Select action using ε-greedy.
        If greedy=True, always choose best action (used for testing).
        """
        if not greedy and np.random.random() < self.epsilon:
            return np.random.choice(self.env.actions)
        else:
            row, col = state
            q_vals = self.q_table[row, col, :]
            max_q = np.max(q_vals)
            # In case multiple actions share the max value, choose randomly among them
            best_actions = [a for a in self.env.actions if q_vals[a] == max_q]
            return np.random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """Q-learning update rule."""
        row, col = state
        next_row, next_col = next_state
        current_q = self.q_table[row, col, action]

        if done:
            td_target = reward
        else:
            max_next_q = np.max(self.q_table[next_row, next_col, :])
            td_target = reward + self.gamma * max_next_q

        self.q_table[row, col, action] += self.lr * (td_target - current_q)

    def save_q_table(self, filename):
        """Save Q-table to a numpy .npy file."""
        np.save(filename, self.q_table)
        print(f"Q-table saved to {filename}")

    def load_q_table(self, filename):
        """Load Q-table from a numpy .npy file."""
        self.q_table = np.load(filename)
        print(f"Q-table loaded from {filename}")
