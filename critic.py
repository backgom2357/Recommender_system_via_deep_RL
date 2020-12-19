import tensorflow as tf
import numpy as np

class CriticNetwork(tf.keras.Model):
    def __init__(self, hidden_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.concat = tf.keras.layers.Concatenate()
        self.fc2 = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation = 'relu'),
            tf.keras.layers.Dense(1, activation = 'relu')
        ])
        
    def call(self, state, action):
        fc1 = self.fc1(state)
        concat = self.concat([action, fc1])
        return self.fc2(concat)

class Critic(object):
    
    def __init__(self, hidden_dim, learning_rate, embedding_dim, tau):
        
        self.embedding_dim = embedding_dim

        # 크리틱 네트워크 critic network / 타겟 네트워크 target network
        self.network = CriticNetwork(hidden_dim)
        self.target_network = CriticNetwork(hidden_dim)
        # 옵티마이저 optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # MSE
        self.loss = tf.keras.losses.MeanSquaredError()

        # 소프트 타겟 네트워크 업데이트 하이퍼파라미터 soft target network update hyperparameter
        self.tau = tau

    def build_networks(self):
        self.network(np.zeros((1,1,3*self.embedding_dim)), np.zeros((1,1,self.embedding_dim)))
        self.target_network(np.zeros((1,1,3*self.embedding_dim)), np.zeros((1,1,self.embedding_dim)))

    def update_target_network(self):
        c_omega = self.network.get_weights()
        t_omega = self.target_network.get_weights()
        for i in range(len(c_omega)):
            t_omega[i] = self.tau * c_omega[i] + (1 - self.tau) * t_omega[i]
        self.target_network.set_weights(t_omega)
        
    def dq_da(self, states, actions):
        with tf.GradientTape() as g:
            actions = tf.convert_to_tensor(actions)
            g.watch(actions)
            outputs = self.network(states, actions)
        q_grads = g.gradient(outputs, actions)
        return q_grads
    
    def train(self, states, actions, td_targets):
        with tf.GradientTape() as g:
            output = self.network(states, actions)
            loss = self.loss(td_targets, output)
        g_omega = g.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(g_omega, self.network.trainable_weights))

    def train_on_batch(self, states, actions, td_targets):
        self.network.train_on_batch([states, actions], td_targets)
            
    def save_weights(self, path):
        self.network.save_weights(path)
        
    def load_weights(self, path):
        self.network.load_weights(path)