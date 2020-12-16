import tensorflow as tf
import numpy as np

from state_reps import *

class ActorNetwork(tf.keras.Model):
    def __init__(self, embedding_dim, hidden_dim):
        super(ActorNetwork, self).__init__()
        self.srm = DRRAveStateRepresentation(embedding_dim)
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(embedding_dim, activation='tanh')
        ])
    def call(self, user_eb, items_ebs):
        state = self.srm(user_eb, items_ebs)
        return self.fc(state), state

'''
Actor를 설정할 때 2가지 방법으로 나뉨
    1. include embedding layer
    2. not include embedding layer

    I use first method here.
'''

# Include embedding layer
## state is (user_id, tiems_id)

class Actor(object):
    
    def __init__(self, users_num, items_num, embedding_dim, hidden_dim, learning_rate, tau):
        
        self.items_num = items_num
        
        # 임베딩 네트워크 embedding network
        self.embedding_network = Embedding(users_num, items_num, embedding_dim)
        # 엑터 네트워크 actor network / 타겟 네트워크 target network
        self.network = ActorNetwork(embedding_dim, hidden_dim)
        self.target_network = ActorNetwork(embedding_dim, hidden_dim)
        # 옵티마이저 optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        # 소프트 타겟 네트워크 업데이트 하이퍼파라미터 soft target network update hyperparameter
        self.tau = tau
        
    def get_action(self, weight, items_ids=None):
        if items_ids == None:
            items_ids = [i for i in range(self.items_num)]
        items_ebs = self.embedding_network._get_item_eb(items_ids)
        item_idx = np.argmax(tf.keras.backend.dot(items_ebs, weight))
        return items_ids[item_idx]
    
    def update_target_network(self):
        # 소프트 타겟 네트워크 업데이트 soft target network update
        c_theta, t_theta = self.network.get_weights(), self.target_network.get_weights()
        for i in range(len(c_theta)):
            t_theta[i] = self.tau * c_theta[i] + (1 - self.tau) * t_theta[i]
        self.target_network.set_weights(t_theta)
        
    def train(self, user_id, items_ids, values, dq_das):
        with tf.GradientTape() as g:
            user_eb, items_ebs = self.embedding_network(user_id, items_ids)
            weight, _ = self.network(user_eb, items_ebs)
            obj = np.mean(values * weight, axis=1)
        dj_dtheta = g.gradient(obj, self.network.trainable_weights, -dq_das)
        grads = zip(dj_dtheta, self.network.trainable_weights)
        self.optimizer.apply_gradients(grads)
        
    def save_weights(self, path):
        self.network.save_weights(path)
        
    def load_weights(self, path):
        self.network.load_weights(path)