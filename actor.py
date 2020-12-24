import tensorflow as tf
import numpy as np

class Embedding(tf.keras.Model):
    def __init__(self, users_num, items_num, embedding_dim):
        super(Embedding, self).__init__()
        self.user_eb = tf.keras.layers.Embedding(users_num, embedding_dim)
        self.item_eb = tf.keras.layers.Embedding(items_num, embedding_dim)
        
    def call(self, user_id, items_ids):
        return self.user_eb(user_id), self.item_eb(items_ids)
    
    def get_item_eb(self, items_ids):
        return self.item_eb(items_ids)

class DRRAveStateRepresentation(tf.keras.layers.Layer):
    def __init__(self, embedding_dim):
        super(DRRAveStateRepresentation, self).__init__()
        self.embedding_dim = embedding_dim
        self.wav = tf.keras.layers.Conv1D(1, 1, 1)
        self.concat = tf.keras.layers.Concatenate()
        self.flatten = tf.keras.layers.Flatten()
        
    def call(self, user_eb, items_eb):
        items_eb = tf.transpose(items_eb, perm=(0,2,1))/self.embedding_dim
        wav = self.wav(items_eb)
        wav = tf.transpose(wav, perm=(0,2,1))
        user_wav = tf.keras.layers.multiply([user_eb, wav])
        concat = self.concat([user_eb, user_wav, wav])
        return self.flatten(concat)

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
## (user_id, tiems_id)

class Actor(object):
    
    def __init__(self, users_num, items_num, embedding_dim, hidden_dim, learning_rate, state_size, tau):
        
        self.items_num = items_num
        self.embedding_dim = embedding_dim
        self.state_size = state_size
        
        # 임베딩 네트워크 embedding network
        self.embedding_network = Embedding(users_num, items_num, embedding_dim)
        # 엑터 네트워크 actor network / 타겟 네트워크 target network
        self.network = ActorNetwork(embedding_dim, hidden_dim)
        self.target_network = ActorNetwork(embedding_dim, hidden_dim)
        # 옵티마이저 optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        # 소프트 타겟 네트워크 업데이트 하이퍼파라미터 soft target network update hyperparameter
        self.tau = tau
        # ε-탐욕 탐색 하이퍼파라미터 ε-greedy exploration hyperparameter
        self.epsilon = 1.
        self.epsilon_decay = (self.epsilon - 0.1)/1000000
    
    def build_networks(self):
        # 네트워크들 빌딩 / Build networks
        self.embedding_network(np.zeros(1),np.zeros(1))
        self.network(np.zeros((1,1,self.embedding_dim)),np.zeros((1,self.state_size,self.embedding_dim)))
        self.target_network(np.zeros((1,1,self.embedding_dim)),np.zeros((1,self.state_size,self.embedding_dim)))
        

    def recommend_item(self, action, recommended_items, top_k=False, items_ids=None, is_test=False):
        if items_ids == None:
            items_ids = np.array(list(set(i for i in range(self.items_num)) - recommended_items))
        
        # ε-greedy exploration
        if self.epsilon > np.random.uniform() and not is_test:
            self.epsilon -= self.epsilon_decay
            if top_k:
                return np.random.choice(items_ids, top_k)
            return np.random.choice(items_ids)

        items_ebs = self.embedding_network.get_item_eb(items_ids)
        action = tf.transpose(action, perm=(1,0))
        if top_k:
            item_indice = np.argsort(tf.transpose(tf.keras.backend.dot(items_ebs, action), perm=(1,0)))[0][-top_k:]
            return items_ids[item_indice]
        else:    
            item_idx = np.argmax(tf.keras.backend.dot(items_ebs, action))
            return items_ids[item_idx]
    
    def update_target_network(self):
        # 소프트 타겟 네트워크 업데이트 soft target network update
        c_theta, t_theta = self.network.get_weights(), self.target_network.get_weights()
        for i in range(len(c_theta)):
            t_theta[i] = self.tau * c_theta[i] + (1 - self.tau) * t_theta[i]
        self.target_network.set_weights(t_theta)
        
    def train(self, user_id, items_ids, dq_das):
        with tf.GradientTape() as g:
            user_eb, items_ebs = self.embedding_network(user_id, items_ids)
            outputs, _ = self.network(user_eb, items_ebs)
        dj_dtheta = g.gradient(outputs, self.network.trainable_weights, -dq_das)
        grads = zip(dj_dtheta, self.network.trainable_weights)
        self.optimizer.apply_gradients(grads)
        
    def save_weights(self, path):
        self.network.save_weights(path)
        
    def load_weights(self, path):
        self.network.load_weights(path)