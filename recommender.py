import tensorflow as tf
import numpy as np

from actor import Actor
from critic import Critic
from replay_memory import ReplayMemory

class DRRAgent:
    
    def __init__(self, users_num, items_num, state_size):

        self.users_num = users_num
        self.items_num = items_num
        
        self.embedding_dim = 16
        self.actor_hidden_dim = 32
        self.actor_learning_rate = 0.001
        self.critic_hidden_dim = 32
        self.critic_learning_rate = 0.001
        self.discount_factor = 0.9
        self.tau = 0.001

        self.replay_memory_size = 100000
        
        self.actor = Actor(users_num, items_num, self.embedding_dim, self.actor_hidden_dim, self.actor_learning_rate, self.tau)
        self.critic = Critic(self.critic_hidden_dim, self.critic_learning_rate, self.tau)
        
        self.buffer = ReplayMemory(replay_memory_size, state_size)
        
    def ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return x + rho*(mu - x)*dt + sigma*np.sqrt(dt)*np.random.normal(size=dim)
    
    def td_target(self, rewards, q_values, dones):
        y_t = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            y_t = rewards[i] + (1 - dones[i])*(self.discount_factor * q_values[i])
        return y_t
        
    def save_model(self, actor_path, critic_path):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        
    def load_model(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)