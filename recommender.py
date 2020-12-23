import tensorflow as tf
import numpy as np

from actor import Actor
from critic import Critic
from replay_memory import ReplayMemory

import matplotlib.pyplot as plt

class DRRAgent:
    
    def __init__(self, env, users_num, items_num, state_size):
        
        self.env = env

        self.users_num = users_num
        self.items_num = items_num
        
        self.embedding_dim = 100
        self.actor_hidden_dim = 128
        self.actor_learning_rate = 0.001
        self.critic_hidden_dim = 128
        self.critic_learning_rate = 0.001
        self.discount_factor = 0.9
        self.tau = 0.001

        self.replay_memory_size = 1000000
        self.batch_size = 32
        
        self.actor = Actor(users_num, items_num, self.embedding_dim, self.actor_hidden_dim, self.actor_learning_rate, state_size, self.tau)
        self.critic = Critic(self.critic_hidden_dim, self.critic_learning_rate, self.embedding_dim, self.tau)
        
        self.buffer = ReplayMemory(self.replay_memory_size, self.embedding_dim, state_size)
    
    def calculate_td_target(self, rewards, q_values, dones):
        y_t = np.copy(q_values)
        for i in range(q_values.shape[0]):
            y_t[i] = rewards[i] + (1 - dones[i])*(self.discount_factor * q_values[i])
        return y_t
        
    def train(self, max_episode_num, load_model=False):
        # 타겟 네트워크들 초기화
        self.actor.update_target_network()
        self.critic.update_target_network()

        if load_model:
            self.load_model("/home/ubuntu/DRR/save_weights/actor_7000.h5", "/home/ubuntu/DRR/save_weights/critic_7000.h5")
            print('Completely load weights!')

        episodic_precision_history = []

        for episode in range(max_episode_num):
            # episodic reward 리셋
            episode_reward = 0
            correct_count = 0
            # Environment 리셋
            user_id, items_ids, done = self.env.reset()
            print(f'user_id : {user_id}, rated_items_length:{len(self.env.user_items)}')
            print('items : ', self.env.get_items_names(items_ids))
            while not done:
                
                # Observe current state & Find action
                ## Embedding 해주기
                user_id = tf.convert_to_tensor(user_id)
                items_ids = tf.convert_to_tensor(items_ids)
                user_eb, items_eb = self.actor.embedding_network(user_id, items_ids)
                user_eb = tf.reshape(user_eb, (1,1, *user_eb.shape))
                items_eb = tf.reshape(items_eb, (1,*items_eb.shape))
                ## Action(ranking score) 출력
                action, _ = self.actor.network(user_eb, items_eb)
                ## Item 추천
                recommended_item = self.actor.recommend_item(action, self.env.recommended_items)
                
                # Calculate reward & observe new state (in env)
                ## Step
                next_items_ids, reward, done, _ = self.env.step(recommended_item)

               # buffer에 저장
                self.buffer.append(user_id, items_ids, action, reward, next_items_ids, done)
                
                if len(self.buffer.users_ids[500]) != 0:
                    # Sample a minibatch
                    batch_users_ids, batch_items_ids, batch_actions, batch_rewards, batch_next_items_ids, batch_dones = self.buffer.sample(self.batch_size)
                    # Set TD targets
                    batch_user_ebs, batch_next_items_ebs = self.actor.embedding_network(batch_users_ids, batch_next_items_ids)
                    target_next_action, next_states = self.actor.target_network(batch_user_ebs, batch_next_items_ebs)
                    target_qs = self.critic.target_network(next_states, target_next_action)
                    td_targets = self.calculate_td_target(batch_rewards, target_qs, batch_dones)
                    # Update critic network
                    batch_items_ebs = self.actor.embedding_network.get_item_eb(batch_items_ids)
                    _, batch_states = self.actor.network(batch_user_ebs, batch_items_ebs)
                    self.critic.train(batch_states, batch_actions, td_targets)
                    # Update actor network
                    s_grads = self.critic.dq_da(batch_states, batch_actions)
                    dq_das = np.array(s_grads).reshape((-1, self.embedding_dim))
                    self.actor.train(batch_users_ids, batch_items_ids, dq_das)
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                items_ids = next_items_ids
                episode_reward += reward

                if reward > 0:
                    correct_count += 1
                
                print(f'recommended items : {len(self.env.recommended_items)}, reward : {reward:+}', end='\r')

                if done:
                    print()
                    precision = int(correct_count/len(self.env.recommended_items) * 100)
                    print(f'{episode}/{max_episode_num}, precision : {precision:2}%, total_reward:{episode_reward}')
                    episodic_precision_history.append(precision)
             
            if (episode+1)%50 == 0:
                plt.plot(episodic_precision_history)
                plt.savefig(f'episodic_reward_history')

            if (episode+1)%100 == 0:
                self.save_model(f'/home/ubuntu/DRR/save_weights/actor_{episode+1}.h5', f'/home/ubuntu/DRR/save_weights/critic_{episode+1}.h5')

    def save_model(self, actor_path, critic_path):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        
    def load_model(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)