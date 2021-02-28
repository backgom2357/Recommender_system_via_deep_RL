import numpy as np
from sum_tree import SumTree

class PriorityExperienceReplay(object):

    '''
    apply PER, later
    '''

    def __init__(self, buffer_size, embedding_dim):
        self.buffer_size = buffer_size
        self.crt_idx = 0
        
        '''
            state : (300,), 
            next_state : (300,) 변할 수 잇음, 
            actions : (100,), 
            rewards : (1,), 
            dones : (1,)
        '''
        self.states = np.zeros((buffer_size, 3*embedding_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, embedding_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, 3*embedding_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, np.bool)

        self.sum_tree = SumTree(buffer_size)
        self.max_prioirty = 1.0
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_constant = 0.00001

    def append(self, state, action, reward, next_state, done):
        self.states[self.crt_idx] = state
        self.actions[self.crt_idx] = action
        self.rewards[self.crt_idx] = reward
        self.next_states[self.crt_idx] = next_state
        self.dones[self.crt_idx] = done

        self.sum_tree.add_data(self.max_prioirty)
        self.crt_idx = (self.crt_idx + 1) if (self.crt_idx + 1) < self.buffer_size else self.buffer_size

    def sample(self, batch_size):
        rd_idx = []
        weight_batch = []
        index_batch = []
        sum_priority = self.sum_tree.sum_all_prioirty()
        




        batch_states = self.states[rd_idx]
        batch_actions = self.actions[rd_idx]
        batch_rewards = self.rewards[rd_idx]
        batch_next_states = self.next_states[rd_idx]
        batch_dones = self.dones[rd_idx]

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones