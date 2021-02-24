import numpy as np

class ReplayMemory(object):

    '''
    apply PER, later
    '''

    def __init__(self, replay_memory_size, embedding_dim):
        self.rm_size = replay_memory_size
        self.crt_idx = 0
        
        '''
            state : (300,), 
            next_state : (300,) 변할 수 잇음, 
            actions : (100,), 
            rewards : (1,), 
            dones : (1,)
        '''

        self.states = np.zeros((replay_memory_size, 3*embedding_dim), dtype=np.float32)
        self.actions = np.zeros((replay_memory_size, embedding_dim), dtype=np.float32)
        self.rewards = np.zeros((replay_memory_size), dtype=np.float32)
        self.rewards[replay_memory_size-1] = 777
        self.next_states = np.zeros((replay_memory_size, 3*embedding_dim), dtype=np.float32)
        self.dones = np.zeros(replay_memory_size, np.bool)

    def is_full(self):
        return self.rewards[-1] != 777

    def append(self, state, action, reward, next_state, done):
        self.states[self.crt_idx] = state
        self.actions[self.crt_idx] = action
        self.rewards[self.crt_idx] = reward
        self.next_states[self.crt_idx] = next_state
        self.dones[self.crt_idx] = done

        self.crt_idx = (self.crt_idx + 1) % self.rm_size

    def sample(self, batch_size):
        rd_idx = np.random.choice((1 - self.is_full())*self.crt_idx + self.is_full()*self.rm_size-1, batch_size)
        batch_states = self.states[rd_idx]
        batch_actions = self.actions[rd_idx]
        batch_rewards = self.rewards[rd_idx]
        batch_next_states = self.next_states[rd_idx]
        batch_dones = self.dones[rd_idx]

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones