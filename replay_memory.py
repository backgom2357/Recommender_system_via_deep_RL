import numpy as np

class ReplayMemory(object):

    '''
    apply PER, later
    '''

    def __init__(self, replay_memory_size, embedding_dim, state_size):
        self.rm_size = replay_memory_size
        self.crt_idx = 0
        
        '''
            user_id : (1,), 
            items_ids : (10,) 변할 수 잇음, 
            actions : (1,), 
            rewards : (1,), 
            next_items_ids : (10,), 
            dones : (1,)
        '''
    
        self.users_ids = np.zeros((replay_memory_size, 1), dtype=np.uint8)
        self.items_ids = np.zeros((replay_memory_size, state_size), dtype=np.uint8)
        self.actions = np.zeros((replay_memory_size, embedding_dim), dtype=np.float32)
        self.rewards = np.zeros((replay_memory_size, 1), dtype=np.uint8)
        self.rewards[-1] = 777
        self.next_items_ids = np.zeros((replay_memory_size, state_size), dtype=np.uint8)
        self.dones = np.zeros(replay_memory_size, np.bool)

    def is_full(self):
        return self.rewards[-1] != 777

    def append(self, user_id, items_ids, action, rewards, next_items_ids, done):
        self.users_ids[self.crt_idx] = user_id
        self.items_ids[self.crt_idx] = items_ids
        self.actions[self.crt_idx] = action
        self.rewards[self.crt_idx] = rewards
        self.next_items_ids[self.crt_idx] = next_items_ids
        self.dones[self.crt_idx] = done

        self.crt_idx = (self.crt_idx + 1) % self.rm_size

    def sample(self, batch_size):
        rd_idx = np.random.choice((1 - self.is_full())*self.crt_idx + self.is_full()*self.rm_size-1, batch_size)
        batch_users_ids = self.users_ids[rd_idx]
        batch_items_ids = self.items_ids[rd_idx]
        batch_actions = self.actions[rd_idx]
        batch_rewards = self.rewards[rd_idx]
        batch_next_items_ids = self.next_items_ids[rd_idx]
        batch_dones = self.dones[rd_idx]

        return batch_users_ids, batch_items_ids, batch_actions, batch_rewards, batch_next_items_ids, batch_dones