import numpy as np

class OfflineEnv(object):
    
    def __init__(self, users_dict, users_history_len, movies_id_to_movies, state_size, user_id=None):
        
        self.users_dict = users_dict
        self.users_history_len = users_history_len
        self.items_id_to_name = movies_id_to_movies
        
        self.state_size = state_size
        self.available_users = self._generate_available_users()
        
        self.user = user_id if user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        self.done_count = 3000
        
    def _generate_available_users(self):
        available_users = []
        for i, length in enumerate(self.users_history_len):
            if length > self.state_size:
                available_users.append(i+1)
        return available_users
    
    def reset(self, user_id=None):
        self.user = user_id if user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        return self.user, self.items, self.done
        
    def step(self, action):

        reward = 0
        
        if action in self.user_items.keys() and action not in self.recommended_items:
            reward = self.user_items[action] - 2  # reward
            del self.user_items[action]
        
        if reward > 0:
            self.items = self.items[1:] + [action]
        
        self.recommended_items.add(action)
        if self.user_items == {} or len(self.recommended_items) > self.done_count:
            self.done = True
        
        return self.items, reward, self.done, self.recommended_items