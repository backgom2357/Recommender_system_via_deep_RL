import numpy as np

class OfflineEnv(object):
    
    def __init__(self, users_dict, users_history_lens, movies_id_to_movies, state_size, fix_user_id=None):

        self.users_dict = users_dict
        self.users_history_lens = users_history_lens
        self.items_id_to_name = movies_id_to_movies
        
        self.state_size = state_size
        self.available_users = self._generate_available_users()

        self.fix_user_id = fix_user_id

        self.user = fix_user_id if fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        self.done_count = 3000
        
    def _generate_available_users(self):
        available_users = []
        for i, length in zip(self.users_dict.keys(), self.users_history_lens):
            if length > self.state_size:
                available_users.append(i)
        return available_users
    
    def reset(self):
        self.user = self.fix_user_id if self.fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        return self.user, self.items, self.done
        
    def step(self, action, top_k=False):

        reward = -0.5
        
        if top_k:
            correctly_recommended = []
            rewards = []
            for act in action:
                if act in self.user_items.keys() and act not in self.recommended_items:
                    correctly_recommended.append(act)
                    rewards.append((self.user_items[act] - 3)/2)
                else:
                    rewards.append(-0.5)
                self.recommended_items.add(act)
            if max(rewards) > 0:
                self.items = self.items[len(correctly_recommended):] + correctly_recommended
            reward = rewards

        else:
            if action in self.user_items.keys() and action not in self.recommended_items:
                reward = self.user_items[action] -3  # reward
            if reward > 0:
                self.items = self.items[1:] + [action]
            self.recommended_items.add(action)

        if len(self.recommended_items) > self.done_count or len(self.recommended_items) >= self.users_history_lens[self.user-1]:
            self.done = True
            
        return self.items, reward, self.done, self.recommended_items

    def get_items_names(self, items_ids):
        items_names = []
        for id in items_ids:
            try:
                items_names.append(self.items_id_to_name[str(id)])
            except:
                items_names.append(list(['Not in list']))
        return items_names
