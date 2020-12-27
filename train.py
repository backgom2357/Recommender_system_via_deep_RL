#Dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import time

from envs import OfflineEnv
from recommender import DRRAgent

import os
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m/')
STATE_SIZE = 10
MAX_EPISODE_NUM = 1000000

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == "__main__":

    print('Data loading...')

    #Loading datasets
    ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'ratings.dat'), 'r').readlines()]
    users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'users.dat'), 'r').readlines()]
    movies_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'movies.dat'),encoding='latin-1').readlines()]
    ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = np.uint32)
    movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
    movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

    print("Data loading complete!")
    print("Data preprocessing...")

    # 영화 id를 영화 제목으로
    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}
    ratings_df = ratings_df.applymap(int)

    # 유저별로 본 영화들 순서대로 정리
    users_dict = {user : [] for user in set(ratings_df["UserID"])}

    # 시간 순으로 정렬하기
    ratings_df = ratings_df.sort_values(by='Timestamp', ascending=True)

    # 유저 딕셔너리에 (영화, 평점)쌍 넣기
    ratings_df_gen = ratings_df.iterrows()
    for data in ratings_df_gen:
        users_dict[data[1]['UserID']].append((data[1]['MovieID'], data[1]['Rating']))

    # 각 유저별 영화 히스토리 길이
    users_history_lens = [len(users_dict[u]) for u in set(ratings_df["UserID"])]

    users_num = max(ratings_df["UserID"])+1
    items_num = max(ratings_df["MovieID"])+1

    # Training setting
    train_users_num = int(users_num * 0.8)
    train_items_num = items_num
    train_users_dict = {k:users_dict[k] for k in range(1, train_users_num+1)}
    train_users_history_lens = users_history_lens[:train_users_num]
    
    print('DONE!')
    time.sleep(2)

    env = OfflineEnv(train_users_dict, train_users_history_lens, movies_id_to_movies, STATE_SIZE, fix_user_id=None)
    recommender = DRRAgent(env, users_num, items_num, STATE_SIZE)
    recommender.actor.build_networks()
    recommender.critic.build_networks()
    recommender.train(MAX_EPISODE_NUM, load_model=True, wandb=True)