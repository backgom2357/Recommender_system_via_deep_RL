# Deep Reinforcement Learning based Recommender System in Tensorflow
The implemetation of Deep Reinforcement Learning based Recommender System from the paper [Deep Reinforcement Learning based Recommendation with Explicit User-Item Interactions Modeling](https://arxiv.org/abs/1810.12027) by Liu et al. Build recommender system with [DDPG](https://arxiv.org/abs/1509.02971) algorithm. Add state representation module to produce trainable state for RL algorithm from data.

# Dataset
[MovieLens 1M Datset](https://grouplens.org/datasets/movielens/1m/)

```
unzip ./ml-1m.zip
```

# Procedure
Trying to improve performance of RL based recommender system. The report contains the result of Using the actor network with embedding layer, reducing overestimated Q value, using several pretrained embedding and applying [PER](https://arxiv.org/abs/1511.05952).



# Result

### Please check here - [Experiment Report (Korean)](https://www.notion.so/DRR-8e910fc598d242968bd371b27ac20e01)

<br>

![image](https://user-images.githubusercontent.com/30210944/109442330-40b37180-7a7b-11eb-8303-d45a8083dbc7.png)

- for evalutation data
    - precision@5 : 0.479, ndcg@5 : 0.471
    - precision@10 : 0.444, ndcg@10 : 0.429

# Usage
### Training
```
python train.py
```
### Evalutation
Follow [evaluation.ipynb](https://github.com/backgom2357/DRR/blob/develop/evaluation.ipynb)

# requirements
```
tensorflow==2.2.0
scikit-learn==0.23.2
matplotlib==3.3.3
```

# reference

https://github.com/LeejwUniverse/RL_Rainbow_Pytorch

https://github.com/kyunghoon-jung/MacaronRL

https://github.com/pasus/Reinforcement-Learning-Book