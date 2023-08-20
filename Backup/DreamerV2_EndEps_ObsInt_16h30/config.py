import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Tuple, Dict

# Following HPs are not a result of detailed tuning.   

@dataclass
class MinAtarConfig():
    '''default HPs that are known to work for MinAtar envs '''
    #env desc
    env : str                                           
    obs_shape: Tuple                                            
    action_size: int
    pixel: bool = False
    action_repeat: int = 1
    
    #buffer desc
    capacity: int = int(1e6)
    obs_dtype: np.dtype = np.uint8
    action_dtype: np.dtype = np.float32

    #training desc
    train_steps: int = int(5e6)
    train_every: int = 60 #50                                  #reduce this to potentially improve sample requirements
    collect_intervals: int = 5 
    batch_size: int = 60 #60 
    seq_len: int = 60 #60
    eval_episode: int = 4
    eval_render: bool = True
    save_every: int = int(5e4)# int(1e5)
    seed_steps: int = 4000
    model_dir: int = 'results'
    gif_dir: int = 'results'
    
    #latent space desc
    rssm_type: str = 'discrete'
    embedding_size: int = 200
    rssm_node_size: int = 200
    rssm_info: Dict = field(default_factory=lambda:{'deter_size':200, 'stoch_size':20, 'class_size':16, 'category_size':16, 'min_std':0.1})
    #rssm_info: Dict = field(default_factory=lambda:{'deter_size':600, 'stoch_size':256, 'class_size':32, 'category_size':32, 'min_std':0.1})
    
    #objective desc
    grad_clip: float = 100 #100
    discount_: float = 0.99 #0.99
    lambda_: float = 0.95
    horizon: int = 10 #10
    #lr: Dict = field(default_factory=lambda:{'model':2e-4, 'actor':4e-5, 'critic':1e-4})
    lr: Dict = field(default_factory=lambda:{'model':2e-4, 'actor':4e-5, 'critic':1e-4})
    loss_scale: Dict = field(default_factory=lambda:{'kl':0.1, 'reward':1.0, 'discount':5.0})
    kl: Dict = field(default_factory=lambda:{'use_kl_balance':True, 'kl_balance_scale':0.8, 'use_free_nats':False, 'free_nats':0.0})
    use_slow_target: float = True
    slow_target_update: int = 100 #*2 train steps
    slow_target_fraction: float = 1.00

    #actor critic
    actor: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'one_hot', 'min_std':1e-4, 'init_std':5, 'mean_scale':5, 'activation':nn.ELU})
    critic: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist': 'normal', 'activation':nn.ELU})
    #actor: Dict = field(default_factory=lambda:{'layers':4, 'node_size':400, 'dist':'one_hot', 'min_std':1e-4, 'init_std':5, 'mean_scale':5, 'activation':nn.ELU})
    #critic: Dict = field(default_factory=lambda:{'layers':4, 'node_size':400, 'dist': 'normal', 'activation':nn.ELU})
    expl: Dict = field(default_factory=lambda:{'train_noise':0.4, 'eval_noise':0.0, 'expl_min':0.05, 'expl_decay':7000.0, 'expl_type':'epsilon_greedy'})
    actor_grad: str ='reinforce'
    actor_grad_mix: int = 0 #0
    actor_entropy_scale: float = 1e-3

    #learnt world-models desc
    obs_encoder: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist': None, 'activation':nn.ELU, 'kernel':3, 'depth':16})
    obs_decoder: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU, 'kernel':3, 'depth':16})
    reward: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU})
    discount: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'binary', 'activation':nn.ELU, 'use':True})

    #DQN
    dqn: Dict = field(default_factory=lambda:{'layers':3, 'node_size':128, 'dist': 'normal', 'activation':nn.ReLU})
    expl_dqn: Dict = field(default_factory=lambda:{'train_noise':0.4, 'eval_noise':0.0, 'expl_min':0.05, 'expl_decay':7000.0, 'expl_type':'epsilon_greedy'})
    dqn_batch_size = 5
    dqn_slow_target_update = 10
    dqn_lr = 1e-4
    dqn_discount = float = 0.99

    #A2C
    a2c: Dict = field(default_factory=lambda:{'layers':3, 'node_size':128, 'dist': 'normal', 'activation':nn.ReLU})
    #a2c_batch_size = 5
    #a2c_slow_target_update = 10
    a2c_lr = 3e-2
    a2c_discount = float = 0.99

@dataclass
class MiniGridConfig():
    '''default HPs that are known to work for MiniGrid envs'''
    #env desc
    env : str                                           
    obs_shape: Tuple                                            
    action_size: int
    pixel: bool = False
    action_repeat: int = 1
    time_limit: int = 1000
    
    #buffer desc
    capacity: int = int(1e6)
    obs_dtype: np.dtype = np.uint8
    action_dtype: np.dtype = np.float32

    #training desc
    train_steps: int = int(1e6)
    train_every: int = 5
    collect_intervals: int = 5
    batch_size: int = 50
    seq_len: int = 8
    eval_episode: int = 5
    eval_render: bool = False
    save_every: int = int(5e4)
    seed_steps: int = 5
    model_dir: int = 'results'
    gif_dir: int = 'results'

    #latent space desc
    rssm_type: str = 'discrete'
    embedding_size: int = 100
    rssm_node_size: int = 100
    rssm_info: Dict = field(default_factory=lambda:{'deter_size':100, 'stoch_size':256, 'class_size':16, 'category_size':16, 'min_std':0.1})

    #objective desc
    grad_clip: float = 100.0
    discount_: float = 0.99
    lambda_: float = 0.95
    horizon: int = 8
    lr: Dict = field(default_factory=lambda:{'model':2e-4, 'actor':4e-5, 'critic':1e-4})
    loss_scale: Dict = field(default_factory=lambda:{'kl':1, 'reward':1.0, 'discount':10.0})
    kl: Dict = field(default_factory=lambda:{'use_kl_balance':True, 'kl_balance_scale':0.8, 'use_free_nats':False, 'free_nats':0.0})
    use_slow_target: float = True
    slow_target_update: int = 50
    slow_target_fraction: float = 1.0

    #actor critic
    actor: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'one_hot', 'min_std':1e-4, 'init_std':5, 'mean_scale':5, 'activation':nn.ELU})
    critic: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist': 'normal', 'activation':nn.ELU})
    expl: Dict = field(default_factory=lambda:{'train_noise':0.4, 'eval_noise':0.0, 'expl_min':0.05, 'expl_decay':10000.0, 'expl_type':'epsilon_greedy'})
    actor_grad: str ='reinforce'
    actor_grad_mix: int = 0.0
    actor_entropy_scale: float = 1e-3

    #learnt world-models desc
    obs_encoder: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist': None, 'activation':nn.ELU, 'kernel':2, 'depth':16})
    obs_decoder: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU, 'kernel':2, 'depth':16})
    reward: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU})
    discount: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'binary', 'activation':nn.ELU, 'use':True})

    
