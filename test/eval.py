import argparse
import os
import torch
import numpy as np
import gym
import pickle

from dreamerv2.utils.wrapper import GymMinAtar, OneHotAction, breakoutPOMDP, space_invadersPOMDP, seaquestPOMDP, asterixPOMDP, freewayPOMDP, FlattenObservation
from dreamerv2.training.config import MinAtarConfig
from dreamerv2.training.evaluator import Evaluator
from singlepath_gym import SinglepathEnvGym

from stable_baselines3.common.env_util import make_vec_env

pomdp_wrappers = {
    'breakout':breakoutPOMDP,
    'seaquest':seaquestPOMDP,
    'space_invaders':space_invadersPOMDP,
    'asterix':asterixPOMDP,
    'freeway':freewayPOMDP,
}

def main(args):
    print(args)

    fcc_test = np.asarray(pickle.load(open("./bw/fcc_test100kb.pickle", "rb")))
    fcc_test = np.repeat(fcc_test, 10, axis=1)
    lte_test = np.asarray(pickle.load(open("./bw/LTE_test100kb.pickle", "rb")))
    #train = np.concatenate((fcc_train, lte_train), axis=0)

    env_name = args.env
    if args.pomdp==1:
        exp_id = args.id + '_pomdp'
        PomdpWrapper = pomdp_wrappers[env_name]
        env = PomdpWrapper(OneHotAction(GymMinAtar(env_name)))
        print('using partial state info')
    else:
        exp_id = args.id
        env = OneHotAction(SinglepathEnvGym(bitrate_list=lte_test, train=False))
        env = FlattenObservation(env)
        #obs_shape = env.observation_space.shape
        #action_size = env.action_space.shape[0]
        print('using complete state info')
    
    if args.eval_episode == 1:
        eval_render = True
    else:
        eval_render = False

    if torch.cuda.is_available() and args.device:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('using :', device)  
    
    result_dir = os.path.join('results', '{}_{}'.format(env_name, exp_id))
    model_dir = os.path.join(result_dir, 'models')
    DQN_SB3_dir = os.path.join(result_dir, 'DQN_SB3')      
    A2C_SB3_dir = os.path.join(result_dir, 'A2C_SB3') 
    PPO_SB3_dir = os.path.join(result_dir, 'PPO_SB3')                          

    #obs_shape = env.observation_space.shape
    #action_size = env.action_space.shape[0]
    obs_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]

    #obs_dtype = bool 
    action_dtype = np.float32

    config = MinAtarConfig(
        env=env_name,
        obs_shape=obs_shape,
        action_size=action_size,
        #obs_dtype = obs_dtype,
        action_dtype = action_dtype,
        model_dir=model_dir, 
        eval_episode=args.eval_episode,
        eval_render=False
    )

    evaluator = Evaluator(config, device)
    best_score = 0

    
    for f in sorted(os.listdir(model_dir)):
        eval_score = evaluator.eval_saved_agent(env,  os.path.join(model_dir, f))
        if eval_score > best_score:
            print('..saving model number')
            best_score=eval_score
    

    #env = SinglepathEnvGym(bitrate_list=fcc_test, train=False)
    #env = make_vec_env(SinglepathEnvGym, env_kwargs=dict(bitrate_list=fcc_test, train=False))
    #eval_score = evaluator.eval_DQN_SB3(env=env, model_path=os.path.join(DQN_SB3_dir, 'model.zip'))
    #best_score=eval_score

    #env = SinglepathEnvGym(bitrate_list=fcc_test, train=False)
    #env = make_vec_env(SinglepathEnvGym, env_kwargs=dict(bitrate_list=fcc_test, train=False))
    #eval_score = evaluator.eval_A2C_SB3(env=env, model_path=os.path.join(A2C_SB3_dir, 'model.zip'))
    #best_score=eval_score

    #env = SinglepathEnvGym(bitrate_list=fcc_test, train=False)
    #env = make_vec_env(SinglepathEnvGym, env_kwargs=dict(bitrate_list=fcc_test, train=False))
    #eval_score = evaluator.eval_PPO_SB3(env=env, model_path=os.path.join(PPO_SB3_dir, 'model.zip'))
    #best_score=eval_score

    print('best mean evaluation score amongst stored models is : ', best_score)

if __name__ == "__main__":

    """there are tonnes of HPs, if you want to do an ablation over any particular one, please add if here"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, help='mini atari env name')
    parser.add_argument('--eval_episode', type=int, default=10, help='number of episodes to eval')
    parser.add_argument("--id", type=str, default='0', help='Experiment ID')
    parser.add_argument("--eval_render", type=int, help='to render while evaluation')
    parser.add_argument("--pomdp", type=int, help='partial observation flag')
    parser.add_argument('--device', default='cuda', help='CUDA or CPU')
    args = parser.parse_args()
    main(args)
