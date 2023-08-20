import wandb
import argparse
import os
import torch
import numpy as np
import gym
import pickle
from itertools import count

from dreamerv2.models.DQN import DQN, ReplayMemory, Train
from dreamerv2.models.DQN_New import TrainDQN
from dreamerv2.models.A2C import TrainA2C

from wandb.integration.sb3 import WandbCallback
from stable_baselines3.dqn import DQN
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
from stable_baselines3.common.env_util import make_vec_env
from dreamerv2.utils.wrapper import GymMinAtar, OneHotAction, FlattenObservation
from dreamerv2.training.config import MinAtarConfig, MiniGridConfig
from dreamerv2.training.trainer import Trainer
from dreamerv2.training.evaluator import Evaluator
from singlepath_gym import SinglepathEnvGym

def main(args):
    wandb.login()
    env_name = args.env
    exp_id = args.id

    '''make dir for saving results'''
    result_dir = os.path.join('results', '{}_{}'.format(env_name, exp_id))
    model_dir = os.path.join(result_dir, 'models')                                                  #dir to save learnt models
    os.makedirs(model_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device:
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')
    print('using :', device)  
    
    fcc_train = np.asarray(pickle.load(open("./bw/fcc_train100kb.pickle", "rb")))
    fcc_train = np.repeat(fcc_train, 10, axis=1)
    lte_train = np.asarray(pickle.load(open("./bw/LTE_train100kb.pickle", "rb")))
    train = np.concatenate((fcc_train, lte_train), axis=0)

    #env = OneHotAction(GymMinAtar(env_name))
    env = OneHotAction(SinglepathEnvGym(bitrate_list=train))
    env = FlattenObservation(env)
    obs_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]
    
    obs_dtype = bool
    action_dtype = np.float32
    batch_size = args.batch_size
    seq_len = args.seq_len

    config = MinAtarConfig(
        env=env_name,
        obs_shape=obs_shape,
        action_size=action_size,
        #obs_dtype = obs_dtype,
        action_dtype = action_dtype,
        #seq_len = seq_len,
        #batch_size = batch_size,
        model_dir=model_dir, 
    )

    config_dict = config.__dict__
    trainer = Trainer(config, device)
    evaluator = Evaluator(config, device)

    with wandb.init(project='mastering MinAtar with world models', config=config_dict, sync_tensorboard=False): #sync = True for SB3
        '''
        #For SB3 DQN
        result_dir = os.path.join('results', '{}_{}'.format(env_name, exp_id))
        model_dir = os.path.join(result_dir, 'DQN_SB3') 
        log_dir = os.path.join(model_dir, "log")
        env = SinglepathEnvGym(bitrate_list=train)
        env = make_vec_env(SinglepathEnvGym, env_kwargs=dict(bitrate_list=train))
        #model = DQN(policy="MultiInputPolicy", env=env, learning_rate=4e-5, buffer_size=int(1e6), batch_size=config.dqn_batch_size, gamma=0.99, verbose=2, device=device, 
                    #target_update_interval=config.slow_target_update, tensorboard_log=log_dir, train_freq=(5, "step"))
        #model = DQN("MultiInputPolicy", env, 4e-5, int(1e6), 50000, 50, 1, 0.99, 4, 1, None, None, False, 100, 0.1, 1, 0.05, 10, None, None, 1, None, device, True )
        model = DQN(policy="MultiInputPolicy", env=env, verbose=2, device=device, tensorboard_log=log_dir)
        model.learn(total_timesteps=510000, log_interval=1, callback=WandbCallback(verbose=2, model_save_freq=50000, model_save_path=model_dir))
        
        wandb.finish()
        '''

        '''
        #For SB3 A2C
        result_dir = os.path.join('results', '{}_{}'.format(env_name, exp_id))
        model_dir = os.path.join(result_dir, 'A2C_SB3') 
        log_dir = os.path.join(model_dir, "log")
        env = SinglepathEnvGym(bitrate_list=train)
        env = make_vec_env(SinglepathEnvGym, env_kwargs=dict(bitrate_list=train))
        #model = DQN(policy="MultiInputPolicy", env=env, learning_rate=4e-5, buffer_size=int(1e6), batch_size=config.dqn_batch_size, gamma=0.99, verbose=2, device=device, 
                    #target_update_interval=config.slow_target_update, tensorboard_log=log_dir, train_freq=(5, "step"))
        #model = DQN("MultiInputPolicy", env, 4e-5, int(1e6), 50000, 50, 1, 0.99, 4, 1, None, None, False, 100, 0.1, 1, 0.05, 10, None, None, 1, None, device, True )
        model = A2C(policy="MultiInputPolicy", env=env, verbose=2, device=device, tensorboard_log=log_dir)
        model.learn(total_timesteps=1010000, log_interval=1, callback=WandbCallback(verbose=2, model_save_freq=50000, model_save_path=model_dir))
        
        wandb.finish()
        '''

        '''
        #For SB3 PPO
        result_dir = os.path.join('results', '{}_{}'.format(env_name, exp_id))
        model_dir = os.path.join(result_dir, 'PPO_SB3') 
        log_dir = os.path.join(model_dir, "log")
        env = SinglepathEnvGym(bitrate_list=train)
        env = make_vec_env(SinglepathEnvGym, env_kwargs=dict(bitrate_list=train))
        #model = DQN(policy="MultiInputPolicy", env=env, learning_rate=4e-5, buffer_size=int(1e6), batch_size=config.dqn_batch_size, gamma=0.99, verbose=2, device=device, 
                    #target_update_interval=config.slow_target_update, tensorboard_log=log_dir, train_freq=(5, "step"))
        #model = DQN("MultiInputPolicy", env, 4e-5, int(1e6), 50000, 50, 1, 0.99, 4, 1, None, None, False, 100, 0.1, 1, 0.05, 10, None, None, 1, None, device, True )
        model = PPO(policy="MultiInputPolicy", env=env, verbose=2, device=device, tensorboard_log=log_dir)
        model.learn(total_timesteps=1010000, log_interval=1, callback=WandbCallback(verbose=2, model_save_freq=50000, model_save_path=model_dir))
        
        wandb.finish()
        '''
        
        """training loop"""
        print('...training...')

        
        #DreamerV2
        train_metrics = {}
        trainer.collect_seed_episodes(env)
        obs, score = env.reset(), 0
        done = False
        prev_rssmstate = trainer.RSSM._init_rssm_state(1)
        prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
        episode_actor_ent = []
        scores = []
        best_mean_score = -99
        best_save_path = os.path.join(model_dir, 'models_best.pth')

        
        for iter in range(1, trainer.config.train_steps):  
            if iter%trainer.config.train_every == 0 and iter != 1:
                train_metrics = trainer.train_batch(train_metrics)
            if iter%trainer.config.slow_target_update == 0:
                trainer.update_target()                
            if iter%trainer.config.save_every == 0:
                trainer.save_model(iter)
            with torch.no_grad():
                embed = trainer.ObsEncoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(trainer.device))  
                _, posterior_rssm_state = trainer.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                model_state = trainer.RSSM.get_model_state(posterior_rssm_state)
                action, action_dist = trainer.ActionModel(model_state)
                action = trainer.ActionModel.add_exploration(action, iter).detach()
                action_ent = torch.mean(action_dist.entropy()).item()
                episode_actor_ent.append(action_ent)

            next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
            score += rew

            if done:
                #train_metrics = trainer.train_batch(train_metrics)
                trainer.buffer.add(obs, action.squeeze(0).cpu().numpy(), rew, done)
                train_metrics['train_rewards'] = score
                train_metrics['action_ent'] =  np.mean(episode_actor_ent)
                #wandb.log(train_metrics, step=iter)
                scores.append(score)
                if len(scores)>100:
                    scores.pop(0)
                    current_average = np.mean(scores)
                    train_metrics['mean_score'] = current_average
                    if current_average>best_mean_score:
                        best_mean_score = current_average 
                        #train_metrics['mean_score'] = current_average
                        print('saving best model with mean score : ', best_mean_score)
                        save_dict = trainer.get_save_dict()
                        torch.save(save_dict, best_save_path)
                wandb.log(train_metrics, step=iter)
                obs, score = env.reset(), 0
                done = False
                prev_rssmstate = trainer.RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
                episode_actor_ent = []
            else:
                trainer.buffer.add(obs, action.squeeze(0).detach().cpu().numpy(), rew, done)
                obs = next_obs
                prev_rssmstate = posterior_rssm_state
                prev_action = action
        

        '''
        #DQN
        train_metrics = {}
        DQN_BATCH_SIZE = 5
        env = SinglepathEnvGym(bitrate_list=train)
        env = FlattenObservation(env)
        obs_shape = env.observation_space.shape
        action_size = env.action_space.n
        state = env.reset()
        n_obs = len(state)

        #obs, score = env.reset(), 0
        #obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        #memory = ReplayMemory(config.capacity)

        result_dir = os.path.join('results', '{}_{}'.format(env_name, exp_id))
        model_dir = os.path.join(result_dir, 'DQN')  

        config = MinAtarConfig(
            env=env_name,
            obs_shape=obs_shape,
            action_size=action_size,
            #obs_dtype = obs_dtype,
            action_dtype = action_dtype,
            #seq_len = seq_len,
            #batch_size = batch_size,
            model_dir=model_dir, 
        )

        #trainer = Train(config, env, device)
        trainer = TrainDQN(config, env, n_obs, device)
        
        scores = []
        best_mean_score = -99
        best_save_path = os.path.join(model_dir, 'models_best.pth')

        state = env.reset()
        score = 0
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        for iter in range(1, trainer.config.train_steps): 
            #if (iter - 1) >= trainer.config.dqn_batch_size:
                #trainer.optimize_model()
            #if iter%trainer.config.dqn_slow_target_update == 0:
                #trainer.target_net.load_state_dict(trainer.policy_net.state_dict())               
            if iter%trainer.config.save_every == 0:
                trainer.save_model(iter)
            action = trainer.select_action(state)
            observation, reward, terminated, truncated = env.step(action.item())
            score += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            # Store the transition in memory
            trainer.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            trainer.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = trainer.target_net.state_dict()
            policy_net_state_dict = trainer.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*0.005 + target_net_state_dict[key]*(1-0.005)
            trainer.target_net.load_state_dict(target_net_state_dict)

            if done:
                state = env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                train_metrics['train_rewards'] = score
                train_metrics["loss"] = trainer.loss
                wandb.log(train_metrics, step=iter)
                scores.append(score)
                if len(scores)>100:
                    scores.pop(0)
                    current_average = np.mean(scores)
                    if current_average>best_mean_score:
                        best_mean_score = current_average 
                        print('saving best model with mean score : ', best_mean_score)
                        save_dict = trainer.get_save_dict()
                        torch.save(save_dict, best_save_path)
                
                score = 0
        '''

        '''
        #A2C
        train_metrics = {}
        env = SinglepathEnvGym(bitrate_list=train)
        env = FlattenObservation(env)
        obs_shape = env.observation_space.shape
        action_size = env.action_space.n
        state = env.reset()
        n_obs = len(state)

        result_dir = os.path.join('results', '{}_{}'.format(env_name, exp_id))
        model_dir = os.path.join(result_dir, 'A2C')  

        config = MinAtarConfig(
            env=env_name,
            obs_shape=obs_shape,
            action_size=action_size,
            #obs_dtype = obs_dtype,
            action_dtype = action_dtype,
            #seq_len = seq_len,
            #batch_size = batch_size,
            model_dir=model_dir, 
        )
        trainer = TrainA2C(config, env, n_obs, device)
        
        scores = []
        best_mean_score = -99
        best_save_path = os.path.join(model_dir, 'models_best.pth')

        state = env.reset()
        score = 0
        ep_reward = 0
        #state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for iter in range(1, trainer.config.train_steps): 
            if iter%trainer.config.save_every == 0:
                trainer.save_model(iter)
            # select action from policy
            action = trainer.select_action(state)

            # take the action
            state, reward, done, _ = env.step(action)

            trainer.model.rewards.append(reward)
            score += reward
            ep_reward += reward
            if done:
                # update cumulative reward
                state = env.reset()
                #state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                ep_reward = 0
                train_metrics['train_rewards'] = score
                train_metrics["loss"] = trainer.loss
                wandb.log(train_metrics, step=iter)
                scores.append(score)
                # perform backprop
                trainer.finish_episode()

                if len(scores)>100:
                    scores.pop(0)
                    current_average = np.mean(scores)
                    if current_average>best_mean_score:
                        best_mean_score = current_average 
                        print('saving best model with mean score : ', best_mean_score)
                        save_dict = trainer.get_save_dict()
                        torch.save(save_dict, best_save_path)
                
                score = 0
        '''

        


    '''evaluating probably best model'''
    #evaluator.eval_saved_agent(env, best_save_path)
    

if __name__ == "__main__":

    """there are tonnes of HPs, if you want to do an ablation over any particular one, please add if here"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, help='mini atari env name')
    parser.add_argument("--id", type=str, default='0', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--device', default='cuda', help='CUDA or CPU')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=50, help='Sequence Length (chunk length)')
    args = parser.parse_args()
    main(args)
