import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as td
from collections import namedtuple, deque
import os 


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN (nn.Module):
    def __init__(
            self, 
            output_size,
            input_size, 
            info,
            expl_info, 
            device
        ):
        super(DQN, self).__init__()
        self._output_size = output_size
        self._input_size = input_size
        self.device = device
        self._layers = info['layers']
        self._node_size = info['node_size']
        self.activation = info['activation']
        self.dist = info['dist']
        self.train_noise = expl_info['train_noise']
        self.eval_noise = expl_info['eval_noise']
        self.expl_min = expl_info['expl_min']
        self.expl_decay = expl_info['expl_decay']
        self.expl_type = expl_info['expl_type']
        #self.model = self.build_model()
        self.layer1 = nn.Linear(self._input_size, self._node_size)
        self.layer2 = nn.Linear(self._node_size, self._node_size)
        self.layer3 = nn.Linear(self._node_size, self._output_size)

    def build_model(self):
        model = [nn.Linear(self._input_size, self._node_size)]
        model += [self.activation()]
        for i in range(self._layers-1):
            model += [nn.Linear(self._node_size, self._node_size)]
            model += [self.activation()]
        model += [nn.Linear(self._node_size, self._output_size)]
        return nn.Sequential(*model)
    
    def forward(self, x):
        #x = x.to(self.device)
        #return self.model(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
        
    '''
    def select_action(self, x):
        #action = self.model(x)
        #if self.dist == 'normal':
        return self.model(x).max(1)[1].view(1, 1)
            #return action_dist.sample(), action_dist, action
        #raise NotImplementedError(self._dist)
    
    
    def add_exploration(self, action: torch.Tensor, itr: int, mode='train'):
        if mode == 'train':
            expl_amount = self.train_noise
            expl_amount = expl_amount - itr/self.expl_decay
            expl_amount = max(self.expl_min, expl_amount)
        elif mode == 'eval':
            expl_amount = self.eval_noise
        else:
            raise NotImplementedError
            
        if self.expl_type == 'epsilon_greedy':
            if np.random.uniform(0, 1) < expl_amount:
                return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
            return action

        raise NotImplementedError
    '''
        
class Train():
    def __init__(self, config, env, device):
        self.config = config
        self.device = device
        self.memory = ReplayMemory(config.capacity)
        self.env = env
        state = self.env.reset()
        self.policy_net = DQN(env.action_space.n, len(state), config.dqn, config.expl_dqn, device).to(self.device)
        self.target_net = DQN(env.action_space.n, len(state), config.dqn, config.expl_dqn, device).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        #self.optimizer = optim.Adam(self.policy_net.parameters(), config.dqn_lr)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.config.dqn_lr, amsgrad=True)
        self.loss = 0
        self.train_noise = config.expl_dqn['train_noise']
        self.eval_noise = config.expl_dqn['eval_noise']
        self.expl_min = config.expl_dqn['expl_min']
        self.expl_decay = config.expl_dqn['expl_decay']
        self.expl_type = config.expl_dqn['expl_type']
        #self.env = env
    
    def select_action(self, state):
        #action = self.model(x)
        #if self.dist == 'normal':
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)
            #return action_dist.sample(), action_dist, action
        #raise NotImplementedError(self._dist)

    def add_exploration(self, action: torch.Tensor, itr: int, mode='train'):
        if mode == 'train':
            expl_amount = self.train_noise
            expl_amount = expl_amount - itr/self.expl_decay
            expl_amount = max(self.expl_min, expl_amount)
        elif mode == 'eval':
            expl_amount = self.eval_noise
        else:
            raise NotImplementedError
            
        if self.expl_type == 'epsilon_greedy':
            if np.random.uniform(0, 1) < expl_amount:
                return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
            return action

        raise NotImplementedError

    def optimize_model(self):
        transitions = self.memory.sample(self.config.dqn_batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        #action_batch = action_batch.type(torch.int64)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.config.dqn_batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.config.dqn_discount) + reward_batch

    # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
    # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def get_save_dict(self):
        return {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict()
        }
    
    def save_model(self, iter):
        save_dict = self.get_save_dict()
        model_dir = self.config.model_dir
        save_path = os.path.join(model_dir, 'models_%d.pth' % iter)
        torch.save(save_dict, save_path)

        
