import numpy as np
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from collections import namedtuple

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, n_observations, n_actions):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(n_observations, 128)

        # actor's layer
        self.action_head = nn.Linear(128, n_actions)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values
    
class TrainA2C():
    def __init__(self, config, env, n_obs, device):
        self.config = config
        self.env = env
        self.device = device
        self.n_obs = n_obs
        self.loss = 0

        self.GAMMA = config.a2c_lr

        self.model = Policy(n_obs, env.action_space.n).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-2)
        self.eps = np.finfo(np.float32).eps.item()

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        probs, state_value = self.model(state)

    # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

    # and sample an action using the distribution
        action = m.sample()

    # save to action buffer
        self.model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
        return action.item()


    def finish_episode(self):
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """
        R = 0
        saved_actions = self.model.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
        for r in self.model.rewards[::-1]:
        # calculate the discounted value
            R = r + self.GAMMA * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

        # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R], device = self.device)))

    # reset gradients
        self.optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        self.loss = loss

    # perform backprop
        loss.backward()
        self.optimizer.step()

    # reset rewards and action buffer
        del self.model.rewards[:]
        del self.model.saved_actions[:]

    def get_save_dict(self):
        return {
            "policy": self.model.state_dict()
        }
    
    def save_model(self, iter):
        save_dict = self.get_save_dict()
        model_dir = self.config.model_dir
        save_path = os.path.join(model_dir, 'models_%d.pth' % iter)
        torch.save(save_dict, save_path)
