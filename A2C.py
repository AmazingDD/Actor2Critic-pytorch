import os
import gc
import gym
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class Actor(nn.Module):
    def __init__(self, epochs, state_dim, action_size=2, action_limit=1.):
        super(Actor, self).__init__()
        self.epochs = epochs
        self.state_dim = state_dim
        self.action_dim = action_size
        self.action_lim = action_limit
        
        ''' softmax network '''
        hidden_layers=[64, 32, 8]
        modules = []
        seq = [state_dim] + hidden_layers
        for in_dim, out_dim in zip(seq[: -1], seq[1:]):
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(nn.ReLU())
        self.hidden = nn.Sequential(*seq)
        
        self.out = nn.Linear(seq[-1], action_size)
        
        self._init_weight()

    def forward(self, state):
        x = self.hidden(state)
        x = self.out(x)
        action = F.tanh(x)

        action *= self.action_lim

        return action
    
    def _init_weight(self):
        for m in self.hidden:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

        nn.init.normal_(self.softmax_in.weight)
        nn.init.constant_(self.softmax_in.bias, 0.01)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
 
        s_layer = [64, 32, 8]
        modules = []
        seq = [state_dim] + s_layer
        for in_dim, out_dim in zip(seq[: -1], seq[1:]):
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(nn.ReLU())
        self.s_hidden = nn.Sequential(*seq)

        s_layer = [64, 32, 8]
        modules = []
        seq = [state_dim] + s_layer
        for in_dim, out_dim in zip(seq[: -1], seq[1:]):
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(nn.ReLU())
        self.s_hidden = nn.Sequential(*seq)

        a_layer = [32, 8]
        modules = []
        seq = [action_dim] + s_layer
        for in_dim, out_dim in zip(seq[: -1], seq[1:]):
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(nn.ReLU())
        self.a_hidden = nn.Sequential(*seq)

        self.out = nn.Linear(a_layer[-1] + s_layer[-1], 1)

        self._init_weight()

    def _init_weight(self):
        for m in self.s_hidden:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)
        for m in self.a_hidden:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

        nn.init.normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.01)


    def forward(self, state, action):
        '''
        Q(s, a)
        '''
        s = self.s_hidden(state)
        a = self.a_hidden(action)

        x = torch.cat((s, a), dim=1)
        x = self.out(x)

        return x

class Noise(object):
    """
    implement ornstein-uhlenbeck noise
    Example:
    >>> no = Noise(1)
    >>> states = []
    >>> for i in range(1000):
    ...     states.append(no.sample())
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(states)
    >>> plt.show()
    """
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = mu * np.ones(action_dim)

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx += self.sigma * np.random.randn(len(self.X))
        self.X += dx
        return self.X


class Trainer(object):
    def __init__(self, buffer, state_dim, action_dim, action_limit, batch_size=128, lr=0.001, gamma=0.99, tau=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_limit
        self.buffer = buffer
        self.iter = 0
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.noise = Noise(action_dim)

        self.actor = Actor(state_dim, action_dim, action_limit)
        self.target_actor = Actor(state_dim, action_dim, action_limit)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr)

        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr)

        self._update(self.target_actor, self.actor)
        self._update(self.target_critic, self.critic)

    def _update(self, tar, src):
        for tar_param, param in zip(tar.parameters(), src.parameters()):
            tar_param.data.copy_(param.data)

    def _soft_update(self, tar, src):
        for target_param, param in zip(tar.parameters(), src.parameters()):
            target_param.data.copy_(
                target_param.data * (1 - self.tau) + param.data * self.tau
            )

    def get_exploitation_action(self, state):
        state = torch.from_numpy(state)
        action = self.target_actor.forward(state).detach()
        return action.data.numpy()

    def get_exploration_action(self, state):
        state = torch.from_numpy(state)
        action = self.actor.forward(state).detach()
        new_action = action.data.numpy() + (self.noise.sample() * self.action_lim)

        return new_action

    def optimize(self):
        s1, a1, r1, s2 = self.buffer.sample(self.batch_size)

        s1 = torch.from_numpy(s1)
        a1 = torch.from_numpy(a1)
        r1 = torch.from_numpy(r1)
        s2 = torch.from_numpy(s2)

        ''' optimize critic '''
        a2 = self.target_actor.forward(s2).detach()
        next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
        val_expected = r1 + self.gamma * next_val
        val_predicted = torch.squeeze(self.critic.forward(s1, a1))

        critic_loss = F.mse_loss(val_predicted, val_expected)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        ''' optimize actor '''
        pred_a1 = self.actor.forward(s1)
        actor_loss = -1 * torch.sum(self.critic.forward(s1, pred_a1))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)

        if self.iter % 100 == 0:
            print(f'Iteration :- {self.iter}, Loss_actor :- {actor_loss.data.numpy()}, Loss_critic :- {critic_loss.data.numpy()}')
        self.iter += 1

    def save(self, eps_cnt):
        if not os.path.exists('./model/'):
            os.makedirs('./model/')
        torch.save(self.target_actor.state_dict(), f'./model/{eps_cnt}_actor.pt')
        torch.save(self.target_critic.state_dict(), f'./model/{eps_cnt}_critic.pt')
        print('Models saved successfully')

    def load(self, eps_cnt):
        self.actor.load_state_dict(torch.load(f'./model/{eps_cnt}_actor.pt'))
        self.critic.load_state_dict(torch.load(f'./model/{eps_cnt}_critic.pt'))
        self._update(self.target_actor, self.actor)
        self._update(self.target_critic, self.critic)
        print('Models loaded successfully')

class Buffer(object):
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.max_size = size
        self.len = 0

    def sample(self, cnt):
        """
        samples a random batch from the replay memory buffer
        :param cnt: batch size
        :return: batch (numpy array)
        """
        batch = []
        cnt = min(cnt, self.len)

        s_arr = np.float32([arr[0] for arr in batch])
        a_arr = np.float32([arr[1] for arr in batch])
        r_arr = np.float32([arr[2] for arr in batch])
        s1_arr = np.float32([arr[3] for arr in batch])

        return s_arr, a_arr, r_arr, s1_arr

    def add(self, s, a, r, s1):
        """
        add a particular transaction in the memory buffer
        :param s: current state
        :param a: action taken
        :param r: reward received
        :param s1: next state
        """
        transaction = (s, a, r, s1)
        self.len += 1
        if self.len > self.max_size:
            self.len = self.max_size
        self.buffer.append(transaction)

    def length(self):
        return self.len

if __name__ == '__main__':
    max_episodes = 400
    # state_dim = 10
    # action_dim = 2
    # action_max = 1
    max_step = 1000

    env = gym.make('BipedalWalker-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_max = env.action_space.high[0]

    print(
        f'State Dimension : {state_dim}', 
        f'action Dimension : {action_dim}', 
        f'action limitation : {action_max}',
        sep='\n'
    )

    ram = Buffer(max_episodes)
    trainer = Trainer(ram, state_dim, action_dim, action_max)
    for eps in range(max_episodes):
        observation = env.reset()
        print(f'[EPISODE {eps}]')
        for r in range(max_step):
            state = np.float32(observation)
            action = trainer.get_exploration_action(state)
            new_observation, reward, done, info = env.step(action)
            if done:
                new_state = None
            else:
                new_state = np.float32(new_observation)
                # push this experience in ram
                ram.add(state, action, reward, new_state)

            observation = new_observation

            trainer.optimize()
            if done:
                break
        gc.collect()

        if eps % 100 == 0:
            trainer.save(eps)

    print('Complete!')
