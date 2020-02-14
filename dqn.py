import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np

from common.replay_buffers import BasicBuffer
from models import ConvDQN, DQN
    
    
class DQNAgent:

    def __init__(self, env, use_conv=True, learning_rate=3e-4, gamma=0.99, buffer_size=10000):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.use_conv = use_conv
        if self.use_conv:
            self.model = ConvDQN(env.observation_space.shape, len(env.action_space)).to(self.device)
        else:
            self.model = DQN(env.observation_space.shape, len(env.action_space)).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state, eps=0.20):
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        #print(qvals)
        action = np.argmax(qvals.cpu().detach().numpy())
        action = np.max(action)
        
        if (np.random.randn() < eps):
            return np.random.choice(self.env.action_space)

        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)
        #print(self.model.forward(states))
        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class DoubleDQNAgent:

    def __init__(self, env, use_conv=True, learning_rate=3e-4, gamma=0.99, tau=0.01, buffer_size=10000):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = BasicBuffer(max_size=buffer_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_conv = use_conv
        if self.use_conv:
            self.model1 = ConvDQN(env.observation_space.shape, env.action_space.n).to(self.device)
            self.model2 = ConvDQN(env.observation_space.shape, env.action_space.n).to(self.device)
        else:
            self.model1 = DQN(env.observation_space.shape, len(env.action_space)).to(self.device)
            self.model2 = DQN(env.observation_space.shape, len(env.action_space)).to(self.device)

        self.optimizer1 = torch.optim.Adam(self.model1.parameters())
        self.optimizer2 = torch.optim.Adam(self.model2.parameters())

    def get_action(self, state, eps=0.20):
        if (np.random.randn() < eps):
            return np.random.choice(self.env.action_space)

        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model1.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())

        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # resize tensors
        actions = actions.view(actions.size(0), 1)
        dones = dones.view(dones.size(0), 1)

        # compute loss
        curr_Q1 = self.model1.forward(states).gather(1, actions)
        curr_Q2 = self.model2.forward(states).gather(1, actions)

        next_Q1 = self.model1.forward(next_states)
        next_Q2 = self.model2.forward(next_states)
        next_Q = torch.min(
            torch.max(self.model1.forward(next_states), 1)[0],
            torch.max(self.model2.forward(next_states), 1)[0]
        )
        next_Q = next_Q.view(next_Q.size(0), 1)
        expected_Q = rewards + (1 - dones) * self.gamma * next_Q

        loss1 = F.mse_loss(curr_Q1, expected_Q.detach())
        loss2 = F.mse_loss(curr_Q2, expected_Q.detach())

        return loss1, loss2

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss1, loss2 = self.compute_loss(batch)

        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()

        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()
