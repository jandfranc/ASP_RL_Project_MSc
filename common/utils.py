import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import math

# KL divergence of two univariate Gaussian distributions
def KL_divergence_mean_std(p_mean, p_std, q_mean, q_std):
    kld = torch.log(q_std/p_std) + (torch.pow(p_std) + torch.pow(p_mean - q_mean, 2))/(2 * torch.pow(q_std)) - 0.5
    return kld

# compute KL divergence of two distributions
def KL_divergence_two_dist(dist_p, dist_q):
    kld = torch.sum(dist_p * (torch.log(dist_p) - torch.log(dist_q)))
    return kld

# project value distribution onto atoms as in Categorical Algorithm
def dist_projection(optimal_dist, rewards, dones, gamma, n_atoms, Vmin, Vmax, support):
    batch_size = rewards.size(0)
    m = torch.zeros(batch_size, n_atoms)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)

    for sample_idx in range(batch_size):
        reward = rewards[sample_idx]
        done = dones[sample_idx]

        for atom in range(n_atoms):
            # compute projection of Tz_j
            Tz_j = reward + (1 - done) * gamma * support[atom]
            Tz_j = torch.clamp(Tz_j, Vmin, Vmax)
            b_j = (Tz_j - Vmin) / delta_z
            l = torch.floor(b_j).long().item()
            u = torch.ceil(b_j).long().item()

            # distribute probability of Tz_j
            m[sample_idx][l] = m[sample_idx][l] + optimal_dist[sample_idx][atom] * (u - b_j)
            m[sample_idx][u] = m[sample_idx][u] + optimal_dist[sample_idx][atom] * (b_j - l)

    #print(m)
    return m

def projection_distribution(next_state, rewards, dones):
    batch_size  = next_state.size(0)
    
    delta_z = float(Vmax - Vmin) / (num_atoms - 1)
    support = torch.linspace(Vmin, Vmax, num_atoms)
    print(support)
    next_dist   = model(next_state)[0].data.cpu() * support
    next_action = next_dist.sum(2).max(1)[1]
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
    next_dist   = next_dist.gather(1, next_action).squeeze(1)
        
    rewards = rewards.unsqueeze(1).expand_as(next_dist)
    dones   = dones.unsqueeze(1).expand_as(next_dist)
    support = support.unsqueeze(0).expand_as(next_dist)
    print(support)
    
    Tz = rewards + (1 - dones) * 0.99 * support
    Tz = Tz.clamp(min=Vmin, max=Vmax)
    b  = (Tz - Vmin) / delta_z
    l  = b.floor().long()
    u  = b.ceil().long()
        
    offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long()\
                    .unsqueeze(1).expand(batch_size, num_atoms)
    
    print(offset)

    proj_dist = torch.zeros(next_dist.size())
    print((l + offset).view(-1))
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        
    return next_dist, proj_dist

def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset_world()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)   

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

    return episode_rewards


import matplotlib.pyplot as plt
import pickle

def test(env, agent):
    with open('episode_rewards_means_DDQN.pickle', 'rb') as learner:
        episode_rewards_means = pickle.load(learner)

    plt.plot(episode_rewards_means)
    plt.show()
    state = env.reset_world()
    with open('deepQ_DDQN_1.pickle', 'rb') as learner:
        agent.model1 = pickle.load(learner)
    with open('deepQ_DDQN_2.pickle', 'rb') as learner:
        agent.model2 = pickle.load(learner)
    while True:
        action = agent.get_action(state, eps=0)
        state, reward, done = env.step(action)

        if done:
            env.reset_world()

def mini_batch_train_frames(env, agent, max_frames, batch_size):
    episode_rewards = []
    episode_rewards_means = []
    state = env.reset_world()
    episode_reward = 0
    try:
        with open('deepQ_DDQN_1.pickle', 'rb') as learner:
            agent.model1 = pickle.load(learner)
        with open('deepQ_DDQN_2.pickle', 'rb') as learner:
            agent.model2 = pickle.load(learner)
        with open('episode_rewards_means_DDQN.pickle', 'rb') as learner:
            episode_rewards_means = pickle.load(learner)
    except:
        print('No previous model found')

    for frame in range(max_frames):
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        episode_reward += reward

        if len(agent.replay_buffer) > batch_size:
            agent.update(batch_size)   

        if done:
            episode_rewards.append(episode_reward)
            episode_rewards_means.append(np.mean(episode_rewards[-50:-1]))
            print("Frame " + str(frame) + ": " + str(episode_reward))
            print(len(episode_rewards_means))
            env.reset_world()
            episode_reward = 0
            if len(episode_rewards) % 10 == 0:
                with open(r"episode_rewards_means_DDQN.pickle", "wb") as output_file:
                    pickle.dump(episode_rewards_means, output_file)
                with open(r"deepQ_DDQN_1.pickle", "wb") as output_file:
                    pickle.dump(agent.model1, output_file)
                with open(r"deepQ_DDQN_2.pickle", "wb") as output_file:
                    pickle.dump(agent.model2, output_file)
            if len(episode_rewards_means) % 500 == 0:
                plt.plot(episode_rewards_means)
                plt.show()
        state = next_state
    print(episode_rewards)
    return episode_rewards

# run environment
def run_environment(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))

    return episode_rewards

# process episode rewards for multiple trials
def process_episode_rewards(many_episode_rewards):
    minimum = [np.min(episode_reward) for episode_reward in episode_rewards]
    maximum = [np.max(episode_reward) for episode_reward in episode_rewards]
    mean = [np.mean(episode_reward) for episode_reward in episode_rewards]

    return minimum, maximum, mean


class NoisyLinear(nn.Module):

    def __init__(self, num_in, num_out, is_training=True):
        super(NoisyLinear, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.is_training = is_training

        self.mu_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.mu_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.sigma_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.register_buffer("epsilon_weight", torch.FloatTensor(num_out, num_in)) 
        self.register_buffer("epsilon_bias", torch.FloatTensor(num_out))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        self.reset_noise()

        if self.is_training:
            weight = self.mu_weight + self.sigma_weight.mul(autograd.Variable(self.epsilon_weight)) 
            bias = self.mu_bias + self.sigma_bias.mul(autograd.Variable(self.epsilon_bias))
        else:
            weight = self.mu_weight
            buas = self.mu_bias

        y = F.linear(x, weight, bias)
        
        return y

    def reset_parameters(self):
        std = math.sqrt(3 / self.num_in)
        self.mu_weight.data.uniform_(-std, std)
        self.mu_bias.data.uniform_(-std,std)

        self.sigma_weight.data.fill_(0.017)
        self.sigma_bias.data.fill_(0.017)

    def reset_noise(self):
        self.epsilon_weight.data.normal_()
        self.epsilon_bias.data.normal_()

    
class FactorizedNoisyLinear(nn.Module):

    def __init__(self, num_in, num_out, is_training=True):
        super(FactorizedNoisyLinear, self).__init__()
        self.num_in = num_in
        self.num_out = num_out 
        self.is_training = is_training

        self.mu_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.mu_bias = nn.Parameter(torch.FloatTensor(num_out)) 
        self.sigma_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.register_buffer("epsilon_i", torch.FloatTensor(num_in))
        self.register_buffer("epsilon_j", torch.FloatTensor(num_out))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        self.reset_noise()
        
        if self.is_training:
            epsilon_weight = self.epsilon_j.ger(self.epsilon_i)
            epsilon_bias = self.epsilon_j
            weight = self.mu_weight + self.sigma_weight.mul(autograd.Variable(epsilon_weight))
            bias = self.mu_bias + self.sigma_bias.mul(autograd.Variable(epsilon_bias))
        else:
            weight = self.mu_weight
            bias = self.mu_bias

        y = F.linear(x, weight, bias)
        
        return y

    def reset_parameters(self):
        std = 1 / math.sqrt(self.num_in)
        self.mu_weight.data.uniform_(-std, std)
        self.mu_bias.data.uniform_(-std, std)

        self.sigma_weight.data.fill_(0.5 / math.sqrt(self.num_in))
        self.sigma_bias.data.fill_(0.5 / math.sqrt(self.num_in))

    def reset_noise(self):
        eps_i = torch.randn(self.num_in)
        eps_j = torch.randn(self.num_out)
        self.epsilon_i = eps_i.sign() * (eps_i.abs()).sqrt()
        self.epsilon_j = eps_j.sign() * (eps_j.abs()).sqrt()
