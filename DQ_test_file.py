from stack_env import Environment
from dqn import DQNAgent
from common.utils import mini_batch_train_frames, test

MAX_FRAMES = 1000000
BATCH_SIZE = 32
mode = 'train'
env = Environment()
agent = DQNAgent(env, use_conv=True)
if mode == 'train':
    episode_rewards = mini_batch_train_frames(env, agent, MAX_FRAMES, BATCH_SIZE)
elif mode == 'test':
    test(env, agent)