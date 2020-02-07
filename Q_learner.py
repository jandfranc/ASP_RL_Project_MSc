import random
from stack_env import Environment
import numpy as np
import matplotlib.pyplot as plt
import pygame


def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]


class FeatureTransform:
    def __init__(self):
        # self.obj_y_bins = np.linspace(40, 1000, 193)
        self.obj_x_bins = np.linspace(25, 190, 34)

    def transform(self, observation):
        return to_bin(observation, self.obj_x_bins)-1


class Q_learner:

    def __init__(self, gamma, above_epsilon, feature_transformer):
        self.gamma = gamma
        self.above_epsilon = above_epsilon
        self.feature_transformer = feature_transformer
        self.possible_actions = np.linspace(25, 190, 34)
        self.state_action_pairs = np.zeros((1000, 34, 34))
        self.sa_times_seen = np.zeros((1000, 34, 34))
        self.curr_x = 0
        self.total_turns = []
        # state_dict architecture: upper_level is height (so number of turns),
        # second is positions of objects in bottom-up height order, third is actions,
        # final level is reward.

    def perform_iteration(self):
        self.env = Environment()
        success_bool = True
        first_loop = True
        turn = -1
        while success_bool:
            turn += 1
            action = self.choose_move(first_loop, turn)
            obj_list, sizes, reward, success_bool = self.env.play_action(action, (40, 40))
            self.obj_list = obj_list
            curr_obj = ft.transform(obj_list[-1].position[0])
            self.curr_x = curr_obj
            if first_loop:
                next_move = [curr_obj, ft.transform(action), reward]
                first_loop = False
            else:
                update_move = next_move[:]
                next_move = [curr_obj, ft.transform(action), reward]
                self.update_val(next_move, update_move, turn)

            if turn == 999:
                success_bool = False
        if turn == 0:
            update_move = next_move[:]
            update_move[1] = ft.transform(update_move[1])
        self.total_turns.append(turn)
        self.final_value_update(reward, update_move, turn)

    def choose_move(self, first_loop, turn):
        prev_reward = float('-inf')
        action = None
        np.random.shuffle(self.possible_actions)
        if random.uniform(0, 1) > self.above_epsilon or not first_loop:
            for move, move_reward in enumerate(self.state_action_pairs[turn][self.curr_x]):
                if move_reward > prev_reward:
                    action = self.possible_actions[move]
                    prev_reward = move_reward
        else:
            action = np.random.choice(self.possible_actions)
        return action


    def update_val(self, next_move, update_move, turn):
        self.sa_times_seen[turn - 1][update_move[0]][update_move[1]] += 1

        step_size = 1 / self.sa_times_seen[turn-1][update_move[0]][update_move[1]]
        update_return = self.state_action_pairs[turn-1][update_move[0]][update_move[1]]
        next_return = self.state_action_pairs[turn][next_move[0]][next_move[1]]
        next_reward = next_move[2]
        self.state_action_pairs[turn-1][update_move[0]][update_move[1]] = \
            update_return + step_size * (next_reward + self.gamma * next_return - update_return)

    def final_value_update(self, reward, update_move, turn):
        self.sa_times_seen[turn - 1][update_move[0]][update_move[1]] += 1
        step_size = 1 / self.sa_times_seen[turn - 1][update_move[0]][update_move[1]]
        update_return = self.state_action_pairs[turn - 1][update_move[0]][update_move[1]]
        next_return = 0
        self.state_action_pairs[turn - 1][update_move[0]][update_move[1]] = \
            update_return + step_size * (reward + self.gamma * next_return - update_return)

    def update_screen(self):
        screen.fill((0, 0, 0, 0))
        iterator = 0
        for iterator, body in enumerate(self.env.body_list):
            #print(len(self.env.body_list))
            for fixture in body.fixtures:
                # The fixture holds information like density and friction,g
                # and also the shape.
                shape = fixture.shape

                # Naively assume that this is a polygon shape. (not good normally!)
                # We take the body's transform and multiply it with each
                # vertex, and then convert from meters to pixels with the scale
                # factor.
                vertices = [(body.transform * v) * PPM for v in shape.vertices]

                # But wait! It's upside-down! Pygame and Box2D orient their
                # axes in different ways. Box2D is just like how you learned
                # in high school, with positive x and y directions going
                # right and up. Pygame, on the other hand, increases in the
                # right and downward directions. This means we must flip
                # the y components.
                vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]

                pygame.draw.polygon(screen, self.env.colour_list[iterator], vertices)
        SARSA_agent.env.world.Step(TIME_STEP, 10, 10)

        # Flip the screen and try to keep at the target FPS
        pygame.display.flip()
        clock.tick(TARGET_FPS)


if __name__ == '__main__':
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 480
    PPM = 1.0
    TARGET_FPS = 120
    TIME_STEP = 1.0 / TARGET_FPS
    ft = FeatureTransform()
    # screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), display=1)
    # pygame.display.set_caption('test')
    # clock = pygame.time.Clock()
    SARSA_agent = Q_learner(0.9, 0.1, ft)
    mean_list = []
    for i in range(10000):
        SARSA_agent.perform_iteration()
        mean_list.append(np.mean(SARSA_agent.total_turns))
        # SARSA_agent.update_screen()
    for i in SARSA_agent.obj_list:
        print(i.position[0])
    plt.plot(mean_list)
    plt.show()
    plt

