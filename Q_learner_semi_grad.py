import random
from stack_env import Environment
import numpy as np
import matplotlib.pyplot as plt
import pygame

from features import FeatureConverter


class Q_learner:

    def __init__(self, gamma, above_epsilon, max_stack, step_size):
        self.gamma = gamma
        self.above_epsilon = above_epsilon
        self.curr_x = 0
        self.possible_actions = np.linspace(160, 480, 50)
        self.fc = FeatureConverter(max_stack, self.possible_actions)
        self.max_turns = max_stack - 1
        self.total_turns = []
        self.biggest_change = float('-inf')
        self.step_size = step_size
        self.all_rewards = []
        # state_dict architecture: upper_level is height (so number of turns),
        # second is positions of objects in bottom-up height order, third is actions,
        # final level is reward.

    def perform_iteration(self, iter):
        self.env = Environment()
        self.biggest_change = float('-inf')
        success_bool = True
        first_loop = True
        turn = -1
        self.curr_state = []
        curr_reward = []
        self.curr_state = [self.env.body_list[0].position]
        while success_bool:
            turn += 1
            action = self.choose_move(first_loop, turn)
            obj_list, sizes, reward, success_bool = self.env.play_action(action, (40, 40))
            self.curr_state = []
            for box in obj_list:
                self.curr_state.append(box.position)
            #print(self.curr_state)
            if first_loop:
                next_move = [reward, self.curr_state, action]
                first_loop = False
            else:
                update_move = next_move[:]
                next_move = [reward, self.curr_state, action]
                self.update_val(next_move, update_move, turn)
            if turn == self.max_turns:
                success_bool = False
            curr_reward.append(reward)
            #if iter % 100 == 0:
                #self.update_screen()

        if turn == 0:
            update_move = next_move[:]

        self.all_rewards.append(max(curr_reward))
        self.total_turns.append(turn)
        self.final_value_update(reward, update_move, turn)

    def choose_move(self, first_loop, turn):
        prev_reward = float('-inf')
        action = None
        if not first_loop and random.uniform(0, 1) > self.above_epsilon:
            for move in self.possible_actions:
                move_reward = self.fc.predict(self.curr_state, move, turn)
                if move_reward > prev_reward:
                    action = move
                    prev_reward = move_reward
        else:

            action = np.random.choice(self.possible_actions)

        return action

    def update_val(self, next_move, update_move, turn):
        old_theta = self.fc.theta.copy()

        update_g = self.step_size * (next_move[0] + self.gamma * self.fc.predict(next_move[1], next_move[2], turn) -
                                      self.fc.predict(update_move[1], update_move[2], turn)) * \
                         self.fc.grad(update_move[1], update_move[2], turn)
        self.fc.theta += update_g
        self.biggest_change = max(self.biggest_change, np.abs(self.fc.theta - old_theta).sum())

    def final_value_update(self, reward, update_move, turn):
        old_theta = self.fc.theta.copy()
        update_g = self.step_size * (reward - self.fc.predict(update_move[1], update_move[2], turn)) * \
                         self.fc.grad(update_move[1], update_move[2], turn)
        self.fc.theta += update_g
        self.biggest_change = max(self.biggest_change, np.abs(self.fc.theta - old_theta).sum())

    def update_screen(self):
        screen.fill((0, 0, 0, 0))
        iterator = 0
        for iterator, body in enumerate(self.env.body_list):
            # print(len(self.env.body_list))
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
    TARGET_FPS = 60
    TIME_STEP = 1.0 / TARGET_FPS
    #screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), display=1)
    #pygame.display.set_caption('test')
    #clock = pygame.time.Clock()
    SARSA_agent = Q_learner(0.1, 0.01, 100, 1e-7)
    mean_list = []
    changes = []
    all_vals = []
    for i in range(10000):
        if True:
            SARSA_agent.perform_iteration(i)
            mean_list.append(np.mean(SARSA_agent.all_rewards))
            all_vals.append(SARSA_agent.total_turns)
            changes.append(SARSA_agent.biggest_change)
            if i % 100 == 0:
                print(i)
            if i % 1000 == 0 and i != 0:
                plt.plot(mean_list)
                plt.show()
    print(SARSA_agent.fc.theta)
    plt.plot(mean_list)
    plt.show()
    plt.plot(changes)
    plt.show()
