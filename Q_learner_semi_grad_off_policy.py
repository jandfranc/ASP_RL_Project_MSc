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
        self.possible_actions = np.linspace(160, 480, 10)
        self.fc = FeatureConverter(max_stack, self.possible_actions)
        self.max_turns = max_stack - 1
        self.total_turns = []
        self.biggest_change = float('-inf')
        self.step_size = step_size
        self.all_rewards = []
        self.previous_action = 'EEEE'
        self.mean_reward = 0
        # state_dict architecture: upper_level is height (so number of turns),
        # second is positions of objects in bottom-up height order, third is actions,
        # final level is reward.

    def train(self, iter):
        self.env = Environment()
        self.biggest_change = float('-inf')
        success_bool = True
        first_loop = True
        turn = -1
        self.curr_state = []
        curr_reward = []
        self.init_state = self.env.body_list[0].position
        updates = []

        while success_bool:
            turn += 1
            action = self.choose_move_train(first_loop, turn)
            obj_list, sizes, reward, success_bool_set = self.env.play_action(action, (32, 32))
            reward = reward - self.mean_reward
            self.curr_state = [self.init_state]
            for box in obj_list[1:]:
                self.curr_state.append(box.position)
            # print(self.curr_state)
            updates.append([reward, self.curr_state, action])

            if turn == self.max_turns:
                success_bool = False
            curr_reward.append(reward)
            # if iter % 100 == 0:
                # self.update_screen()
            success_bool = success_bool_set

        first_update = True
        updates.reverse()
        for update in updates:
            if first_update:
                self.final_value_update(reward, update, turn)
                next_move = update
            else:
                self.update_val(next_move, update, turn)
                next_move = update
        #self.all_rewards.append(max(curr_reward))
        #self.total_turns.append(turn)
        #self.final_value_update(reward, update_move, turn)

    def test(self, iter, show_screen, action):
        self.env = Environment()
        self.biggest_change = float('-inf')
        success_bool = True
        first_loop = True
        turn = -1
        self.curr_state = []
        curr_reward = []
        self.init_state = self.env.body_list[0].position
        while success_bool:
            turn += 1
            action = self.choose_move_test(first_loop, turn, action)
            obj_list, sizes, reward, success_bool_set = self.env.play_action(action, (32, 32))
            self.curr_state = [self.init_state]
            first_loop = False
            for box in obj_list[1:]:
                self.curr_state.append(box.position)
            if turn == self.max_turns:
                success_bool = False
            curr_reward.append(reward)
            # if iter % 100 == 0:
            if show_screen:
                self.update_screen()
            success_bool = success_bool_set
        self.all_rewards.append(max(curr_reward))
        self.total_turns.append(turn)

    def choose_move_train(self, first_loop, turn):
        possible_actions = self.possible_actions.tolist()
        if not first_loop and random.uniform(0, 1) > self.above_epsilon:
            action = False
            prev_reward = float('-inf')
            for move in possible_actions:
                move_reward = self.fc.predict(self.curr_state, move, turn)
                if move_reward > prev_reward:
                    action = move
                    prev_reward = move_reward
        else:

            action = np.random.choice(self.possible_actions)

        self.previous_action = action

        return action

    def choose_move_test(self, first_loop, turn, action):
        possible_actions = self.possible_actions.tolist()
        if not first_loop:
            action = False
            prev_reward = float('-inf')
            for move in possible_actions:
                move_reward = self.fc.predict(self.curr_state, move, turn)
                if move_reward > prev_reward:
                    action = move
                    prev_reward = move_reward
        else:

            action = self.possible_actions[action]

        self.previous_action = action

        return action

    def update_val(self, next_move, update_move, turn):
        old_theta = self.fc.theta.copy()
        #max_action = float('-inf')
        #for move in self.possible_actions:
        #    a = self.fc.predict(next_move[1], move, turn)
        #    if a > max_action:
        #        max_action = a

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

    SARSA_agent = Q_learner(0.9, 0.1, 1000, 1e-3)
    mean_list = []
    changes = []
    all_vals = []
    for i in range(1000000):
        if True:
            SARSA_agent.train(i)
            #mean_list.append(np.mean(SARSA_agent.all_rewards[-500:-1]))
            #all_vals.append(SARSA_agent.total_turns)
            #changes.append(SARSA_agent.biggest_change)
            if i % 50000 == 0 and i != 1:
                total_test = 10
                show_screen = True
                screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), display=1)
                pygame.display.set_caption('test')
                clock = pygame.time.Clock()
                plt.plot(all_vals)
                plt.show()
                print(SARSA_agent.fc.theta)
            else:
                total_test = 10
                show_screen = False
            if i % 100 == 0 and i != 1:
                #print('beginning test')
                curr_means = []
                for iterator in range(total_test):
                    SARSA_agent.test(i, show_screen, iterator)
                    curr_means.append(int(SARSA_agent.all_rewards[-1]))
                mean_list.append(np.mean(curr_means))
                all_vals.append(np.mean(mean_list[-100:-1]))
                # updates mean reward so negative rewards become given for vales lower than test mean
                #SARSA_agent.mean_reward = all_vals[-1]

                print(i)
                print((mean_list[-1], max(curr_means)))
                if show_screen:
                    pygame.display.quit()

            #if i % 10000 == 0:
             #   plt.plot(all_vals)
              #  plt.show()
    print(SARSA_agent.fc.theta)
    plt.plot(all_vals)
    plt.show()
    plt.plot(changes)
    plt.show()
