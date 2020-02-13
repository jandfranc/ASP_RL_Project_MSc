import random
import numpy as np
import matplotlib.pyplot as plt

from features import FeatureConverter
import pygame
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody)
import pickle

class Environment:

    def __init__(self):
        self.item_sizes = []
        self.height_reward = 0
        self.world = world(gravity=(0, -1000), sleep=True)
        self.colour_list = []
        self.ground_body = self.world.CreateStaticBody(
            position=(320, 20),
            shapes=polygonShape(box=(640, 30)),
            angle=0
        )
        self.body_list = [self.ground_body]
        self.init_height = 82
        self.colour_list.append((255, 255, 255, 255))

    def add_box(self, position, density, friction, angle, size):
        self.body_list.append(self.world.CreateDynamicBody(position=position, angle=angle))
        box = self.body_list[-1].CreatePolygonFixture(box=size, density=density, friction=friction)
        self.item_sizes.append(size)
        self.colour_list.append((100, 100, 100, 100))

    def activate_wind(self):
        if len(self.body_list) > 1:
            for box in self.body_list[1:]:
                box.ApplyLinearImpulse((20, 0), box.position, True)

    def reset_world(self):
        for i_body in range(len(self.body_list)):
            if i_body != 0:
                self.world.DestroyBody(self.body_list[i_body])
        self.body_list = [self.ground_body]
        self.height_reward = 0
        self.item_sizes = []

    def play_action(self, x_pos, box):
        #edit error check
        high_box = 0

        for box_check in self.body_list[1:]:
            if x_pos - 33 < box_check.position[0] < x_pos + 33:
                if high_box < box_check.position[1]:
                    high_box = box_check.position[1]

        self.init_height = high_box + 64
        self.add_box((x_pos, self.init_height), 0.1, 0.5, 0, box)
        height_list = []
        success_bool = True
        for _ in range(100):
            self.world.Step(TIME_STEP, 10, 10)
            #SARSA_agent.update_screen()
        for n_body in self.body_list:
            height_list.append(n_body.position[1])
        #self.activate_wind()
        for _ in range(2000):
            #self.activate_wind()
            self.world.Step(TIME_STEP, 10, 10)
            #SARSA_agent.update_screen()
        for i_body, n_body in enumerate(self.body_list):
            if n_body.position[1] < height_list[i_body] - 5:
                success_bool = False
                prev_list = self.body_list
                prev_sizes = self.item_sizes[:]
                self.reset_world()
                return prev_list, prev_sizes, -1, success_bool
        curr_pos = float('-inf')
        return self.body_list, self.item_sizes, 1, success_bool

class Q_learner:

    def __init__(self, gamma, above_epsilon, max_stack, step_size):
        self.gamma = gamma
        self.above_epsilon = above_epsilon
        self.curr_x = 0
        self.possible_actions = np.linspace(176, 464, 10)
        self.fc = FeatureConverter(max_stack, self.possible_actions)
        self.max_turns = max_stack - 1
        self.total_turns = []
        self.biggest_change = float('-inf')
        self.step_size = step_size
        self.all_rewards = []
        self.previous_action = 'EEEE'
        # state_dict architecture: upper_level is height (so number of turns),
        # second is positions of objects in bottom-up height order, third is actions,
        # final level is reward.

    def train(self, iter):
        #screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), display=1)
        #pygame.display.set_caption('test')
        #clock = pygame.time.Clock()
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
            action = self.choose_move_train(first_loop, turn)
            obj_list, sizes, reward, success_bool = self.env.play_action(action, (32, 32))
            self.curr_state = [self.init_state]
            for box in obj_list[1:]:
                self.curr_state.append(box.position)
            # print(self.curr_state)
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
            # if iter % 100 == 0:
                # self.update_screen()

        if turn == 0:
            update_move = next_move[:]

        #self.all_rewards.append(max(curr_reward))
        #self.total_turns.append(turn)
        self.final_value_update(reward, update_move, turn)

    def test(self, iter, show_screen):
        #screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), display=1)
        #pygame.display.set_caption('test')
        #clock = pygame.time.Clock()
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
            action = self.choose_move_test(first_loop, turn)
            obj_list, sizes, reward, success_bool = self.env.play_action(action, (32, 32))
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
        self.all_rewards.append(max(curr_reward))
        self.total_turns.append(turn)

    def choose_move_train(self, first_loop, turn):
        possible_actions = self.possible_actions.tolist()
        if random.uniform(0, 1) > self.above_epsilon:
            action = False
            prev_reward = float('-inf')
            for move in possible_actions:
                move_reward = self.fc.predict(self.curr_state, move, turn)
                if move_reward > prev_reward:
                    action = move
                    prev_reward = move_reward
        else:

            action = np.random.choice(self.possible_actions)

        return action

    def choose_move_test(self, first_loop, turn):
        possible_actions = self.possible_actions.tolist()
        action = False
        prev_reward = float('-inf')
        for move in possible_actions:
            move_reward = self.fc.predict(self.curr_state, move, turn)
            if move_reward > prev_reward:
                action = move
                prev_reward = move_reward

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
    #screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), display=1)
    #pygame.display.set_caption('test')
    #clock = pygame.time.Clock()
    PPM = 1.0
    TARGET_FPS = 30
    TIME_STEP = 1.0 / TARGET_FPS

    SARSA_agent = Q_learner(0.9, 0.05, 1000, 1e-5)
    #with open('SARSA_fc.pickle', 'rb') as learner:
       # SARSA_agent.fc = pickle.load(learner)
    mean_list = []
    changes = []
    all_vals = []
    for i in range(1000000000):
        if True:
            SARSA_agent.train(i)
            #mean_list.append(np.mean(SARSA_agent.all_rewards[-500:-1]))
            #all_vals.append(SARSA_agent.total_turns)
            #changes.append(SARSA_agent.biggest_change)
            if i % 1000 == 0:
                with open(r"SARSA_fc.pickle", "wb") as output_file:
                    pickle.dump(SARSA_agent.fc, output_file)
            if i % 50000 == 0:
                show_screen = True
                screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), display=1)
                print(pygame.surfarray.array3d(screen))
                pygame.display.set_caption('test')
                clock = pygame.time.Clock()
                plt.plot(all_vals)
                plt.show()
                print(SARSA_agent.fc.theta)
            else:
                show_screen = False
            if i % 1000 == 0:
                print(i)
                curr_means = []
                SARSA_agent.test(i, show_screen)
                print(np.shape(pygame.surfarray.array2d(screen)))
                plt.show()
                curr_means.append(int(len(SARSA_agent.curr_state)))
                mean_list.append(np.mean(curr_means))
                all_vals.append(np.mean(mean_list[-100:-1]))
                # updates mean reward so negative rewards become given for vales lower than test mean
                #SARSA_agent.mean_reward = all_vals[-1]
                print(mean_list[-1])
                with open(r"all_means.pickle", "wb") as output_file:
                    pickle.dump(all_vals, output_file)
                if show_screen:
                    pygame.display.quit()
            '''
            if i % 50000 == 0:
                plt.plot(all_vals)
                plt.show()
            '''
    #print(SARSA_agent.fc.theta)
    plt.plot(all_vals)
    plt.show()
    plt.plot(changes)
    plt.show()
