import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
import numpy as np
import Box2D
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody)
import random
import matplotlib.pyplot as plt

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 640
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), display=1)
pygame.display.set_caption('test')
clock = pygame.time.Clock()
PPM = 1.0
TARGET_FPS = 30
TIME_STEP = 1.0 / TARGET_FPS


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
        self.possible_actions = np.linspace(176, 464, 10)
        self.action_space = np.linspace(0, 9, 10)
        self.observation_space = self.reset_world()

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
        self.update_screen()
        return pygame.surfarray.array2d(screen).flatten()

    def step(self, action):
        #edit error check
        high_box = 0
        action = int(action)
        x_pos = self.possible_actions[action]
        for box_check in self.body_list[1:]:
            if x_pos - 33 < box_check.position[0] < x_pos + 33:
                if high_box < box_check.position[1]:
                    high_box = box_check.position[1]

        self.init_height = high_box + 64
        self.add_box((x_pos, self.init_height), 0.1, 0.5, 0, (32, 32))
        self.update_screen()
        height_list = []
        success_bool = True
        for _ in range(100):
            self.world.Step(TIME_STEP, 10, 10)

        for n_body in self.body_list:
            height_list.append(n_body.position[1])
        self.activate_wind()

        for _ in range(2000):
            #self.activate_wind()
            self.world.Step(TIME_STEP, 10, 10)
        self.update_screen()

        for i_body, n_body in enumerate(self.body_list):
            if n_body.position[1] < height_list[i_body] - 3:
                frame = pygame.surfarray.array2d(screen)
                return frame.flatten(), -10, True

        return pygame.surfarray.array2d(screen).flatten(), 1, False

    def update_screen(self):
        screen.fill((0, 0, 0, 0))
        iterator = 0
        for iterator, body in enumerate(self.body_list):
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

                pygame.draw.polygon(screen, self.colour_list[iterator], vertices)
        self.world.Step(TIME_STEP, 10, 10)

        # Flip the screen and try to keep at the target FPS
        pygame.display.flip()
        clock.tick(TARGET_FPS)

class Environment2:

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
        self.init_height = 74
        self.colour_list.append((255, 255, 255, 255))

    def add_box(self, position, density, friction, angle, size):
        self.body_list.append(self.world.CreateDynamicBody(position=position, angle=angle))
        box = self.body_list[-1].CreatePolygonFixture(box=size, density=density, friction=friction)
        self.item_sizes.append(size)
        self.colour_list.append((0, 100, 100, 0))

    def activate_wind(self):
        if len(self.body_list) > 1:
            for box in self.body_list[1:]:
                box.ApplyLinearImpulse((1000, 0), box.position, True)

    def reset_world(self):
        for i_body in range(len(self.body_list)):
            if i_body != 0:
                self.world.DestroyBody(self.body_list[i_body])
        self.body_list = [self.ground_body]
        self.height_reward = 0
        self.item_sizes = []

    def play_action(self, x_pos, box):
        self.add_box((x_pos, self.init_height), 1, 0.5, 0, box)
        height_list = []
        success_bool = True
        for _ in range(100):
            self.world.Step(TIME_STEP, 10, 10)
            SARSA_agent.update_screen()
        for n_body in self.body_list:
            height_list.append(n_body.position[1])
        #self.activate_wind()
        for _ in range(2000):
            self.world.Step(TIME_STEP, 10, 10)
            SARSA_agent.update_screen()
        for i_body, n_body in enumerate(self.body_list):
            if n_body.position[1] < height_list[i_body] - 5:
                success_bool = False
                reward = -100
                reward = len(self.body_list)-1
                prev_list = self.body_list[1:]
                prev_sizes = self.item_sizes[:]
                self.reset_world()
                return prev_list, prev_sizes, -10, success_bool
        curr_pos = float('-inf')
        for box in self.body_list:
            if box.position[1] > curr_pos:
                self.height_reward = box.position[1]
                curr_pos = box.position[1]
        self.init_height = int(curr_pos) + 64
        reward = ((self.height_reward-10)/64)**2
        return self.body_list, self.item_sizes, 1, success_bool


if __name__ == '__main__':
    env = Environment()
    i = 40
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), display=1, flags=pygame.FULLSCREEN)
    pygame.display.set_caption('test')
    clock = pygame.time.Clock()
    while True:

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == P):
                # The user closed the window or pressed escape
                running = False

        i += 1
        if i % 50 == 0:
            env.add_box((random.randint(20, 21), i/10 + 20), 1, 1, 0, (1, 1))
        if i % 1251 == 0:
            env.add_box((20, i/3), 1, 1, random.randint(0, 360), (50, 50))
        env.activate_wind()
        screen.fill((0, 0, 0, 0))
        iterator = 0
        for iterator, body in enumerate(env.body_list):
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

                pygame.draw.polygon(screen, env.colour_list[iterator], vertices)
        env.world.Step(TIME_STEP, 10, 10)

        # Flip the screen and try to keep at the target FPS
        pygame.display.flip()
        clock.tick(TARGET_FPS)