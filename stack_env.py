import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)

import Box2D
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody)
import random

PPM = 25.0
TARGET_FPS = 120
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 1080, 1920


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
        self.init_height = 40
        self.colour_list.append((255, 255, 255, 255))

    def add_box(self, position, density, friction, angle, size):
        self.body_list.append(self.world.CreateDynamicBody(position=position, angle=angle))
        box = self.body_list[-1].CreatePolygonFixture(box=size, density=density, friction=friction)
        self.item_sizes.append(size)
        self.colour_list.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    def reset_world(self):
        for i_body in range(len(self.body_list)):
            if i_body != 0:
                self.world.DestroyBody(self.body_list[i_body])
        self.body_list = [self.ground_body]
        self.height_reward = 0
        self.item_sizes = []

    def play_action(self, x_pos, box):
        self.add_box((x_pos, self.init_height), 1, 1, 0, box)
        self.init_height += box[1]
        height_list = []
        success_bool = True
        for n_body in self.body_list:
            height_list.append(n_body.position[1])
        for _ in range(1000):
            self.world.Step(TIME_STEP, 10, 10)
        for i_body, n_body in enumerate(self.body_list):
            if n_body.position[1] < height_list[i_body] - 5:
                success_bool = False
                reward = -10000000
                prev_list = self.body_list[1:]
                prev_sizes = self.item_sizes[:]
                self.reset_world()
                return prev_list, prev_sizes, reward, success_bool
        curr_pos = float('-inf')
        for box in self.body_list:
            if box.position[1] > curr_pos:
                self.height_reward = box.position[1]
                curr_pos = box.position[1]
        return self.body_list, self.item_sizes, self.height_reward, success_bool


if __name__ == '__main__':
    env = Environment()
    i = 40
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), display=1, flags=pygame.FULLSCREEN)
    pygame.display.set_caption('test')
    clock = pygame.time.Clock()
    while True:
        '''
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == P):
                # The user closed the window or pressed escape
                running = False

        i += 1
        if i % 50 == 0:
            env.add_box((random.randint(25, 191), i), i*i*i, 1, random.randint(0, 360), (5, 5))
        if i % 1251 == 0:
            env.add_box((108, i/3), i*i*i, 1, random.randint(0, 360), (50, 50))
        '''
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