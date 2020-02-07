import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)

import Box2D
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody)
import random

PPM = 20.0
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 1000

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
pygame.display.set_caption('test')
clock = pygame.time.Clock()


class Environment:

    def __init__(self):
        self.world = world(gravity=(0, -10))
        self.colour_list = []
        ground_body = self.world.CreateStaticBody(
            position=(-5, 20),
            shapes=polygonShape(box=(50, 5)),
            angle = 15
        )
        self.body_list = [ground_body]

        ground_body = self.world.CreateStaticBody(
            position=(5, -5),
            shapes=polygonShape(box=(50, 5)),
            angle=-15
        )
        self.body_list.append(ground_body)
        self.colour_list.append((255, 255, 255, 255))
        self.colour_list.append((255, 255, 255, 255))

    def add_box(self, position, density, friction, angle):
        self.body_list.append(self.world.CreateDynamicBody(position=position, angle=angle))
        box = self.body_list[-1].CreatePolygonFixture(box=(0.01, 0.01), density=density, friction=friction)
        self.colour_list.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))


if __name__ == '__main__':
    env = Environment()
    i = 0
    colors = {
        staticBody: (255, 255, 255, 255),
        dynamicBody: (127, 127, 127, 255)
    }

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # The user closed the window or pressed escape
                running = False

        i += 1
        env.add_box((random.randint(0, 30), i/3), i, 0.3, 0)
        env.add_box((random.randint(0, 30), i / 3), i, 0.3, 0)
        env.add_box((random.randint(0, 30), i / 3), i, 0.3, 0)
        screen.fill((0, 0, 0, 0))
        iterator = 0
        for iterator, body in enumerate(env.body_list):
            for fixture in body.fixtures:
                # The fixture holds information like density and friction,
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