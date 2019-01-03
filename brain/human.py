"""人工生成优质经验"""
from game.flappy_bird import GameState
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_SPACE, K_UP, K_RIGHT
import pygame
import sys
from dqn import image2state
import pickle

exp = []
game_state = GameState()
start = False
t = 1000
flag = True
while flag:
    for event in pygame.event.get():
        print(len(exp))
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
            image_data, reward, terminal = game_state.frame_step([0, 1])
            if not start:
                start = True
                state = image2state(image_data)
            else:
                state = image2state(image_data)
                exp.append((last_state, 1, state, reward, terminal))
            last_state = state
            if len(exp) == 1000:
                flag = False
        elif event.type == KEYDOWN and event.key == K_RIGHT:
            image_data, reward, terminal = game_state.frame_step([1, 0])
            if not start:
                start = True
                state = image2state(image_data)
            else:
                state = image2state(image_data)
                exp.append((last_state, 0, state, reward, terminal))
            last_state = state
            if len(exp) == 1000:
                flag = False

with open('../assets/exp.pkl', 'wb') as f:
    pickle.dump(exp, f)
