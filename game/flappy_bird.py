import pygame
import random


# 加载图片、撞击域
def load():
    # 背景图片（上半部分，黑色背景）
    BACKGROUND_PATH = '../assets/sprites/background-black.png'
    # 背景图片（下半部分，绿色草坪）
    LAWN_PATH = '../assets/sprites/lawn.png'
    # 小鸟飞：上 中 下
    BIRD_PATH = (
        '../assets/sprites/redbird-upflap.png',
        '../assets/sprites/redbird-midflap.png',
        '../assets/sprites/redbird-downflap.png'
    )
    # 管道
    PIPE_PATH = '../assets/sprites/pipe-green.png'

    # 初始化容器
    IMAGES, SOUNDS, HITMASKS = {}, {}, {}
    # =====================  图片  =====================
    # 背景
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()
    # 草坪
    IMAGES['lawn'] = pygame.image.load(LAWN_PATH).convert_alpha()
    # 小鸟
    IMAGES['bird'] = [
        pygame.image.load(BIRD_PATH[i]).convert_alpha() for i in range(len(BIRD_PATH))
    ]
    # 管道，上下两根
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPE_PATH).convert_alpha(), 180),
        pygame.image.load(PIPE_PATH).convert_alpha(),
    )

    #  ===================== 撞击域 =====================
    # 小鸟
    HITMASKS['bird'] = [get_hit_mask(IMAGES['bird'][i]) for i in range(len(IMAGES['bird']))]
    # 管道
    HITMASKS['pipe'] = [get_hit_mask(IMAGES['pipe'][i]) for i in range(len(IMAGES['pipe']))]

    return IMAGES, HITMASKS


# 修改了原始的 行列遍历顺序，使 mask 符合人的直觉
def get_hit_mask(image):
    mask = []
    for y in range(image.get_height()):
        mask.append([])
        for x in range(image.get_width()):
            # 最后一维 为透明度
            # (255, 255, 255, 0) 0，透明，不发生碰撞
            # (84, 56, 71, 255) 非0，不透明，碰撞
            mask[y].append(bool(image.get_at((x, y))[3]))
    return mask


FPS = 30
SCREENWIDTH = 288
SCREENHEIGHT = 512

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')

IMAGES, HITMASKS = load()
PIPEMINGAP = 100  # 管道上下最小间距，理论上，要允许让 bird 通过
LAWN_Y = SCREENHEIGHT * 0.79  # 草坪的位置

BIRD_WIDTH = IMAGES['bird'][0].get_width()
BIRD_HEIGHT = IMAGES['bird'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()


class GameState:
    def __init__(self):
        self.score = 0
        self.birdIndex = 0

        # 起始 bird 位置
        self.bird_x = int(SCREENWIDTH * 0.2)
        self.bird_y = int((SCREENHEIGHT - BIRD_HEIGHT) / 2)

        # 初始生成两组管道，这里主要利用其 y，x 相对固定
        init_pipe_1 = get_random_pipe()
        init_pipe_2 = get_random_pipe()
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': init_pipe_1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': init_pipe_2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': init_pipe_1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': init_pipe_2[1]['y']},
        ]

        # 速度设定
        self.pipeVelX = -4  # 管道 x轴 固定移动速度，y轴不动
        self.birdVelY = 0  # bird 当前速度，刚进入时为0
        self.birdMaxVelY = 10  # bird 最大自由落体速度（下的快)
        self.birdMinVelY = -8  # bird 最大向上flappy速度（上的慢)
        self.birdAccY = 1  # bird 自由下落的加速度
        self.birdFlapAcc = -9  # bird 拍拍翅膀的加速度

    '''
        reward:
            0.1:默认, -1:挂了, 1:通过管道
        terminal:
            是否结束
    '''

    def frame_step(self, input_actions):

        pygame.event.pump()

        reward = 0.1  # 默认为 0.1
        terminal = False

        # 检查当前是否越过管道，并更新 reward
        birdMidPos = self.bird_x + BIRD_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_WIDTH / 2
            # 刚好越过管道
            if pipeMidPos <= birdMidPos < pipeMidPos + 4:
                self.score += 1
                reward = 10

        # input_actions[0] == 1: 不动
        # input_actions[1] == 1: 拍拍翅膀
        if input_actions[1] == 1:
            # 拍翅膀
            if self.birdVelY > self.birdMinVelY:
                self.birdVelY = max(self.birdVelY + self.birdFlapAcc, self.birdMinVelY)
        else:
            # 没拍翅膀，自由落体
            if self.birdVelY < self.birdMaxVelY:
                self.birdVelY = min(self.birdVelY + self.birdAccY, self.birdMaxVelY)

        # 更新当前 bird y轴 坐标
        # min 不可能超出 草坪 边界！
        # max 不可能突破天际！
        self.bird_y = max(min(self.bird_y + self.birdVelY, LAWN_Y - BIRD_HEIGHT), 0)

        # 总共三种姿态的bird
        self.birdIndex = (self.birdIndex + 1) % 3

        # 管道左移
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # 新增管道，当第一个管道快消失时，考虑添加新管道
        if 0 < self.upperPipes[0]['x'] < 5:
            new_pipe = get_random_pipe()
            self.upperPipes.append(new_pipe[0])
            self.lowerPipes.append(new_pipe[1])

        # 第一个管子消失
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # 检查是否撞到管道
        is_crash = check_crash({'x': self.bird_x, 'y': self.bird_y,
                                'index': self.birdIndex},
                               self.upperPipes, self.lowerPipes)

        if is_crash:
            terminal = True
            reward = -1
            print(self.score)
            # 立刻重新开始
            self.__init__()

        # 渲染
        # 背景
        SCREEN.blit(IMAGES['background'], (0, 0))
        # 柱子
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        # 草坪
        SCREEN.blit(IMAGES['lawn'], (0, LAWN_Y))
        # bird
        SCREEN.blit(IMAGES['bird'][self.birdIndex],
                    (self.bird_x, self.bird_y))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)

        return image_data, reward, terminal


# 随机生成管道
def get_random_pipe():
    # init
    pipeY = int(LAWN_Y * 0.2)  # 管道起始位置，相当于 在草坪上方区域 0.2 的位置

    # random
    randomYs = [20, 30, 40, 50, 60, 70, 80, 90]
    index = random.randint(0, len(randomYs) - 1)
    pipeY += randomYs[index]

    pipeX = SCREENWIDTH + 10  # 在屏幕外等着，暂未展示

    return [
        {'x': pipeX, 'y': pipeY - PIPE_HEIGHT},  # 上管道坐标
        {'x': pipeX, 'y': pipeY + PIPEMINGAP},  # 下管道坐标
    ]


def check_crash(bird, upperPipes, lowerPipes):
    bird_index = bird['index']
    bird_x = bird['x']
    bird_y = bird['y']
    bird_width = IMAGES['bird'][0].get_width()
    bird_height = IMAGES['bird'][0].get_height()

    # 撞到地上
    if bird_y + bird_height >= LAWN_Y:
        return True
    else:
        birdRect = pygame.Rect(bird_x, bird_y, bird_width, bird_height)
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            # player and upper/lower pipe hitmasks
            birdHitMask = HITMASKS['bird'][bird_index]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(birdRect, uPipeRect, birdHitMask, uHitmask)
            lCollide = pixelCollision(birdRect, lPipeRect, birdHitMask, lHitmask)

            if uCollide or lCollide:
                return True

    return False


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    # 两个矩形不相交，直接pass
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for y in range(rect.height):
        for x in range(rect.width):
            if hitmask1[y1 + y][x1 + x] and hitmask2[y2 + y][x2 + x]:
                return True
    return False


from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_SPACE, K_UP, K_RIGHT
import sys

if __name__ == '__main__':
    game_state = GameState()
    for i in range(5000):
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                image_data, reward, terminal = game_state.frame_step([0, 1])
            elif event.type == KEYDOWN and event.key == K_RIGHT:
                image_data, reward, terminal = game_state.frame_step([1, 0])
