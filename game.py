import pygame
import random
import os


class Game:
    # ============= 资源文件 ==============
    # base_path
    BASE_PATH = os.path.split(os.path.abspath(__file__))[0]
    # 背景图片（上半部分，黑色背景）
    BACKGROUND_PATH = os.path.join(BASE_PATH, 'resources/images/background-black.png')
    # 背景图片（下半部分，绿色草坪）
    LAWN_PATH = os.path.join(BASE_PATH, 'resources/images/lawn.png')
    # 小鸟飞：上 中 下
    BIRD_PATH = (
        os.path.join(BASE_PATH, 'resources/images/redbird-upflap.png'),
        os.path.join(BASE_PATH, 'resources/images/redbird-midflap.png'),
        os.path.join(BASE_PATH, 'resources/images/redbird-downflap.png')
    )
    # 管道
    PIPE_PATH = os.path.join(BASE_PATH, 'resources/images/pipe-green.png')

    # 加载资源文件
    def __load_resources(cls):

        # 初始化容器
        images, hit_masks = {}, {}
        # ---------------  图片  -----------------
        # 背景
        images['background'] = pygame.image.load(cls.BACKGROUND_PATH).convert()
        # 草坪
        images['lawn'] = pygame.image.load(cls.LAWN_PATH).convert_alpha()
        # 小鸟
        images['bird'] = [
            pygame.image.load(cls.BIRD_PATH[i]).convert_alpha() for i in range(len(cls.BIRD_PATH))
        ]
        # 管道，上下两根
        images['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(cls.PIPE_PATH).convert_alpha(), 180),
            pygame.image.load(cls.PIPE_PATH).convert_alpha(),
        )

        # ---------------  撞击域  -----------------
        def get_hit_mask(image):
            mask = []
            for x in range(image.get_width()):
                mask.append([])
                for y in range(image.get_height()):
                    # 最后一维 为透明度
                    # (255, 255, 255, 0) 0，透明，不发生碰撞
                    # (84, 56, 71, 255) 非0，不透明，碰撞
                    mask[x].append(bool(image.get_at((x, y))[3]))
            return mask

        # 小鸟
        hit_masks['bird'] = [get_hit_mask(images['bird'][i]) for i in range(len(images['bird']))]
        # 管道
        hit_masks['pipe'] = [get_hit_mask(images['pipe'][i]) for i in range(len(images['pipe']))]

        return images, hit_masks

    def __init__(self, computer_play=True, use_extract_reward=True):
        self.screen_width = 288
        self.screen_height = 512
        # pygame设定
        self.fps = 30
        pygame.init()
        self.fps_clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Flappy Bird')
        # 管道上下最小间距，理论上，要允许让 bird 通过
        self.pipe_min_gap = 100
        # 草坪的位置
        self.lawn_y = self.screen_height * 0.79
        self.images, self.hit_masks = self.__load_resources()
        # 三只不同形态的鸟 宽高一致，故不分
        self.bird_width = self.images['bird'][0].get_width()
        self.bird_height = self.images['bird'][0].get_height()
        self.pipe_width = self.images['pipe'][0].get_width()
        self.pipe_height = self.images['pipe'][0].get_height()
        # 速度设定
        self.pipe_vel_x = -4  # 管道 x轴 固定移动速度，y轴不动
        self.bird_vel_y = 0  # bird 当前速度，刚进入时为0
        self.bird_max_vel_y = 10  # bird 最大自由落体速度（下的快)
        self.bird_min_vel_y = -8  # bird 最大向上flappy速度（上的慢)
        self.bird_acc_y = 1  # bird 自由下落的加速度
        self.bird_flap_acc = -9  # bird 拍拍翅膀的加速度
        # 电脑自己玩，还是人玩
        self.computer_play = computer_play
        # 是否使用 启发式 reward
        self.use_extract_reward = use_extract_reward
        # 游戏开始
        self.__start()

    # 随机生成一根管道
    def __get_random_pipe(self):
        # init
        pipe_y = int(self.lawn_y * 0.2)  # 管道起始位置，相当于 在草坪上方区域 0.2 的位置

        # random
        random_ys = [20, 30, 40, 50, 60, 70, 80, 90]
        index = random.randint(0, len(random_ys) - 1)
        pipe_y += random_ys[index]

        pipe_x = self.screen_width + 10  # 在屏幕外等着，暂未展示

        return [
            {'x': pipe_x, 'y': pipe_y - self.pipe_height},  # 上管道坐标
            {'x': pipe_x, 'y': pipe_y + self.pipe_min_gap},  # 下管道坐标
        ]

    # 开始游戏
    def __start(self):
        self.score = 0
        self.bird_index = 0

        # 起始 bird 位置
        self.bird_x = int(self.screen_width * 0.2)
        self.bird_y = int((self.screen_height - self.bird_height) / 2)

        # 初始生成两组管道，这里主要利用其 y， x 相对固定
        init_pipe_1 = self.__get_random_pipe()
        init_pipe_2 = self.__get_random_pipe()
        self.upper_pipes = [
            {'x': self.screen_width, 'y': init_pipe_1[0]['y']},
            {'x': self.screen_width + (self.screen_width / 2), 'y': init_pipe_2[0]['y']},
        ]
        self.lower_pipes = [
            {'x': self.screen_width, 'y': init_pipe_1[1]['y']},
            {'x': self.screen_width + (self.screen_width / 2), 'y': init_pipe_2[1]['y']},
        ]

    # 启发性 reward
    def __get_extra_reward(self):
        reward = 0

        # 1. bird 位于下一目标管道的中间
        # 前方最近的pipe
        right_nearest_pipe = None
        for pipe in self.upper_pipes:
            if pipe['x'] > self.bird_x + self.bird_width:
                distance = pipe['x'] - (self.bird_x + self.bird_width)
                if right_nearest_pipe is None or right_nearest_pipe['x'] - (self.bird_x + self.bird_width) > distance:
                    right_nearest_pipe = pipe

        min_distance = right_nearest_pipe['x'] - (self.bird_x + self.bird_width)
        # bird 在两管道之间¬
        if right_nearest_pipe['y'] + self.pipe_height < self.bird_y < right_nearest_pipe[
            'y'] + self.pipe_height + self.pipe_min_gap - self.bird_height:
            pos_reward = 0.5 * (1 - 1.0 * min_distance / self.screen_width / 2)
        else:
            pos_reward = -0.5 * (1 - 1.0 * min_distance / self.screen_width / 2)
        reward += pos_reward

        # 2. bird 的速度不能太快，太快不容易刹车
        speed_reward = -0.5 * abs(self.bird_vel_y) / 10
        reward += speed_reward

        return reward

    def __check_crash(self, bird, upper_pipes, lower_pipes):
        bird_index = bird['index']
        bird_x = bird['x']
        bird_y = bird['y']

        def pixel_collision(rect1, rect2, hit_mask1, hit_mask2):
            # 两个矩形不相交，直接pass
            rect = rect1.clip(rect2)

            if rect.width == 0 or rect.height == 0:
                return False

            x1, y1 = rect.x - rect1.x, rect.y - rect1.y
            x2, y2 = rect.x - rect2.x, rect.y - rect2.y

            for x in range(rect.width):
                for y in range(rect.height):
                    if hit_mask1[x1 + x][y1 + y] and hit_mask2[x2 + x][y2 + y]:
                        return True
            return False

        # 撞到屋顶
        if bird_y == 0:
            return True
        # 撞到地上
        elif bird_y + self.bird_height >= self.lawn_y:
            return True
        else:
            bird_rect = pygame.Rect(bird_x, bird_y, self.bird_width, self.bird_height)
            for u_pipe, l_pipe in zip(upper_pipes, lower_pipes):

                # upper and lower pipe rects
                u_pipe_rect = pygame.Rect(u_pipe['x'], u_pipe['y'], self.pipe_width, self.pipe_height)
                l_pipe_rect = pygame.Rect(l_pipe['x'], l_pipe['y'], self.pipe_width, self.pipe_height)

                # player and upper/lower pipe hitmasks
                bird_hit_mask = self.hit_masks['bird'][bird_index]
                u_hitmask = self.hit_masks['pipe'][0]
                l_hitmask = self.hit_masks['pipe'][1]

                # if bird collided with upipe or lpipe
                u_collide = pixel_collision(bird_rect, u_pipe_rect, bird_hit_mask, u_hitmask)
                l_collide = pixel_collision(bird_rect, l_pipe_rect, bird_hit_mask, l_hitmask)

                if u_collide or l_collide:
                    return True

        return False

    def action_and_reward(self, input_actions):
        if self.computer_play:
            # 电脑自己玩的时候，需要将 其他 触发时间 干掉，否则 白屏卡住
            pygame.event.pump()

        # input_actions[0] == 1: 不动
        # input_actions[1] == 1: 拍拍翅膀
        if input_actions[1] == 1:
            # 拍翅膀
            if self.bird_vel_y > self.bird_min_vel_y:
                self.bird_vel_y = max(self.bird_vel_y + self.bird_flap_acc, self.bird_min_vel_y)
        else:
            # 没拍翅膀，自由落体
            if self.bird_vel_y < self.bird_max_vel_y:
                self.bird_vel_y = min(self.bird_vel_y + self.bird_acc_y, self.bird_max_vel_y)

        # 更新当前 bird y轴 坐标
        # min 不可能超出 草坪 边界！
        # max 不可能突破天际！
        self.bird_y = max(min(self.bird_y + self.bird_vel_y, self.lawn_y - self.bird_height), 0)

        # 总共三种姿态的bird
        self.bird_index = (self.bird_index + 1) % 3

        # 管道左移
        for uPipe, lPipe in zip(self.upper_pipes, self.lower_pipes):
            uPipe['x'] += self.pipe_vel_x
            lPipe['x'] += self.pipe_vel_x

        # 新增管道，当第一个管道快消失时，考虑添加新管道
        if 0 < self.upper_pipes[0]['x'] < 5:
            new_pipe = self.__get_random_pipe()
            self.upper_pipes.append(new_pipe[0])
            self.lower_pipes.append(new_pipe[1])

        # 第一个管子消失
        if self.upper_pipes[0]['x'] < -self.pipe_width:
            self.upper_pipes.pop(0)
            self.lower_pipes.pop(0)

        # 计算reward
        terminal = False
        reward = 0.1
        if self.use_extract_reward:
            reward = self.__get_extra_reward()

        # 检查当前是否越过管道，并更新 reward
        bird_mid_pos_x = self.bird_x + self.bird_width / 2
        for pipe in self.upper_pipes:
            pipe_mid_pos_x = pipe['x'] + self.pipe_width / 2
            # 刚好越过管道
            if pipe_mid_pos_x <= bird_mid_pos_x < pipe_mid_pos_x + 4:
                self.score += 1
                reward += 1
        # 检查是否撞到管道
        is_crash = self.__check_crash({'x': self.bird_x, 'y': self.bird_y,
                                       'index': self.bird_index},
                                      self.upper_pipes, self.lower_pipes)
        # 理论上不需要 final score，这里作为衡量模型收敛的参考数据，提供给外界
        final_score = None
        if is_crash:
            terminal = True
            # 装到脑袋后，不好考虑其他 reward 的影响，直接置为-1
            reward = -1
            final_score = self.score
            # 立刻重新开始
            self.__start()

        # 渲染
        # 背景
        self.screen.blit(self.images['background'], (0, 0))
        # 柱子
        for uPipe, lPipe in zip(self.upper_pipes, self.lower_pipes):
            self.screen.blit(self.images['pipe'][0], (uPipe['x'], uPipe['y']))
            self.screen.blit(self.images['pipe'][1], (lPipe['x'], lPipe['y']))

        # 草坪
        self.screen.blit(self.images['lawn'], (0, self.lawn_y))
        # bird
        self.screen.blit(self.images['bird'][self.bird_index],
                         (self.bird_x, self.bird_y))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        self.fps_clock.tick(self.fps)
        return image_data, reward, terminal, final_score


if __name__ == '__main__':
    from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_SPACE, K_UP, K_RIGHT
    import sys

    game = Game(False)
    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                image_data, reward, terminal, final_score = game.action_and_reward([0, 1])
            elif event.type == KEYDOWN and event.key == K_RIGHT:
                image_data, reward, terminal, final_score = game.action_and_reward([1, 0])
