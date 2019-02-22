from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model
from collections import deque
import os
from game import Game
import numpy as np
import cv2
import random
import time


class Brain:
    def __init__(self, cnn_plus):
        self.cnn_plus = cnn_plus
        self.epsilon_init = 0.01  # epsilon 的 init
        self.epsilon_min = 0.0001  # epsilon 的 MIN
        self.epsilon = self.epsilon_init  # epsilon 代表采取随机行为的阈值
        self.epsilon_decay_step = 300000  # epsilon 衰减的总步数
        self.observation_step = 50000  # 观察多少步后，开始降低随机性
        self.memory_size = 50000  # 记忆容量
        self.gamma = 0.99  # 未来奖励 折扣系数
        self.memory = deque()
        self.batch_size = 32
        if self.cnn_plus:
            self.model_path = 'resources/model/cnn+.model'
        else:
            self.model_path = 'resources/model/cnn.model'
        self.step = 0  # 训练到第几步
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        else:
            self.model = self.build_model()

        # 辅助信息
        # 结束局=bird撞墙身亡
        self.best_score = 0

    def build_model(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=4, strides=(4, 4), input_shape=(80, 80, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, kernel_size=2, strides=(2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(256, activation='relu'))

        model.add(Dense(2, activation='linear'))
        model.summary()

        model.compile(loss='mse', optimizer='adam')
        return model

    def image2frame(self, image_data):
        # 1.缩放（可以不用，考虑到资源有限还是缩放比较好）
        resize_image_data = cv2.resize(image_data, (80, 80))
        # 2.转化成灰色图片
        gray_image_data = cv2.cvtColor(resize_image_data, cv2.COLOR_BGR2GRAY)
        # 3.转化成二值图片（供神经网络训练用）
        _, binary_image_data = cv2.threshold(gray_image_data, 1, 1, cv2.THRESH_BINARY)
        return binary_image_data

    def set_init_state(self, image_data):
        current_frame = self.image2frame(image_data)
        # 堆叠4帧图像到同一（x,y）上
        self.current_state = np.stack((current_frame, current_frame, current_frame, current_frame), axis=2)

    def get_action(self):
        action = np.zeros(2)
        if random.random() <= self.epsilon:
            # 采取真随机行为
            action_index = random.randrange(2)
            action[action_index] = 1
        else:
            # 采取经验行为
            result = self.model.predict(self.current_state[np.newaxis, :, :, :])[0]
            action_index = np.argmax(result)
            action[action_index] = 1
        return action

    def set_perception(self, action, image_data, reward, terminal, final_score):
        # 额外统计信息
        self.extra(terminal, final_score)

        # 取出当前一帧
        next_frame = self.image2frame(image_data)
        # 当前一帧做第一帧，后面三帧从原来取
        next_state = np.append(np.reshape(next_frame, (80, 80, 1)), self.current_state[:, :, :3], axis=2)
        self.memory.append((self.current_state, action, next_state, reward, terminal))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()
        if self.step > self.observation_step:
            self.train_model()
        self.current_state = next_state
        self.step += 1

    def train_model(self):
        # 逐步降低 epsilon，从而减少随机性
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.epsilon_init - self.epsilon_min) / self.epsilon_decay_step
        # 可能 sample 不出 BATCH 个样本
        mini_batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        X = np.zeros((len(mini_batch), 80, 80, 4))
        y = np.zeros((len(mini_batch), 2))
        # replay experience
        for i in range(0, len(mini_batch)):
            memory_state = mini_batch[i][0]
            action_index = np.argmax(mini_batch[i][1])
            next_memory_state = mini_batch[i][2]
            reward = mini_batch[i][3]
            terminal = mini_batch[i][4]
            X[i] = memory_state
            if terminal:
                y[i, action_index] = reward
            else:
                Q = self.model.predict(next_memory_state[np.newaxis, :, :, :])[0]
                y[i, action_index] = reward + self.gamma * np.max(Q)

        loss = self.model.train_on_batch(X, y)

        if self.step % 100 == 0:
            self.model.save(self.model_path)

    def extra(self, terminal, final_score):
        def format_time(t):
            return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))
        if terminal and final_score is not None:
            if final_score > self.best_score:
                self.best_score = final_score
                print(format_time(time.time()), self.step, "best score", self.best_score)



def play_bird(cnn_plus):
    # 1. 实例化 brain
    brain = Brain(cnn_plus)

    # 2. 创建游戏
    game = Game(use_extract_reward=cnn_plus)

    # 3. 玩游戏
    # 3.1. 初始化状态
    action = np.array([1, 0])
    image_data, reward, terminal, final_score = game.action_and_reward(action)
    brain.set_init_state(image_data)

    # 3.2 开始玩
    while True:
        action = brain.get_action()
        image_data, reward, terminal, final_score = game.action_and_reward(action)
        brain.set_perception(action, image_data, reward, terminal, final_score)


if __name__ == '__main__':
    cnn_plus = True
    play_bird(cnn_plus)
