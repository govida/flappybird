from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import load_model
from collections import deque
import os
from game import Game
import numpy as np
import cv2
import random


class Brain:
    def __init__(self):
        self.epsilon_init = 0.1  # epsilon 的 init
        self.epsilon_min = 0.0001  # epsilon 的 MIN
        self.epsilon = self.epsilon_init  # epsilon 代表采取随机行为的阈值
        self.epsilon_decay_step = 300000  # epsilon 衰减的总步数
        self.observation_step = 50000  # 观察多少步后，开始降低随机性
        self.memory_size = 50000  # 记忆容量
        self.gamma = 0.99  # 未来奖励 折扣系数
        self.memory = deque()
        self.batch_size = 64
        self.model_path = '../resources/model/cnn.model'
        self.step = 0  # 训练到第几步
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        else:
            self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=4, strides=(4, 4), input_shape=(80, 80, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, kernel_size=2, strides=(2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='linear'))
        model.summary()

        model.compile(loss='mse', optimizer='adam')
        return model

    def image2state(self, image_data):
        # 1.缩放（可以不用，考虑到资源有限还是缩放比较好）
        resize_image_data = cv2.resize(image_data, (80, 80))
        # 2.转化成灰色图片
        gray_image_data = cv2.cvtColor(resize_image_data, cv2.COLOR_BGR2GRAY)
        # 3.转化成二值图片（供神经网络训练用）
        _, binary_image_data = cv2.threshold(gray_image_data, 1, 1, cv2.THRESH_BINARY)
        # 4.转成一个状态，可以直接用于训练
        state = binary_image_data.reshape((1, 80, 80, 1))
        return state

    def set_init_state(self, image_data):
        self.current_state = self.image2state(image_data)

    def get_action(self):
        action = np.zeros(2)
        if random.random() <= self.epsilon:
            # 采取真随机行为
            action_index = random.randrange(2)
            action[action_index] = 1
        else:
            # 采取经验行为
            result = self.model.predict(self.current_state)[0]
            action_index = np.argmax(result)
            action[action_index] = 1
        return action

    def set_perception(self, action, image_data, reward, terminal):
        next_state = self.image2state(image_data)
        self.memory.append((self.current_state, action, next_state, reward, terminal))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()
        if self.step > self.observation_step:
            self.train_model()
        self.current_state = next_state
        self.step += 1

    def train_model(self):
        # 降低 epsilon，从而降低随机性
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.epsilon_init - self.epsilon_min) / self.epsilon_decay_step
        # 可能 sample 不出 BATCH 个样本
        mini_batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        X = np.zeros((len(mini_batch), 80, 80, 1))
        y = np.zeros((len(mini_batch), 2))
        # replay experience
        weights = []
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
                Q = self.model.predict(next_memory_state)
                y[i, action_index] = reward + self.gamma * np.max(Q)
            weights.append(abs(y[i, action_index]))

        loss = self.model.train_on_batch(X, y, weights)

        if self.step % 100 == 0:
            print(self.step, loss)
            self.model.save(self.model_path)


def play_bird():
    # 1. 实例化 brain
    brain = Brain()

    # 2. 创建游戏
    game = Game()

    # 3. 玩游戏！
    # 3.1. 初始化状态
    action = np.array([1, 0])
    image_data, reward, terminal = game.action_and_reward(action)
    brain.set_init_state(image_data)

    # 3.2 开始玩
    while True:
        action = brain.get_action()
        image_data, reward, terminal = game.action_and_reward(action)
        brain.set_perception(action, image_data, reward, terminal)


if __name__ == '__main__':
    play_bird()
