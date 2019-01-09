from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, TimeDistributed, GRU
from keras.models import load_model, Model
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
        self.observation_step = 5000  # 观察多少步后，开始降低随机性
        self.memory_size = 10000  # 记忆容量
        self.gamma = 0.99  # 未来奖励 折扣系数
        self.memory = deque()
        self.batch_size = 64
        self.model_path = '../resources/model/rnn.model'
        self.step = 0  # 训练到第几步
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        else:
            self.model = self.build_model()

    def build_model(self):
        frames_input = Input(shape=(3, 80, 80, 1))

        vision_model = Sequential()
        vision_model.add(Conv2D(32, (4, 4), strides=(4, 4), input_shape=(80, 80, 1), activation='relu'))
        vision_model.add(MaxPooling2D(pool_size=(2, 2)))
        vision_model.add(Conv2D(64, (2, 2), strides=(2, 2), activation='relu'))
        vision_model.add(MaxPooling2D(pool_size=(2, 2)))
        vision_model.add(Flatten())

        frame_feature = GRU(32, dropout=0.2)(TimeDistributed(vision_model)(frames_input))
        predict_reward = Dense(2, activation='linear')(frame_feature)

        model = Model(frames_input, predict_reward)
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        return model

    def image2state(self, image_data):
        # 1.缩放（可以不用，考虑到资源有限还是缩放比较好）
        resize_image_data = cv2.resize(image_data, (80, 80))
        # 2.转化成灰色图片
        gray_image_data = cv2.cvtColor(resize_image_data, cv2.COLOR_BGR2GRAY)
        # 3.转化成二值图片（供神经网络训练用）
        _, binary_image_data = cv2.threshold(gray_image_data, 1, 1, cv2.THRESH_BINARY)
        # 4.转成一个状态，需要关联前面的state才能用于训练
        state = binary_image_data.reshape((80, 80, 1))
        return state

    # 移动滑窗，利用新状态 state 更新 states
    def slide_frame(self, states, state):
        states[0] = states[1]
        states[1] = states[2]
        states[2] = state
        return states

    def states2sample(self, states):
        return states.reshape(1, 3, 80, 80, 1)

    def set_init_state(self, image_data):
        self.current_states = np.zeros((3, 80, 80, 1))
        state = self.image2state(image_data)
        self.current_states = self.slide_frame(self.current_states, state)

    def get_action(self):
        action = np.zeros(2)
        if random.random() <= self.epsilon:
            # 采取真随机行为
            action_index = random.randrange(2)
            action[action_index] = 1
        else:
            # 采取经验行为
            result = self.model.predict(self.states2sample(self.current_states))[0]
            action_index = np.argmax(result)
            action[action_index] = 1
        return action

    def set_perception(self, action, image_data, reward, terminal):
        next_state = self.image2state(image_data)
        next_states = self.slide_frame(self.current_states, next_state)
        self.memory.append((self.current_states, action, next_states, reward, terminal))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()
        if self.step > self.observation_step:
            self.train_model()
        self.current_states = next_states
        self.step += 1

    def train_model(self):
        # 降低 epsilon，从而降低随机性
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.epsilon_init - self.epsilon_min) / self.epsilon_decay_step
        # 可能 sample 不出 BATCH 个样本
        mini_batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        X = np.zeros((len(mini_batch), 3, 80, 80, 1))
        y = np.zeros((len(mini_batch), 2))
        # replay experience
        for i in range(0, len(mini_batch)):
            memory_states = mini_batch[i][0]
            action_index = np.argmax(mini_batch[i][1])
            next_memory_states = mini_batch[i][2]
            reward = mini_batch[i][3]
            terminal = mini_batch[i][4]
            X[i] = memory_states
            if terminal:
                y[i, action_index] = reward
            else:
                Q = self.model.predict(self.states2sample(next_memory_states))
                y[i, action_index] = reward + self.gamma * np.max(Q)
        loss = self.model.train_on_batch(X, y)

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
