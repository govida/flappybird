from game.flappy_bird import GameState
from memory import Memory
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import load_model
import cv2
import numpy as np
import random
import os
import pickle

def image2state(image_data):
    # 1.缩放（可以不用，考虑到资源有限还是缩放比较好）
    resize_image_data = cv2.resize(image_data, (80, 80))
    # 2.转化成灰色图片
    gray_image_data = cv2.cvtColor(resize_image_data, cv2.COLOR_BGR2GRAY)
    # 3.转化成二值图片（供神经网络训练用）
    _, binary_image_data = cv2.threshold(gray_image_data, 1, 255, cv2.THRESH_BINARY)
    # 4.转成一个样本
    state = binary_image_data.reshape((1, 80, 80, 1))
    return state


def build_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=4, strides=(4, 4), input_shape=(80, 80, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=2, strides=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.summary()

    model.compile(loss='mse', optimizer='adam')
    return model


INIT_EPSILON = 0.2  # epsilon 代表采取随机行为的阈值
MIN_EPSILON = 0.0001  # epsilon 的 MIN
EXPLORE = 30000.  # epsilon 衰减的总步数
OBSERVATION = 1000.  # 观察多少步后，开始降低随机性
REPLAY_MEMORY = 5000  # 记忆容量
GAMMA = 0.99  # 未来奖励 折扣系数
BATCH = 64
MODEL_PATH = '../model/dqn.model'
ACTIONS = 2
PRINT_INTERVAL = 10
SAVE_INTERVAL = 1000

def read_human_exp(memory):
    with open('../assets/exp.pkl', 'rb') as f:
        exp = pickle.load(f)
    for e in exp:
        memory.add(e[3], e)


def train():
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        model = build_model()

    game = GameState()
    memory = Memory()
    read_human_exp(memory)

    action = np.zeros(ACTIONS)
    action[0] = 1

    image_data, reward, terminal = game.frame_step(action)
    state = image2state(image_data)

    epsilon = INIT_EPSILON
    t = 0  # 当前迭代步数

    while True:
        action = np.zeros(ACTIONS)
        if random.random() <= epsilon or t < OBSERVATION:
            if t < OBSERVATION:
                # 模拟一个玩的还行的人
                if random.random() < 0.2:
                    action_index = 1
                else:
                    action_index = 0
            else:
                # 采取真随机行为
                action_index = random.randrange(ACTIONS)
            action[action_index] = 1
        else:
            # 采取经验行为
            result = model.predict(state)[0]
            action_index = np.argmax(result)
            action[np.argmax(result)] = 1

        image_data, reward, terminal = game.frame_step(action)
        next_state = image2state(image_data)

        memory.add(reward, (state, action_index, next_state, reward, terminal))

        state = next_state  # 下一状态变为当前状态

        if memory.size() > REPLAY_MEMORY:
            memory.pop(REPLAY_MEMORY * 0.1)

        loss = 0
        # 先观察一段时间
        if t > OBSERVATION:
            # 降低 epsilon，从而降低随机性
            if epsilon > MIN_EPSILON:
                epsilon -= (INIT_EPSILON - MIN_EPSILON) / EXPLORE

            # 可能 sample 不出 BATCH 个样本
            mini_batch = memory.sample(BATCH)

            X = np.zeros((len(mini_batch), 80, 80, 1))
            y = np.zeros((len(mini_batch), ACTIONS))
            # replay experience
            for i in range(0, len(mini_batch)):
                memory_state = mini_batch[i][0]
                action_index = mini_batch[i][1]
                next_memory_state = mini_batch[i][2]
                reward = mini_batch[i][3]
                terminal = mini_batch[i][4]
                X[i] = memory_state
                if terminal:
                    y[i, action_index] = reward
                else:
                    Q = model.predict(next_memory_state)
                    y[i, action_index] = reward + GAMMA * np.max(Q)

            loss = model.train_on_batch(X, y)
            if t % SAVE_INTERVAL == 0:
                model.save(MODEL_PATH)
        t = t + 1  # 总帧数+1

        if t % PRINT_INTERVAL == 0:
            print('step:', t, 'epsilon:', epsilon, 'loss:', loss)


if __name__ == '__main__':
    train()
