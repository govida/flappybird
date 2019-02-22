# Flappy Bird
基于强化学习的Flappy Bird，对[原始项目](https://github.com/yenchenlin/DeepLearningFlappyBird)进行重构。

## Idea
1. 定义 terminal state 和 不同 state 下采取不同 action 所带来的 reward。
2. 尽可能多地观测 state，并记录 current state，action，next state，reward，terminal 到 memory 中。
3. 训练 （state => action_reward）model。
4. 利用 model 驱动 state 变化，形成闭环，最后反馈给 model 自身。

## Code
### Game
1. 可作为单独项目进行测试。
2. 添加额外reward，拟加快收敛速率。

详见 [game.py](game.py)
### Model
1. 抽取核心组件，凸显强化学习流程。
2. 丢弃最后几层无用的卷积与池化，加入dropout。
3. 利用keras重写。

详见 [model.py](model.py)
- CNN: use_extract_reward=True，与原版网络实现基本一致。
- CNN+: use_extract_reward=False，添加额外 reward，加快全局的收敛速率。

## Result
