# Flappy Bird
基于强化学习的Flappy Bird，对[原始项目](https://github.com/yenchenlin/DeepLearningFlappyBird)进行重构：
### 1. Game
1. 可作为单独项目进行测试。
2. 添加额外 reward，拟加快收敛速率。

详见 [game.py](game.py)
### 2. Model
1. 抽取核心组件，凸显强化学习流程。
2. 丢弃最后几层无用的卷积与池化，加入dropout。
3. 利用keras重写。

详见 [model.py](model.py)
- CNN：use_extract_reward=True，与原版网络实现基本一致。
- CNN+：use_extract_reward=False，添加额外 reward，加快全局的收敛速率。

## Result
Model| 100 epoch average score
-----|-----
CNN|
CNN+|