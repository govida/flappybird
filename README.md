# Flappy Bird
基于强化学习的Flappy Bird，对[原始项目](https://github.com/yenchenlin/DeepLearningFlappyBird)进行重构，主要有以下几点：
### 1. Game
1. 对Game，进行重构，可以单独项目进行测试，并加入相应注释，增强可读性。
2. 添加额外 reward，拟加快收敛速率。

详见 [game.py](game.py)
### 2. Reinforce Learning
1. 抽取核心组件，凸显强化学习流程。
2. 修改网络结构，将4帧图像化为1帧，并丢弃最后几层的卷积与池化。
3. 利用keras重写。
4. 采用额外的网络或策略训练网络，拟加快收敛速率。
---
## CNN 
基本与原版网络实现一致。

详见 [CNN/model.py](CNN/model.py)
## CNN+ 
额外添加re-init机制，增加随机性，跳出目前的舒适区（局部最优解），加快全局的收敛速率。

详见 [CNN+/model.py](CNN+/model.py)
## RNN
利用RNN来训练网络

详见 [RNN/model.py](RNN/model.py)

## Result
Model| 100 epoch average score
-----|-----
CNN| 4.8
CNN+| 4.3
RNN| 0
截至 2019.1.10