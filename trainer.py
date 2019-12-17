import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
from game.Game import Game
from nets import MyNet


# common functions

def pre_processing(image, width, height):
    image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    return image[None, :, :].astype(np.float32)


def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.01, 0.01)
            nn.init.constant_(m.bias, 0)


def main(net_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    game = Game(level=2, train=True)

    image_size = 84

    epochs = 4000000

    start_epsilon = 0.1

    end_epsilon = 0.0001

    # memory_size = 50000
    memory_size = 20000

    batch_size = 32

    gamma = 0.99

    observe = 2000.

    q_net = MyNet().to(device)
    target_net = MyNet().to(device)

    if os.path.exists(net_path):
        q_net.load("models/flappy_bird.pth").to(device)
    # init_weight(net)

    criterion = nn.MSELoss(reduce=None, size_average=None)

    optimizer = optim.Adam(q_net.parameters(), weight_decay=0, amsgrad=False, lr=1e-6, betas=(0.9, 0.999), eps=1e-08)

    image, reward, terminal = game.step(0)

    image = pre_processing(image[:game.screen_width, :int(game.base_y)], image_size, image_size)

    image = torch.from_numpy(image).to(device)

    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    replay_memory = deque(maxlen=memory_size)

    for i in range(epochs):
        # 初始的随机动作构建样本池
        if i <= observe:
            # action = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
            action = np.random.choice([0, 1], 1, p=[0.9, 0.1])[0]
        else:
            prediction = q_net(state)[0]
            # Exploration or exploitation
            # 更新探索值
            epsilon = end_epsilon + ((epochs - i) * (start_epsilon - end_epsilon) / epochs)

            if random.random() <= epsilon:
                # 探索
                action = random.randint(0, 1)
                print("-------- random action -------- ", action)
            else:
                # 开发
                action = torch.argmax(prediction).item()

        next_image, reward, terminal = game.step(action)
        next_image = pre_processing(next_image[:game.screen_width, :int(game.base_y)], image_size, image_size)
        next_image = torch.from_numpy(next_image).to(device)
        # 插入新的一张照片，组成新的四张照片组合
        next_state = torch.cat((state[0, 1:, :, :], next_image)).unsqueeze(0)

        replay_memory.append([state, action, reward, next_state, terminal])

        # 更新state
        state = next_state

        # 从样本池中取样本训练
        if i > observe:
            batch = random.sample(replay_memory, min(len(replay_memory), batch_size))
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

            state_batch = torch.cat(state_batch).to(device)
            action_batch = torch.Tensor([[1, 0] if action == 0 else [0, 1] for action in action_batch]).to(device)
            reward_batch = torch.Tensor(reward_batch).unsqueeze(1).to(device)
            next_state_batch = torch.cat(next_state_batch).to(device)

            if i % 40 == 0:
                target_net.load(q_net.state_dict()).to(device)
            # 大于2060时,开始采用两个网络
            if i > 2060:
                current_prediction_batch = q_net(state_batch)
                next_prediction_batch = target_net(next_state_batch)
            else:
                current_prediction_batch = q_net(state_batch)
                next_prediction_batch = q_net(next_state_batch)

            y_batch = torch.cat(
                [reward if terminal else reward + gamma * torch.max(next_prediction) for
                 reward, terminal, next_prediction in
                 zip(reward_batch, terminal_batch, next_prediction_batch)])
            q_value = torch.sum(current_prediction_batch * action_batch, dim=1)

            loss = criterion(q_value, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # state = next_state
            print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
                i + 2000, epochs, action, loss, epsilon, reward, torch.max(prediction)))
            # if (i + 1) % 10000 == 0:
            #     torch.save(net, "models/flappy_bird_{}.pth".format(i + 90000))
            if (i - 1) % 1000 == 0:
                torch.save(q_net, "models/flappy_bird.pth")


if __name__ == "__main__":
    main("models/net.pt")
