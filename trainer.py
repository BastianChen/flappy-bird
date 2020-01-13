import cv2
import random
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from collections import deque
import os
from game.Game import Game
from nets import MyNet

'''Dueling DDQN实现'''


class Trainer:
    def __init__(self, net_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.game = Game(level=2, train=True)
        self.image_size = 84
        self.epochs = 4000000
        self.start_epsilon = 0.1
        self.end_epsilon = 0.0001
        self.memory_size = 15000
        self.batch_size = 32
        self.gamma = 0.99
        self.observe = 2000
        self.q_net = MyNet().to(self.device)
        self.target_net = MyNet().to(self.device)
        self.net_path = net_path
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), weight_decay=0.0005)
        self.buffer_memory = deque(maxlen=self.memory_size)
        self.writer = SummaryWriter()
        if os.path.exists(net_path):
            self.q_net.load_state_dict(torch.load(net_path))
            self.target_net.load_state_dict(torch.load(net_path))
        # else:
        #     self.q_net.apply(self.init_weight)
        #     self.target_net.load_state_dict(self.q_net.state_dict())

    def init_weight(self, model):
        if isinstance(model, nn.Linear) or isinstance(model, nn.Conv2d):
            nn.init.normal_(model.weight, mean=0., std=0.1)
            nn.init.constant_(model.bias, 0.)

    def edit_image(self, image, width, height):
        image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        return image[None, :, :].astype(np.float32)

    def train(self):
        image, reward, terminal = self.game.step(0)
        image = self.edit_image(image[:self.game.screen_width, :int(self.game.base_y)], self.image_size,
                                self.image_size)
        image = torch.from_numpy(image).to(self.device)
        state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

        for i in range(self.epochs):
            # 初始的随机动作构建样本池
            if i <= self.observe:
                action = np.random.choice([0, 1], 1, p=[0.9, 0.1])[0]
            else:
                prediction = self.q_net(state)[0]
                # 更新探索值
                epsilon = self.end_epsilon + ((self.epochs - i) * (self.start_epsilon - self.end_epsilon) / self.epochs)

                if random.random() <= epsilon:
                    # 探索
                    action = random.randint(0, 1)
                    print("-------- random action -------- ", action)
                else:
                    # 开发
                    action = torch.argmax(prediction).item()
            next_image, reward, terminal = self.game.step(action)
            next_image = self.edit_image(next_image[:self.game.screen_width, :int(self.game.base_y)], self.image_size,
                                         self.image_size)
            next_image = torch.from_numpy(next_image).to(self.device)
            # 插入新的一张照片，组成新的四张照片组合
            next_state = torch.cat((state[0, 1:, :, :], next_image)).unsqueeze(0)
            self.buffer_memory.append([state, action, reward, next_state, terminal])
            state = next_state

            # 从样本池中取样本训练
            if i > self.observe:
                batch = random.sample(self.buffer_memory, min(len(self.buffer_memory), self.batch_size))
                state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

                state_batch = torch.cat(state_batch).to(self.device)
                action_batch = torch.Tensor([[1, 0] if action == 0 else [0, 1] for action in action_batch]).to(
                    self.device)
                reward_batch = torch.Tensor(reward_batch).unsqueeze(1).to(self.device)
                next_state_batch = torch.cat(next_state_batch).to(self.device)

                if i % 30 == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

                # DDQN使用当前网络先得到动作
                current_prediction_batch = self.q_net(state_batch)
                current_action_batch = torch.argmax(self.q_net(next_state_batch), dim=-1)
                # 使用target网络得到估计Q值
                next_prediction_batch = self.target_net(next_state_batch).gather(1, current_action_batch.unsqueeze(1))

                y_batch = torch.cat(
                    [reward if terminal else reward + self.gamma * next_prediction for reward, terminal, next_prediction
                     in zip(reward_batch, terminal_batch, next_prediction_batch)])
                q_value = torch.sum(current_prediction_batch * action_batch, dim=1)

                loss = self.loss(q_value, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
                    i + 2000, self.epochs, action, loss, epsilon, reward, torch.max(prediction)))

                if (i - 1) % 1000 == 0:
                    self.writer.add_scalar("1/loss", loss, i)
                    self.writer.add_scalar("1/Q-value", torch.max(prediction), i)
                    self.writer.add_scalar("1/epsilon", epsilon, i)
                    self.writer.add_scalar("1/reward", reward, i)
                    self.q_net.add_histogram(self.writer, i)
                    torch.save(self.q_net.state_dict(), self.net_path)


if __name__ == '__main__':
    trainer = Trainer("models/net_30.pth")
    trainer.train()
