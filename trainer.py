import cv2
import numpy as np
import torch.nn as nn
import os
from game.Game import Game
from nets import Actor, Critic
from config import args
from utils import *
from collections import deque
import random
from tensorboardX import SummaryWriter

'''DDPG模型(适用于无限动作的情况，本项目无法实现)'''


class Trainer:
    def __init__(self, actor_net_path, critic_net_path):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.game = Game(level=2, train=True)
        self.actor_net_path = actor_net_path
        self.critic_net_path = critic_net_path
        self.image_size = args.image_size
        self.epochs = args.epochs
        self.start_epsilon = args.start_epsilon
        self.end_epsilon = args.end_epsilon
        self.memory_size = args.memory_size
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.observe = args.observe
        self.tau = args.tau
        self.actor, self.critic = Actor().to(self.device), Critic().to(self.device)
        self.actor_target, self.critic_target = Actor().to(self.device), Critic().to(self.device)
        self.buffer_memory = deque(maxlen=self.memory_size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.MSELoss = nn.MSELoss()
        self.writer = SummaryWriter()
        if os.path.exists(actor_net_path):
            self.actor.load_state_dict(torch.load(actor_net_path))
            self.actor_target.load_state_dict(torch.load(actor_net_path))
            self.critic.load_state_dict(torch.load(critic_net_path))
            self.critic_target.load_state_dict(torch.load(critic_net_path))
        else:
            self.actor.apply(self.weight_init)
            self.actor_target.apply(self.weight_init)
            self.critic.apply(self.weight_init)
            self.critic_target.apply(self.weight_init)

    def weight_init(self, model):
        if isinstance(model, nn.Linear) or isinstance(model, nn.Conv2d):
            nn.init.normal_(model.weight, mean=0., std=0.1)
            nn.init.constant_(model.bias, 0)

    def edit_image(self, image, width, height):
        image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        return image[None, :, :].astype(np.float32)

    def train(self):
        image, reward, terminal = self.game.step(0)
        image = self.edit_image(image[:self.game.screen_width, :int(self.game.base_y)], self.image_size,
                                self.image_size)
        image = torch.from_numpy(image)
        state = torch.cat(tuple(image for _ in range(4))).unsqueeze(0).to(self.device)

        for i in range(self.epochs):
            # 更新探索值
            epsilon = self.end_epsilon + ((self.epochs - i) * (self.start_epsilon - self.end_epsilon) / self.epochs)
            # 初始的随机动作构建样本池
            if i <= self.observe:
                action = np.random.choice([0, 1], 1, p=[0.9, 0.1])[0]
            else:
                if random.random() <= epsilon:
                    # 探索
                    action = random.randint(0, 1)
                    # print("-------- random action -------- ", action)
                else:
                    # 开发
                    action = self.actor.select_action(state).item()
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
                action_batch = torch.Tensor(action_batch).unsqueeze(1).to(self.device)
                reward_batch = torch.Tensor(reward_batch).unsqueeze(1).to(self.device)
                next_state_batch = torch.cat(next_state_batch).to(self.device)
                terminal_batch = torch.Tensor(terminal_batch).unsqueeze(1).to(self.device)

                # 计算目标网络的估计Q值
                action_target = torch.Tensor(self.actor_target.select_action(next_state_batch)).to(
                    self.device)
                target_Q = self.critic_target(next_state_batch, action_target)
                # 计算实际Q值
                target_Q = reward_batch + ((1 - terminal_batch) * args.gamma * target_Q)

                # 计算估计Q值
                current_Q = self.critic(state_batch, action_batch)
                critic_loss = self.MSELoss(current_Q, target_Q)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                current_action = torch.Tensor(self.actor.select_action(state_batch)).to(self.device)
                actor_loss = -self.critic(state_batch, current_action).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # 更新目标网络中的actor以及critic网络的参数
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data = self.tau * param.data + (1 - self.tau) * target_param.data
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data = self.tau * param.data + (1 - self.tau) * target_param.data

                self.writer.add_scalar("loss/actor", actor_loss, i)
                self.writer.add_scalar("loss/critic", critic_loss, i)
                self.writer.add_scalar("1/epsilon", epsilon, i)
                self.writer.add_scalar("1/reward", reward, i)
                self.actor.add_histogram(self.writer, i)
                self.critic.add_histogram(self.writer, i)

                if (i - 1) % 1000 == 0:
                    torch.save(self.actor.state_dict(), self.actor_net_path)
                    torch.save(self.critic.state_dict(), self.critic_net_path)

                # # DDQN使用当前网络先得到动作
                # current_prediction_batch = self.q_net(state_batch)
                # current_action_batch = torch.argmax(self.q_net(next_state_batch), dim=-1)
                # # 使用target网络得到估计Q值
                # next_prediction_batch = self.target_net(next_state_batch).gather(1, current_action_batch.unsqueeze(1))
                #
                # y_batch = torch.cat(
                #     [reward if terminal else reward + self.gamma * next_prediction for reward, terminal, next_prediction
                #      in zip(reward_batch, terminal_batch, next_prediction_batch)])
                # q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
                #
                # loss = self.loss(q_value, y_batch)
                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()
                #
                # print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
                #     i + 2000, self.epochs, action, loss, epsilon, reward, torch.max(prediction)))
                #
                # if (i - 1) % 1000 == 0:
                #     self.writer.add_scalar("1/loss", loss, i)
                #     self.writer.add_scalar("1/Q-value", torch.max(prediction), i)
                #     self.writer.add_scalar("1/epsilon", epsilon, i)
                #     self.writer.add_scalar("1/reward", reward, i)
                #     self.q_net.add_histogram(self.writer, i)
                #     torch.save(self.q_net.state_dict(), self.net_path)


if __name__ == '__main__':
    trainer = Trainer("models/actor.pth", "models/critic.pth")
    trainer.train()
