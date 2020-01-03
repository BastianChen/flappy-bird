import cv2
import numpy as np
import torch.nn as nn
import os
from game.Game import Game
from nets import MyNet
from config import args
import torch.multiprocessing as mp
from utils import *

'''A3C模型（CPU太小暂未实现）'''


class Trainer(mp.Process):
    def __init__(self, global_net, optimizer, global_ep, global_ep_r, res_queue, net_path, number):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.game = Game(level=2, train=True)
        self.net_path = net_path
        self.image_size = args.image_size
        self.epochs = args.epochs
        self.MAX_EP_STEP = args.MAX_EP_STEP
        self.start_epsilon = args.start_epsilon
        self.end_epsilon = args.end_epsilon
        self.memory_size = args.memory_size
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.observe = args.observe
        self.name = 'w%i' % number
        self.global_ep, self.global_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.global_net, self.optimizer = global_net, optimizer
        self.local_net = MyNet()
        # self.buffer_memory = deque(maxlen=self.memory_size)
        # self.writer = SummaryWriter()
        if os.path.exists(net_path):
            self.local_net.load_state_dict(torch.load(net_path))
        else:
            self.local_net.apply(self.weight_init)

    def weight_init(self, model):
        if isinstance(model, nn.Linear) or isinstance(model, nn.Conv2d):
            nn.init.normal_(model.weight, mean=0., std=0.1)
            nn.init.constant_(model.bias, 0)

    def edit_image(self, image, width, height):
        image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        return image[None, :, :].astype(np.float32)

    def run(self):
        image, reward, terminal = self.game.step(0)
        image = self.edit_image(image[:self.game.screen_width, :int(self.game.base_y)], self.image_size,
                                self.image_size)
        image = torch.from_numpy(image)
        state = torch.cat(tuple(image for _ in range(4))).unsqueeze(0)

        for i in range(self.epochs):
            buffer_state, buffer_action, buffer_reward = [], [], []
            epoch_r = 0.
            for t in range(self.MAX_EP_STEP):
                action = self.local_net.select_action(state)
                # reward 只有三种值-1，0.1，1
                next_image, reward, terminal = self.game.step(action)
                next_image = self.edit_image(next_image[:self.game.screen_width, :int(self.game.base_y)],
                                             self.image_size, self.image_size)
                next_image = torch.from_numpy(next_image)
                # 插入新的一张照片，组成新的四张照片组合
                next_state = torch.cat((state[0, 1:, :, :], next_image)).unsqueeze(0)
                if t == args.MAX_EP_STEP - 1:
                    terminal = True
                epoch_r += reward
                buffer_state.append(state[0])
                action = torch.tensor(action)
                buffer_action.append(action)
                buffer_reward.append(reward)

                if i % 5 == 0 or terminal:  # update global and assign to local net
                    # sync
                    push_and_pull(self.optimizer, self.local_net, self.global_net, terminal, next_state, buffer_state,
                                  buffer_action, buffer_reward, self.gamma, self.device)
                    buffer_state, buffer_action, buffer_reward = [], [], []

                    if terminal:  # done and print information
                        record(self.global_ep, self.global_ep_r, epoch_r, self.res_queue, self.name)
                        break
                state = next_state
            torch.save(self.global_net.state_dict(), self.net_path)
        self.res_queue.put(None)
        # # 初始的随机动作构建样本池
        # if i <= self.observe:
        #     action = np.random.choice([0, 1], 1, p=[0.9, 0.1])[0]
        # else:
        #     prediction = self.q_net(state)[0]
        #     # 更新探索值
        #     epsilon = self.end_epsilon + ((self.epochs - i) * (self.start_epsilon - self.end_epsilon) / self.epochs)
        #
        #     if random.random() <= epsilon:
        #         # 探索
        #         action = random.randint(0, 1)
        #         print("-------- random action -------- ", action)
        #     else:
        #         # 开发
        #         action = torch.argmax(prediction).item()
        # next_image, reward, terminal = self.game.step(action)
        # next_image = self.edit_image(next_image[:self.game.screen_width, :int(self.game.base_y)], self.image_size,
        #                              self.image_size)
        # next_image = torch.from_numpy(next_image).to(self.device)
        # # 插入新的一张照片，组成新的四张照片组合
        # next_state = torch.cat((state[0, 1:, :, :], next_image)).unsqueeze(0)
        # self.buffer_memory.append([state, action, reward, next_state, terminal])
        # state = next_state
        #
        # # 从样本池中取样本训练
        # if i > self.observe:
        #     batch = random.sample(self.buffer_memory, min(len(self.buffer_memory), self.batch_size))
        #     state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)
        #
        #     state_batch = torch.cat(state_batch).to(self.device)
        #     action_batch = torch.Tensor([[1, 0] if action == 0 else [0, 1] for action in action_batch]).to(
        #         self.device)
        #     reward_batch = torch.Tensor(reward_batch).unsqueeze(1).to(self.device)
        #     next_state_batch = torch.cat(next_state_batch).to(self.device)
        #
        #     if i % 40 == 0:
        #         self.target_net.load_state_dict(self.q_net.state_dict())
        #
        #     # DDQN使用当前网络先得到动作
        #     current_prediction_batch = self.q_net(state_batch)
        #     current_action_batch = torch.argmax(self.q_net(next_state_batch), dim=-1)
        #     # 使用target网络得到估计Q值
        #     next_prediction_batch = self.target_net(next_state_batch).gather(1, current_action_batch.unsqueeze(1))
        #
        #     y_batch = torch.cat(
        #         [reward if terminal else reward + self.gamma * next_prediction for reward, terminal, next_prediction
        #          in zip(reward_batch, terminal_batch, next_prediction_batch)])
        #     q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        #
        #     loss = self.loss(q_value, y_batch)
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     self.optimizer.step()
        #
        #     print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
        #         i + 2000, self.epochs, action, loss, epsilon, reward, torch.max(prediction)))
        #
        #     if (i - 1) % 1000 == 0:
        #         self.writer.add_scalar("1/loss", loss, i)
        #         self.writer.add_scalar("1/Q-value", torch.max(prediction), i)
        #         self.writer.add_scalar("1/epsilon", epsilon, i)
        #         self.writer.add_scalar("1/reward", reward, i)
        #         self.q_net.add_histogram(self.writer, i)
        #         torch.save(self.q_net.state_dict(), self.net_path)


if __name__ == '__main__':
    net_path = "models/net_40.pth"
    global_net = MyNet()
    if os.path.exists(net_path):
        global_net.load_state_dict(torch.load(net_path))
    optimizer = torch.optim.Adam(global_net.parameters())
    # 在共享内存中放入全局次数以及全局价值总和
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    # 构建多个进程同时训练
    workers = [Trainer(global_net, optimizer, global_ep, global_ep_r, res_queue, net_path, i) for i in
               range(mp.cpu_count())]
    [w.run() for w in workers]
    # trainer = Trainer("models/net_40.pth")
    # trainer.train()
