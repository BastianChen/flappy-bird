import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


# class MyNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv_layer = nn.Sequential(
#             nn.Conv2d(4, 32, 8, 4),
#             nn.PReLU(),
#             nn.Conv2d(32, 64, 4, 2),
#             nn.PReLU(),
#             nn.Conv2d(64, 128, 3, 1),
#             nn.PReLU()
#         )
#         self.linear_layer = nn.Sequential(
#             nn.Linear(7 * 7 * 128, 512),
#             nn.ReLU()
#         )
#         self.action_layer = nn.Linear(512, 2)
#         # self.mu = nn.Linear(512, 2)
#         # self.sigma = nn.Linear(512, 2)
#         self.softmax = nn.LogSoftmax(-1)
#         self.value_layer = nn.Linear(512, 1)
#         self.distribution = torch.distributions.Categorical
#         self.mseloss = nn.MSELoss()
#
#     def forward(self, data):
#         data = self.conv_layer(data)
#         data = data.reshape(data.size(0), -1)
#         data = self.linear_layer(data)
#         action = self.softmax(self.action_layer(data))
#         # mu = 2 * self.tanh(self.mu(linear_layer))
#         # sigma = self.softplus(self.sigma(linear_layer)) + 0.001  # avoid 0
#         value = self.value_layer(data)
#         return action, value
#
#     def select_action(self, state):
#         value, _ = self.forward(state)
#         m = self.distribution(value)
#         return m.sample().cpu().detach().numpy()
#
#     def get_loss(self, state, action, v_t):
#         prob, values = self.forward(state)
#         td = v_t - values
#         value_loss = td.pow(2)
#
#         m = self.distribution(prob)
#         log_prob = m.log_prob(action)
#
#         # entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration;m.scale = sigma;m.loc = mu
#         # exp_v = log_prob * td.detach() + 0.005 * entropy
#         exp_v = log_prob * td.detach()
#         action_loss = -exp_v
#         total_loss = (action_loss + value_loss).mean()
#         return total_loss
#
#     def add_histogram(self, writer, epoch):
#         writer.add_histogram('weight/action', self.action_layer.weight, epoch)
#         writer.add_histogram('weight/value', self.value_layer.weight, epoch)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1),
            nn.ReLU()
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(5 * 5 * 64, 512),
            nn.ReLU()
        )
        self.action_layer = nn.Linear(512, 1)
        # self.relu6 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.distribution = Categorical

    def select_action(self, state):
        action = self.forward(state)
        action = action.cpu().detach().numpy()
        action = np.where(action[:, 0:] > 0.5, 1, 0)
        # # # print(action)
        # # m = self.distribution(action)
        # # # print(m.sample())
        # # return m.sample().cpu().detach().numpy()
        # if action.item() > 0.5:
        #     action = np.array([1])
        # else:
        #     action = np.array([0])
        return action

    def forward(self, state):
        data = self.conv_layer(state)
        data = data.reshape(data.size(0), -1)
        data = self.linear_layer(data)
        data = self.action_layer(data)
        action = self.sigmoid(data)
        return action

    def add_histogram(self, writer, epoch):
        writer.add_histogram("weight/actor", self.action_layer.weight, epoch)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.PReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.PReLU(),
            nn.Conv2d(64, 128, 3, 1),
            nn.PReLU(),
            nn.Conv2d(128, 64, 3, 1),
            nn.PReLU()
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(5 * 5 * 64, 512),
            nn.PReLU()
        )
        self.state_layer = nn.Linear(512, 127)
        self.cat_layer = nn.Sequential(
            nn.Linear(128, 256),
            nn.PReLU()
        )
        self.value_layer = nn.Linear(256, 1)

    def forward(self, state, action):
        state = self.conv_layer(state)
        state = state.reshape(state.size(0), -1)
        state = self.linear_layer(state)
        state = self.state_layer(state)
        data = torch.cat((state, action), dim=-1)
        data = self.cat_layer(data)
        value = self.value_layer(data)
        return value

    def add_histogram(self, writer, epoch):
        writer.add_histogram("weight/critic", self.value_layer.weight, epoch)


if __name__ == '__main__':
    # state = torch.Tensor(1, 4, 84, 84)
    # # net = MyNet()
    # # net.select_action(input)
    # # params = sum([param.numel() for param in net.parameters()])
    # # print(params)
    # actor = Actor()
    # action = actor.select_action(state).reshape(-1, 1)
    # print(action.item())
    # # print(action.shape)
    # critic = Critic()
    # action = torch.Tensor(action)
    # value = critic(state, action)
    # print(value)
    # # print(value.shape)
    # params_actor = sum([param.numel() for param in actor.parameters()])
    # params_critic = sum([param.numel() for param in critic.parameters()])
    # print(params_actor)
    # print(params_critic)
    # print(params_actor + params_critic)

    m = Categorical(torch.tensor([1, 4.6968e-26]))
    a = m.sample()  # equal probability of 0, 1, 2, 3
    print(m)
    print(a)
