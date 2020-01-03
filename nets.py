import torch
import torch.nn as nn
import math


#
# class MyNet(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.conv2d_3 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, dilation=1, groups=1,
#                                   bias=True)
#         self.reLU_10 = nn.ReLU(inplace=False)
#         self.conv2d_4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, dilation=1, groups=1,
#                                   bias=True)
#         self.reLU_11 = nn.ReLU(inplace=False)
#         self.conv2d_5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=1, groups=1,
#                                   bias=True)
#         self.reLU_8 = nn.ReLU(inplace=False)
#         self.linear_7 = nn.Linear(in_features=7 * 7 * 64, out_features=512, bias=True)
#         self.reLU_9 = nn.ReLU(inplace=False)
#         self.linear_6 = nn.Linear(in_features=512, out_features=2, bias=True)
#
#     def forward(self, x_para_1):
#         x_conv2d_3 = self.conv2d_3(x_para_1)
#         x_reLU_10 = self.reLU_10(x_conv2d_3)
#         x_conv2d_4 = self.conv2d_4(x_reLU_10)
#         x_reLU_11 = self.reLU_11(x_conv2d_4)
#         x_conv2d_5 = self.conv2d_5(x_reLU_11)
#         x_reLU_8 = self.reLU_8(x_conv2d_5)
#         x_reshape_12 = torch.reshape(x_reLU_8, shape=(-1, 7 * 7 * 64))
#         x_linear_7 = self.linear_7(x_reshape_12)
#         x_reLU_9 = self.reLU_9(x_linear_7)
#         x_linear_6 = self.linear_6(x_reLU_9)
#         return x_linear_6


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.PReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.PReLU(),
            nn.Conv2d(64, 128, 3, 1),
            nn.PReLU()
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(7 * 7 * 128, 512),
            nn.ReLU()
        )
        self.action_layer = nn.Linear(512, 2)
        # self.mu = nn.Linear(512, 2)
        # self.sigma = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(-1)
        self.value_layer = nn.Linear(512, 1)
        self.distribution = torch.distributions.Categorical
        self.mseloss = nn.MSELoss()

    def forward(self, data):
        data = self.conv_layer(data)
        data = data.reshape(data.size(0), -1)
        data = self.linear_layer(data)
        action = self.softmax(self.action_layer(data))
        # mu = 2 * self.tanh(self.mu(linear_layer))
        # sigma = self.softplus(self.sigma(linear_layer)) + 0.001  # avoid 0
        value = self.value_layer(data)
        return action, value

    def select_action(self, state):
        value, _ = self.forward(state)
        m = self.distribution(value)
        return m.sample().cpu().detach().numpy()

    def get_loss(self, state, action, v_t):
        prob, values = self.forward(state)
        td = v_t - values
        value_loss = td.pow(2)

        m = self.distribution(prob)
        log_prob = m.log_prob(action)

        # entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration;m.scale = sigma;m.loc = mu
        # exp_v = log_prob * td.detach() + 0.005 * entropy
        exp_v = log_prob * td.detach()
        action_loss = -exp_v
        total_loss = (action_loss + value_loss).mean()
        return total_loss

    def add_histogram(self, writer, epoch):
        writer.add_histogram('weight/action', self.action_layer.weight, epoch)
        writer.add_histogram('weight/value', self.value_layer.weight, epoch)


if __name__ == '__main__':
    input = torch.Tensor(4, 4, 84, 84)
    net = MyNet()
    net.select_action(input)
    params = sum([param.numel() for param in net.parameters()])
    print(params)
