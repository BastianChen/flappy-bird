import torch
import torch.nn as nn


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
        self.softmax = nn.LogSoftmax(-1)
        self.value_layer = nn.Linear(512, 1)
        self.distribution = torch.distributions.Categorical
        self.mseloss = nn.MSELoss()

    def forward(self, data):
        data = self.conv_layer(data)
        data = data.reshape(data.size(0), -1)
        data = self.linear_layer(data)
        action = self.softmax(self.action_layer(data))
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
