import torch
import torch.nn as nn


class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
        )
        # V值层
        self.value = nn.Linear(512, 1)
        # A优势函数层
        self.advantage = nn.Linear(512, 2)

    def forward(self, data):
        data = self.conv_layer(data)
        data = data.reshape(data.size(0), -1)
        data = self.linear_layer(data)
        value = self.value(data)
        advantage = self.advantage(data)
        output = value + (advantage - advantage.mean())
        return output

    def add_histogram(self, writer, epoch):
        writer.add_histogram("weight", self.output.weight, epoch)


if __name__ == '__main__':
    input = torch.Tensor(2, 4, 84, 84)
    net = MyNet()
    output = net(input)
    params = sum([param.numel() for param in net.parameters()])
    print(params)
