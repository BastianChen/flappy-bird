import torch
import torch.nn as nn


class ConvolutionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(output_channels),
            nn.PReLU()
        )

    def forward(self, data):
        return self.layer(data)


class ResidualLayer(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv = nn.Sequential(
            ConvolutionLayer(input_channels, input_channels // 2, 1, 1, 0, bias=False),
            ConvolutionLayer(input_channels // 2, input_channels // 2, 3, 1, 1, bias=False),
            ConvolutionLayer(input_channels // 2, input_channels, 1, 1, 0, bias=False)
        )

    def forward(self, data):
        return data + self.conv(data)


class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        # 输入图片大小为[n,4,84,84]
        self.conv_layer = nn.Sequential(
            ConvolutionLayer(4, 16, 3),  # n,32,82,82
            nn.MaxPool2d(3, 2),  # n,32,40,40
            ResidualLayer(16),
            ResidualLayer(16),
            ResidualLayer(16),
            ResidualLayer(16),
            ConvolutionLayer(16, 32, 3),  # n,64,38,38
            nn.MaxPool2d(3, 2),  # n,32,18,18
            ResidualLayer(32),
            ResidualLayer(32),
            ResidualLayer(32),
            ResidualLayer(32),
            ConvolutionLayer(32, 64, 3, 2, 1),  # n,128,9,9
            ConvolutionLayer(64, 128, 3, 2, 1),  # n,256,5,5
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(5 * 5 * 128, 512),
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
        writer.add_histogram("weight/value", self.value.weight, epoch)
        writer.add_histogram("weight/advantage", self.advantage.weight, epoch)


if __name__ == '__main__':
    input = torch.Tensor(2, 4, 84, 84)
    net = MyNet()
    output = net(input)
    params = sum([param.numel() for param in net.parameters()])
    print(params)
