import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', default=0.99, type=float, help="折扣率")
parser.add_argument('--MAX_EP', default=30000, type=int, help='局部最大轮次')
parser.add_argument('--image_size', default=84, type=int, help='输入网络的图片大小')
parser.add_argument('--epochs', default=4000000, type=int, help='一共训练多少步')
parser.add_argument('--start_epsilon', default=0.8, type=float, help='初始探索值')
parser.add_argument('--end_epsilon', default=0.1, type=float, help='最小探索值')
parser.add_argument('--memory_size', default=20000, type=int, help='样本池大小')
parser.add_argument('--batch_size', default=128, type=int, help='训练批次')
parser.add_argument('--observe', default=20, type=int, help='随机采样次数')
parser.add_argument('--tau', default=0.005, type=float, help="用于target网络更新的平滑系数")
args = parser.parse_args()
