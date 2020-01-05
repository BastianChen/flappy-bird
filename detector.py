import argparse
import torch
import cv2
import numpy as np
# torch.nn.Module.dump_patches=True
from game.Game import Game

parser = argparse.ArgumentParser()
parser.add_argument("--level", type=int, default=2, help="The game level (0: easy; 1: normal; 2: difficult")
parser.add_argument("--sound", type=str, default="on", help="The game sound (off on)")


# args = parser.parse_args()

class Detector:
    def __init__(self, net_path):
        self.image_size = 84
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 如果没有GPU的话把在GPU上训练的参数放在CPU上运行，cpu-->gpu 1:lambda storage, loc: storage.cuda(1)
        self.map_location = None if torch.cuda.is_available() else lambda storage, loc: storage
        self.model_path = net_path
        self.model = torch.load(self.model_path, map_location=self.map_location)
        self.model.eval().to(self.device)
        self.game_state = Game(level=2, train=False, sound="off")

    def edit_image(self, image, width, height):
        image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
        # cv2.imshow("image", image)
        _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        # cv2.imshow("image1", image)
        return image[None, :, :].astype(np.float32)

    def detect(self):
        image, reward, terminal = self.game_state.step(0)
        image = self.edit_image(image[:self.game_state.screen_width, :int(self.game_state.base_y)], self.image_size,
                                self.image_size)
        image = torch.from_numpy(image).to(self.device)
        state = torch.cat([image for _ in range(4)]).unsqueeze(0)
        checkpoint = 0
        while True:
            try:
                prediction = self.model(state)[0]
                action = torch.argmax(prediction).item()

                next_image, reward, terminal = self.game_state.step(action)
                next_image = self.edit_image(next_image[:self.game_state.screen_width, :int(self.game_state.base_y)],
                                             self.image_size,
                                             self.image_size)
                next_image = torch.from_numpy(next_image).to(self.device)
                next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

                state = next_state
                if reward == 1:
                    checkpoint += 1
                if terminal:
                    print(f"小鸟飞到第{checkpoint}关")
                    checkpoint = 0
            except KeyboardInterrupt:
                print("Quit")


#
# def pre_processing(image, width, height):
#     image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
#     # cv2.imshow("image", image)
#     _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
#     # cv2.imshow("image1", image)
#     return image[None, :, :].astype(np.float32)
#
#
# def test():
#     image_size = 84
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     # 如果没有GPU的话把在GPU上训练的参数放在CPU上运行，cpu-->gpu 1:lambda storage, loc: storage.cuda(1)
#     map_location = None if torch.cuda.is_available() else lambda storage, loc: storage
#
#     model_path = "models/flappy_bird.pth"
#
#     model = torch.load(model_path, map_location=map_location)
#
#     model.eval().to(device)
#
#     game_state = Game(level=2, train=False, sound="off")
#
#     image, reward, terminal = game_state.step(0)
#     image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], image_size, image_size)
#     image = torch.from_numpy(image).to(device)
#     state = torch.cat([image for _ in range(4)]).unsqueeze(0)
#
#     checkpoint = 0
#     while True:
#         try:
#             prediction = model(state)[0]
#             action = torch.argmax(prediction).item()
#
#             next_image, reward, terminal = game_state.step(action)
#             next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], image_size,
#                                         image_size)
#             next_image = torch.from_numpy(next_image).to(device)
#             next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
#
#             state = next_state
#             if reward == 1:
#                 checkpoint += 1
#             if terminal:
#                 print(f"小鸟飞到第{checkpoint}关")
#                 checkpoint = 0
#         except KeyboardInterrupt:
#             print("Quit")


if __name__ == "__main__":
    # test()
    # "models/flappy_bird.pth"
    detector = Detector("models/flappy_bird.pth")
    detector.detect()
