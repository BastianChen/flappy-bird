import torch
import cv2
import numpy as np
from game.Game import Game
from nets import MyNet


class Detector:
    def __init__(self, net_path):
        self.image_size = 84
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 如果没有GPU的话把在GPU上训练的参数放在CPU上运行，cpu-->gpu 1:lambda storage, loc: storage.cuda(1)
        self.map_location = None if torch.cuda.is_available() else lambda storage, loc: storage
        self.net = MyNet().to(self.device)
        self.net.load_state_dict(torch.load(net_path, map_location=self.map_location))
        self.net.eval()
        self.game_state = Game(level=2, train=False, sound="off")

    def edit_image(self, image, width, height):
        image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
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
                prediction = self.net(state)[0]
                action = torch.argmax(prediction).item()

                next_image, reward, terminal = self.game_state.step(action)
                next_image = self.edit_image(next_image[:self.game_state.screen_width, :int(self.game_state.base_y)],
                                             self.image_size, self.image_size)
                next_image = torch.from_numpy(next_image).to(self.device)
                next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

                state = next_state
                if reward == 1:
                    checkpoint += 1
                if terminal:
                    print(f"飞行到第{checkpoint}关")
                    checkpoint = 0
            except KeyboardInterrupt:
                print("Quit")


if __name__ == "__main__":
    detector = Detector("models/net_30.pth")
    detector.detect()
