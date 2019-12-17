import pygame
import random

from game import utils
from itertools import cycle

########  PARAMS  ########

SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
PIPE_GAP_SIZE = 100

BASE_Y = SCREEN_HEIGHT * 0.8
VEL_X = 4
LEVEL = 2

########   INIT   ########

pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

IMAGES, SOUNDS, HITMASKS = utils.load()

PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()

clock = pygame.time.Clock()
pygame.display.set_caption('Flappy Bird')


########   MAIN   ########

class Base(pygame.sprite.Sprite):

    def __init__(self):
        super(Base, self).__init__()
        self.screen = pygame.display.get_surface()

        self.x = 0
        self.y = BASE_Y
        self.image = IMAGES["base"]

        self.shift = self.image.get_width() - SCREEN_WIDTH

    def update(self):
        self.x = -((-self.x + 100) % self.shift)
        self.screen.blit(self.image, (self.x, self.y))


class Bird(pygame.sprite.Sprite):

    def __init__(self):
        super(Bird, self).__init__()
        self.screen = pygame.display.get_surface()

        self.x = int(SCREEN_WIDTH * 0.2)
        self.y = int((SCREEN_HEIGHT - PLAYER_HEIGHT) / 2)
        self.image = IMAGES["player"]

        self.vel_y = 0
        self.max_vel_y = 10
        self.min_vel_y = -8
        self.acc_y = 1
        self.flap_acc = -9
        self.flapped = False

        self.player_index_gen = cycle([0, 1, 2, 1])
        self.player_index = 0

    def update(self, loopIter):
        self.move()
        if (loopIter + 1) % 3 == 0:
            self.player_index = next(self.player_index_gen)
        self.screen.blit(self.image[self.player_index], (self.x, self.y))

    def move(self):
        if self.vel_y < self.max_vel_y and not self.flapped:
            self.vel_y += self.acc_y
        if self.flapped:
            self.flapped = False
        self.y += min(self.vel_y, BASE_Y - self.y - PLAYER_HEIGHT)
        if self.y < 0:
            self.y = 0

    def flap(self):
        self.vel_y = self.flap_acc
        self.flapped = True

    @property
    def w(self):
        return self.image[0].get_width()

    @property
    def h(self):
        return self.image[0].get_height()


class Pipe(pygame.sprite.Sprite):

    def __init__(self, type=0, x=None, y=None, gap_y=0):  # 0 upper 1 lower
        super(Pipe, self).__init__()

        self.screen = pygame.display.get_surface()
        self.type = type
        self.image = IMAGES["pipe"][type]

        self.vel_x = -VEL_X

        self.gap_y = gap_y

        self.init_pos(x, y)

    def update(self):
        self.x += self.vel_x
        self.screen.blit(self.image, (self.x, self.y))

    def init_pos(self, x, y):
        gapY = self.gap_y

        gapY += int(BASE_Y * 0.2)
        pipeX = SCREEN_WIDTH + 10

        pipe_gap_size = [200, 150, 100][LEVEL]

        if self.type == 1:
            self.x = pipeX
            self.y = gapY + pipe_gap_size
        # + random.randint(0, 50)
        else:
            self.x = pipeX
            self.y = gapY - PIPE_HEIGHT

        self.x = x or self.x
        self.y = y or self.y


class Pipes(pygame.sprite.Group):

    def __init__(self, type=0):
        super(Pipes, self).__init__()
        self.type = type

    def update(self, gap_y=None):
        if 0 < self.sprites()[0].x < VEL_X + 1:
            self.add(Pipe(type=self.type, gap_y=gap_y))
        if self.sprites()[0].x < -PIPE_WIDTH:
            self.remove(self.sprites()[0])
        super().update()


class Score(pygame.sprite.Sprite):

    def __init__(self, score=0):
        super(Score, self).__init__()
        self.screen = pygame.display.get_surface()

        self.score = score
        self.status = 0

    def update(self):
        score_digits = [int(x) for x in list(str(self.score))]
        total_width = 0  # total width of all numbers to be printed

        for digit in score_digits:
            total_width += IMAGES['numbers'][digit].get_width()

        x_offset = (SCREEN_WIDTH - total_width) / 2

        for digit in score_digits:
            self.screen.blit(IMAGES['numbers'][digit], (x_offset, SCREEN_HEIGHT * 0.1))
            x_offset += IMAGES['numbers'][digit].get_width()

    def add(self, number):
        self.score += number


class Game():

    def __init__(self, level=2, train=False, human_play=False, sound="off"):
        global LEVEL
        LEVEL = level
        self.base = Base()
        self.bird = Bird()
        self.init_pipes()
        self.loopIter = 0
        self.score = Score()
        self.status = 1

        self.sound = 0 if sound == "off" else 1  # 0 off 1 on

        self.screen_width = SCREEN_WIDTH
        self.base_y = BASE_Y
        self.human_play = human_play
        self.fps = 300 if train else 30

    def reset(self):
        self.base = Base()
        self.bird = Bird()
        self.score = Score()
        self.init_pipes()
        self.loopIter = 0
        self.status = 0  # 0 wait 1 start

    def init_pipes(self):
        self.upper_pipes = Pipes(type=0)
        self.lower_pipes = Pipes(type=1)
        gap_y_1 = self.get_random_pipe_gap_y()
        gap_y_2 = self.get_random_pipe_gap_y()
        self.upper_pipes.add(Pipe(type=0, gap_y=gap_y_1, x=SCREEN_WIDTH, ))
        self.upper_pipes.add(Pipe(type=0, gap_y=gap_y_2, x=SCREEN_WIDTH + (SCREEN_WIDTH / 2), ))
        self.lower_pipes.add(Pipe(type=1, gap_y=gap_y_1, x=SCREEN_WIDTH, ))
        self.lower_pipes.add(Pipe(type=1, gap_y=gap_y_2, x=SCREEN_WIDTH + (SCREEN_WIDTH / 2), ))

    def step(self, action=None):

        self.loopIter = (self.loopIter + 1) % self.fps

        # get action
        if action is not None:
            pygame.event.pump()
            action = action
        else:
            action = 0  # 0: nothing  1: flappy
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if self.status == 0:
                            self.status = 1
                        action = 1
                    if event.key == pygame.K_RETURN:
                        self.status = 1 - self.status

        if action == 1:
            self.bird.flap()
            self.play_sound("wing")

        # check if crash here and calc reward
        if self.check_crash():
            self.play_sound("hit", "die")
            terminal = True
            self.reset()
            reward = -1
            if not self.human_play:
                self.play()
        else:
            reward = self.calc_reward()
            terminal = False

        # draw bg

        if self.status == 1:
            screen.fill([0, 0, 0])

            # screen.blit(IMAGES['background'], (0, 0))

            # draw bird
            self.bird.update(self.loopIter)

            # draw pipe
            gap_y = self.get_random_pipe_gap_y()
            self.upper_pipes.update(gap_y)
            self.lower_pipes.update(gap_y)

            # draw base
            self.base.update()

            # draw score
            self.score.update()

            pygame.display.update()
        clock.tick(self.fps)

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        return image_data, reward, terminal

    def pause(self):
        self.status = 0

    def play(self):
        self.status = 1

    def play_sound(self, *args):
        if self.sound == 0:
            return
        for arg in args:
            SOUNDS[arg].play()

    def get_random_pipe_gap_y(self):
        gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
        index = random.randint(0, len(gapYs) - 1)
        return gapYs[index]

    def get_mid_pos(self, obj):
        return obj.x + PLAYER_WIDTH / 2

    def calc_reward(self):
        player_mid_pos = self.get_mid_pos(self.bird)
        for pipe in self.upper_pipes:
            pipe_mid_pos = self.get_mid_pos(pipe)
            if pipe_mid_pos <= player_mid_pos < pipe_mid_pos + VEL_X:
                self.score.add(1)
                self.play_sound("point")
                return 1
        return 0.1

    def check_crash(self):
        player = self.bird

        # if player crashes into ground
        if player.y + player.h >= BASE_Y - 1:
            return True
        else:

            playerRect = pygame.Rect(player.x, player.y, player.w, player.h)

            for uPipe, lPipe in zip(self.upper_pipes, self.lower_pipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe.x, uPipe.y, PIPE_WIDTH, PIPE_HEIGHT)
                lPipeRect = pygame.Rect(lPipe.x, lPipe.y, PIPE_WIDTH, PIPE_HEIGHT)

                # player and upper/lower pipe hitmasks
                pHitMask = HITMASKS['player'][player.player_index]
                uHitmask = HITMASKS['pipe'][0]
                lHitmask = HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    return True

        return False


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                return True
    return False
