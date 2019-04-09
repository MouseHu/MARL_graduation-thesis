import numpy as np


class History:
    def __init__(self, history_length, screen_height, screen_width):
        self.history = np.zeros(
            [screen_height, screen_width, history_length], dtype=np.float32)

    def add(self, screen):
        self.history[..., :-1] = self.history[..., 1:]
        self.history[..., -1] = screen

    def reset(self):
        self.history *= 0

    def get(self):
        return self.history
