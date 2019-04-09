import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, state_dims, max_size=10000, history_length=4):
        self.size = max_size
        self.index = 0
        self.actions = np.empty(self.size, dtype=np.uint8)
        self.rewards = np.empty(self.size, dtype=np.uint8)
        self.states = np.empty((state_dims[0], state_dims[1], self.size))  # screen observation
        self.terminals = np.empty(self.size, dtype=np.bool)
        self.dims = state_dims
        self.count = 0
        self.history_length = history_length

    def add(self, action, reward, states, terminal):
        assert states.shape == self.dims, "inconsistent state sizes(dims)"
        self.actions[self.index] = action
        self.terminals[self.index] = terminal
        self.rewards[self.index] = reward
        self.states[..., self.index] = np.array(states)
        self.index = (self.index + 1) % self.size
        self.count = max(self.count, self.index)
        if self.index == 0:
            print("Replay buffer is full.")

    def get_state(self, index):
        assert self.count > 0, "replay memory is empty"
        assert 0 <= index < self.count, "error:index out of buffer range"
        if index >= self.history_length - 1:
            # use faster slicing
            return self.states[..., (index - (self.history_length - 1)):(index + 1)]
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
            return self.states[..., indexes]

    def get_batch(self, batch_size):
        indexes = []
        while len(indexes) < batch_size:
            while True:
                ind = random.randint(self.history_length, self.count - 1)
                # if wraps over current pointer, then get new one
                if self.index <= ind < self.history_length + self.index:
                    continue
                if self.terminals[(ind - self.history_length):ind].any():
                    continue
                break
            indexes.append(ind)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]
        states = [self.get_state(ind) for ind in indexes]
        pre_states = [self.get_state((ind - 1) % self.count) for ind in indexes]

        return actions, rewards, terminals, states, pre_states
