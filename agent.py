import tensorflow as tf
import numpy as np
import random
import cv2
import copy
import gc
import time

screen_width = 84
screen_height = 84


class Agent(object):
    def __init__(self, env, action_network, target_network, obs, exploration_schedule, batchsize=32):
        self.action_network = action_network
        self.target_network = target_network
        self.exploration_schedule = exploration_schedule
        self.env = env
        self.input_obs = obs
        self.min_reward = -1000
        self.max_reward = 1000
        self.discount_rate = 0.9
        self.target = tf.Variable(np.zeros((batchsize, env.action_space.n)))
        self.loss = tf.losses.mean_squared_error(self.action_network, self.target)
        self.train = tf.train.AdamOptimizer().minimize(self.loss)

    def exploration_rate(self, episodes):
        return self.exploration_schedule(episodes)

    def update_target_network(self):
        del self.target_network
        gc.collect()
        self.target_network = copy.copy(self.action_network)

    def update_action_network(self, session, minibatch, total_step):
        actions, rewards, terminals, poststates, prestates = minibatch
        # doing the actual training
        post_qvalues = session.run(self.target_network, feed_dict={self.input_obs: np.array(poststates)})
        # print(post_qvalues)
        max_post_qvalues = np.max(post_qvalues, axis=1)
        # print(max_post_qvalues)
        assert (max_post_qvalues.shape[0] == len(minibatch[0])), "{} {}".format(max_post_qvalues.shape, len(minibatch))
        rewards = np.clip(rewards, self.min_reward, self.max_reward)

        targets = np.zeros((len(minibatch[0]), self.env.action_space.n))
        for i, action in enumerate(actions):
            if terminals[i]:
                targets[i, action] = float(rewards[i])
            else:
                targets[i, action] = float(rewards[i]) + self.discount_rate * max_post_qvalues[i]
        self.target.assign(targets)
        session.run(self.train, feed_dict={self.input_obs: np.array(prestates)})

    def take_action(self, session, state, total_steps):

        qvalues = session.run(self.action_network, feed_dict={self.input_obs: state[np.newaxis,...]})

        if random.random() < self.exploration_rate(total_steps):
            # act randomly
            action = self.env.action_space.sample()
        else:
            action = np.argmax(qvalues)

        observation, reward, terminal, info = self.env.step(action)
        if terminal:
            self.env.reset()
        return self.screen(observation), reward, terminal, action

    def play(self, session, total_steps, history, max_step=1000):
        history.reset()
        obs = self.env.reset()
        terminal = False
        screen = self.screen(obs)
        step = 0
        while not terminal and step < max_step:
            screen, reward, terminal, info = self.take_action(session, history.get(), total_steps)
            history.add(screen)
            self.env.render()
            time.sleep(0.02)
            step += 1
        self.env.close()

    def screen(self, obs):
        return cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (screen_width, screen_height))

    def save_model(self, save_path):
        raise NotImplementedError

    def restart(self):
        obs = self.env.reset()
        return self.screen(obs)


class DistDQNAgent(Agent):

    def take_action(self, session, screen, total_steps):
        qvalues = session.run(self.action_network, feed_dict={self.input_obs: screen[np.newaxis, ..., np.newaxis]})

        if random.random() < self.exploration_rate(total_steps):
            # act randomly
            action = self.env.action_space.sample()
        else:
            action = np.argmax(qvalues)

        observation, reward, terminal, info = self.env.step(action)
        if terminal:
            self.env.reset()
        return self.screen(observation), reward, terminal, action

    def update_action_network(self, session, minibatch, total_step):
        return

    def save_model(self, save_path):
        raise NotImplementedError
