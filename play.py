import tensorflow as tf
import gym
import numpy as np
import random
import cv2
#env = gym.make('CartPole-v0')
screen_width = 128
screen_height = 128


def play(input_x, action_function, env, num_episode, epsilon, buffer=None, max_actions=10000):
    with tf.Session() as sess:
        for i in range(num_episode):
            observation = env.reset()
            screen = cv2.resize(cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY), (screen_width, screen_height))
            terminal = False
            counter = 0
            while not terminal and counter < max_actions:

                qvalues = sess.run(action_function, feed_dict={input_x: screen[np.newaxis, ...]})
                action = env.action_space.sample() if random.uniform(0, 1) < epsilon else np.argmax(
                    qvalues)

                observation, reward, terminal, info = env.step(action)
                screen = cv2.resize(cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY), (screen_width, screen_height))

                if buffer:
                    buffer.add(action, reward, screen, terminal)
                else:
                    env.render()
                counter = counter + 1

            env.close()



