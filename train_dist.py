import tensorflow as tf
import gym
from play import *
from create_model import *
from agent import *
from replay_buffer import *
import logging

logging.basicConfig(format='%(asctime)s %(message)s')

# Hyperparameters
env = gym.make('Breakout-v0')
num_epochs = 100
train_steps = 2500
epsilon = 0.02
batch_size = 10
train_repeat = 1
target_steps = 10000
action_steps = 4
# Networks and Buffer
obs = tf.placeholder(dtype=tf.float32, shape=(None, screen_width, screen_height, 1))
dqn_action_net = dist_dqn_model(obs, env.action_space.n, 50, 0)
dqn_target_net = dist_dqn_model(obs, env.action_space.n, 50, 1)

mem = ReplayBuffer((screen_width, screen_height), 10000)
agent = DistDQNAgent(env, dqn_action_net, dqn_target_net, obs, lambda x: max(0.1, 1 - 0.99 / 100000 * x))


def train():
    explore_steps = 0  # type: int
    #with tf.device("/device:CPU:0"):
    with tf.device("/device:CPU:0"):
        with tf.Session() as session:
            init_op = tf.global_variables_initializer()
            session.run(init_op)
            for epoch in range(num_epochs):
                total_reward = 0.0
                total_episodes = 1
                screen = agent.restart()
                for i in range(train_steps):
                    # perform game step
                    screen, reward, terminal, action = agent.take_action(session, screen, explore_steps)
                    mem.add(action, reward, screen, terminal)
                    total_reward += reward
                    if terminal:
                        total_episodes += 1
                    # Update target network every target_steps steps
                    if i % target_steps == 0:
                        agent.update_target_network()
                    # train after every train_frequency steps
                    if mem.count > batch_size and i % action_steps == 0:
                        # train for train_repeat times
                        for j in range(train_repeat):
                            agent.update_action_network(session, mem.get_batch(batch_size), explore_steps)
                    # increase number of training steps for epsilon decay
                    explore_steps += 1
                    if explore_steps%200 == 0:
                        print(explore_steps)
                agent.play(session, explore_steps)
                print("Average Reward:{} Total:{} Episodes:{}".format(total_reward / total_episodes,total_reward, total_episodes))


def test():
    raise NotImplementedError

if __name__ == '__main__':
    train()
    # test()
