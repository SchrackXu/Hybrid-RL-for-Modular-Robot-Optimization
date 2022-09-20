# This is the training process using  a random agent
# by Yingjie Xu, 28.08.2022
import gym
import numpy as np
from env.modular_robot.envs.environment import BasicEnv#noqa
from common.wrappers import ScaledParameterisedActionWrapper
from mpdqn.common.modular_robot_domain import RobotFlattenedActionWrapper


# ------------------------ test plot random action result----------------------------------------------
# @click.command()
# @click.option('--episodes', default=20, help='Number of epsiodes.', type=int)
# @click.option('--visualise', default=True, help="Render game states. Incompatible with save-frames.", type=bool)
# @click.option('--render-freq', default=10, help='How often to render / save frames of an episode.', type=int)
# @click.option('--max-steps', default=10, help='the maximize steps to take in one episode.', type=int)


def pad_action(act, act_param):
    """
    form a tuple (or dictionary according to env) of discrete action + action parameters for every module type
    """
    #params = env.db.get_zero_param_list
    params = [np.zeros((1,)), np.zeros((2,)), np.zeros((2,)), np.zeros((2,)), np.zeros((2,))]
    params[act][:] = act_param.astype('float64')
    # params = act_param.astype('float64')
    return (act, params)


def ran_run(episodes=1, render_freq=10, visualise=False, max_steps=10):
    """
    This is to plot the result of random choosing action, comparing with RL agent.
    return total_reward: a list of total reward for each episode
    input see description in click
    """
    total_reward = []
    # env = BasicEnv(db=db, mod_num=len(db)-1)  # not possible to choose module with last module id, aka the base module
    env = gym.make('Modular_robot-v0')
    env = RobotFlattenedActionWrapper(env)
    env = ScaledParameterisedActionWrapper(env)
    for i in range(episodes):
        episode_reward = 0.
        # print("episode {}".format(i + 1))
        env.reset()
        action_type = env.action_space.sample()[0]
        action_param = env.action_space.sample()[action_type+1]
        action = pad_action(action_type, action_param)
        for j in range(max_steps):
            obs, reward, terminal, _ = env.step(action)
            action_type = env.action_space.sample()[0]
            action_param = env.action_space.sample()[action_type + 1]
            action = pad_action(action_type, action_param)
            episode_reward += reward
            if visualise and i != 0 and i % render_freq == 0:
                env.render()
            if terminal:
                break

        total_reward.append(episode_reward)

    env.close()
    # plt.plot(returns, label='Episode Reward')
    # plt.xlabel('episode')
    # plt.ylabel('episode reward')
    # plt.title('reward history')
    # plt.legend()
    # plt.show()
    return total_reward


if __name__ == '__main__':
    ran_run()
