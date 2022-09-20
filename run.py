# this is the main program of solving pre-defined scenario using mpdqn, pdqn and spdqn, contains of train progress
# and evaluation progress using classes in environment and agent files
# generate tensorboard graph automatically
# see run.py files from mpdqn paper and algorithm from DDPG paper for more infos
# by Yingjie Xu, 28.08.2022
import copy
import time

import click
import gym
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from common import ClickPythonLiteralOption
from common.modular_robot_domain import RobotFlattenedActionWrapper
from env.modular_robot.envs.environment import BasicEnv#noqa
from common.wrappers import ScaledParameterisedActionWrapper
from mpdqn.random_run import ran_run

writer = SummaryWriter('runs/trial')


def pad_action(act, act_param, env):
    """
    form a tuple (or dictionary according to env) of discrete action + action parameters for every module type
    """
    # params = [np.zeros((1,)), np.zeros((2,))]
    params = env.db.get_zero_param_list
    params[act][:] = act_param.astype('float64')
    # params = act_param.astype('float64')
    return (act, params)


def evaluate(env, agent, episodes=350, max_reward=100, ran_reward=0, point_plot=True):
    """
    Used for evaluation the agent performance (not used during the training process)
    State should be transformed to array here
    """
    returns = []
    timesteps = []
    for i in range(episodes):

        observation = env.reset()
        observation = np.concatenate((observation["robot"][0], observation["robot"][1].reshape(20, ),
                                      observation["base_point"], observation["target"]))
        state = observation.reshape(36, )
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent.act(state)
            action = pad_action(act, act_param, env=env)
            obs, reward, terminal, _ = env.step(action)
            obs = np.concatenate((obs["robot"][0], obs["robot"][1].reshape(20, ), obs["base_point"], obs["target"]))
            state = obs.reshape(36, )
            total_reward += reward
        timesteps.append(t)
        returns.append(total_reward)
        if point_plot:
            writer.add_scalars('Evaluating Reward', {'max reward': max_reward, 'episode reward': total_reward,
                                                     'random reward': ran_reward}, i+1)
    # return np.column_stack((returns, timesteps))
    return np.array(returns)


# click command to define some default hyperparameters and possibility to manually type in
@click.command()
@click.option('--seed', default=10, help='Random seed.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--episodes', default=50000, help='Number of epsiodes.', type=int)
@click.option('--batch-size', default=256, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.99, help='Discount factor.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=0, help='Number of transitions required to start learning.',
              type=int)
@click.option('--use-ornstein-noise', default=False,
              help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
@click.option('--replay-memory-size', default=100000, help='Replay memory size in transitions.', type=int)
@click.option('--epsilon-steps', default=47000,
              help='Number of episodes over which to linearly anneal epsilon.', type=int)
@click.option('--epsilon-final', default=0.01, help='Final epsilon value.', type=float)
@click.option('--tau-actor', default=0.01, help='Soft target network update averaging factor.', type=float)
@click.option('--tau-actor-param', default=0.001, help='Soft target network update averaging factor.', type=float)
@click.option('--learning-rate-actor', default=0.001, help="Actor network learning rate.", type=float)
@click.option('--learning-rate-actor-param', default=0.0001, help="Critic network learning rate.", type=float)  # 1e-5
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--initialise-params', default=True, help='Initialise action parameters.', type=bool)
@click.option('--clip-grad', default=10., help="Parameter gradient clipping limit.", type=float)
@click.option('--split', default=False, help='Separate action-parameter inputs.', type=bool)
@click.option('--multipass', default=True,
              help='Separate action-parameter inputs using multiple Q-network passes.', type=bool)
@click.option('--indexed', default=False, help='Indexed loss function.', type=bool)
@click.option('--weighted', default=False, help='Naive weighted loss function.', type=bool)
@click.option('--average', default=False, help='Average weighted loss function.', type=bool)
@click.option('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
@click.option('--zero-index-gradients', default=False,
              help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.",
              type=bool)
@click.option('--action-input-layer', default=0, help='Which layer to input action parameters.', type=int)
@click.option('--layers', default='[128,]', help='Duplicate action-parameter inputs.', cls=ClickPythonLiteralOption)
@click.option('--save-freq', default=0, help='How often to save models (0 = never).', type=int)
@click.option('--save-dir', default="results/platform", help='Output directory.', type=str)
@click.option('--render-freq', default=2000, help='How often to render / save frames of an episode.', type=int)
@click.option('--save-frames', default=False,
              help="Save render frames from the environment. Incompatible with visualise.", type=bool)
@click.option('--visualise', default=True, help="Render game states. Incompatible with save-frames.", type=bool)
@click.option('--title', default="PDDQN", help="Prefix of output files", type=str)
def run(seed, episodes, evaluation_episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold,
        replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor,
        learning_rate_actor_param, epsilon_final, zero_index_gradients, initialise_params, scale_actions,
        clip_grad, split, indexed, layers, multipass, weighted, average, random_weighted, render_freq,
        save_freq, save_dir, save_frames, visualise, action_input_layer, title):
    """
    main part of the training and evaluation process

    """
    #initialization of the environment

    # env = BasicEnv(db=db, mod_num=5)
    env = gym.make('Modular_robot-v0')
    # max_mod_num = env.max_mod_num # len(env.observation_space.spaces["robot"].spaces[0])
    # and env.observation_space.spaces["robot"].spaces[1].shape[0] = init 10
    # max_param_num = env.max_param_num  #env.observation_space.spaces["robot"].spaces[1].shape[1] = init 2
    # mod_num = env.mod_num  # env.action_space.spaces[0].n = init 5
    # step_count = []
    # step_count2 = []
    initial_params_ = (np.random.randn(env.db.get_param_num,) + 1).tolist()
    # initial_params_ = [0.,1.,2.]
    if scale_actions:
        for a in range(env.db.get_param_num):
            initial_params_[a] = 2. * (initial_params_[a] - env.action_space.spaces[1].low[0]) / (
                    env.action_space.spaces[1].high[0] - env.action_space.spaces[1].low[0]) - 1.

    # env = ScaledStateWrapper(env)
    env = RobotFlattenedActionWrapper(env)
    if scale_actions:
        env = ScaledParameterisedActionWrapper(env)
    low_bound = np.concatenate((np.zeros((10, )), env.observation_space.spaces["robot"].spaces[1].low.reshape(20,),
                                env.observation_space.spaces["base_point"].low.reshape(3, ),
                                env.observation_space.spaces["target"].low.reshape(3, )))
    high_bound = np.concatenate((np.ones((10, )), env.observation_space.spaces["robot"].spaces[1].high.reshape(20,),
                                 env.observation_space.spaces["base_point"].high.reshape(3, ),
                                 env.observation_space.spaces["target"].high.reshape(3, )))
    my_obs = gym.spaces.Box(low_bound, high_bound, (36,), dtype=np.float64)

    env.seed(seed)
    np.random.seed(seed)

    print("observation space is", env.observation_space)
    print("action space is", env.action_space)

    # define the agent, agent type chosen through parameters via click
    from agents.pdqn import PDQNAgent
    from agents.pdqn_split import SplitPDQNAgent
    from agents.pdqn_multipass import MultiPassPDQNAgent
    assert not (split and multipass)
    agent_class = PDQNAgent
    if split:
        agent_class = SplitPDQNAgent
    elif multipass:
        agent_class = MultiPassPDQNAgent

    agent = agent_class(
        observation_space=my_obs, action_space=env.action_space,
        batch_size=batch_size,
        learning_rate_actor=learning_rate_actor,
        learning_rate_actor_param=learning_rate_actor_param,
        epsilon_steps=epsilon_steps,
        gamma=gamma,
        tau_actor=tau_actor,
        tau_actor_param=tau_actor_param,
        clip_grad=clip_grad,
        indexed=indexed,
        weighted=weighted,
        average=average,
        random_weighted=random_weighted,
        initial_memory_threshold=initial_memory_threshold,
        use_ornstein_noise=use_ornstein_noise,
        replay_memory_size=replay_memory_size,
        epsilon_final=epsilon_final,
        inverting_gradients=inverting_gradients,
        actor_kwargs={'hidden_layers': layers,
                      'action_input_layer': action_input_layer, },
        actor_param_kwargs={'hidden_layers': layers,
                            'squashing_function': False,
                            'output_layer_init_std': 0.0001, },
        zero_index_gradients=zero_index_gradients,
        seed=seed)

    # only 3 actions right now
    if initialise_params:
        initial_weights = np.zeros((env.db.get_param_num, env.observation_space.spaces["robot"].spaces[1].shape[0]
                                    * env.observation_space.spaces["robot"].spaces[1].shape[1]
                                    + len(env.observation_space.spaces["robot"].spaces[0])
                                    + env.observation_space.spaces["base_point"].shape[0]
                                    + env.observation_space.spaces["target"].shape[0]))
        initial_bias = np.zeros(env.db.get_param_num)
        for a in range(env.db.get_param_num):
            initial_bias[a] = initial_params_[a]
        agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)
    print(agent)
    max_steps = 15
    total_reward = 0.
    max_reward = 350
    ran_reward = np.array(ran_run(episodes=50)).mean()
    returns = []
    start_time = time.time()
    k = 0
    # agent.epsilon_final = 0.
    # agent.noise = None

    #main loop
    for i in range(episodes):
        # if save_freq > 0 and save_dir and i % save_freq == 0:
        #     agent.save_models(os.path.join(save_dir, str(i)))
        # intermediate evaluating part
        # ------------------------------------------------
        if i % 30 == 0 and i != 0:
            print("Intermediate Evaluating over 10 episodes")
            agent_eva = copy.deepcopy(agent)
            agent_eva.epsilon_final = 0.
            agent_eva.epsilon = 0.
            agent_eva.noise = None
            evaluation_returns = evaluate(env, agent_eva, episodes=10, point_plot=False)
            inter_reward = evaluation_returns.mean()
            #tensorboard visualization
            writer.add_scalars('Training Reward',
                               {'intermediate evaluate': inter_reward}, i + 1)
        # ------------------------------------------------

        print("episode {}".format(i+1))
        observation = env.reset()
        print("observation is:{}".format(observation))
        observation = np.concatenate((observation["robot"][0], observation["robot"][1].reshape(20, ),
                                      observation["base_point"], observation["target"]))
        state = observation.reshape(36,)
        state = np.array(state, dtype=np.float32, copy=False)
        # if visualise and i % render_freq == 0:
        #     env.render()

        act, act_param, all_action_parameters = agent.act(state)
        action = pad_action(act, act_param, env=env)

        episode_reward = 0.
        agent.start_episode()
        for j in range(max_steps):
            print("step {}".format(j + 1))
            ret = env.step(action)
            obs, reward, terminal, _ = ret
            obs = np.concatenate((obs["robot"][0], obs["robot"][1].reshape(20, ), obs["base_point"], obs["target"]))
            next_state = obs.reshape(36,)
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            print(next_state)

            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
            next_action = pad_action(next_act, next_act_param, env=env)
            agent.step(state, (act, all_action_parameters), reward, next_state,
                       (next_act, next_all_action_parameters), terminal)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            state = next_state

            episode_reward += reward
            if terminal and j == 0:
                k += 1
                if k % 500 == 0:
                    env.render()
            if visualise and i % render_freq == 0:
                env.render()

            if terminal:
                writer.add_scalar('Step Count', j + 1, i + 1)
                break
        # ran_reward = 0
        writer.add_scalars('Training Reward', {'max reward': max_reward,
                                               'episode reward': episode_reward, 'random reward': ran_reward}, i+1)
        writer.add_scalar('Epsilon Curve', agent.epsilon, i+1)
        agent.end_episode()

        returns.append(episode_reward)
        total_reward += episode_reward
        if i % 100 == 0:
            print(
                '{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(i), total_reward / (i + 1), np.array(returns[-100:]).mean()))
    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    env.close()
    plt.clf()

    # returns = env.get_episode_rewards()
    plt.plot(returns, label='Episode Reward')
    plt.axhline(y=max_reward, color='green', label='Max Reward')
    plt.xlabel('episode')
    plt.ylabel('episode reward')
    plt.title('final training reward')
    plt.legend()
    plt.show()
    # plt.savefig('show.pdf')
    print("Ave. return =", sum(returns) / len(returns))
    print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)

    # np.save(os.path.join(dir, title + "{}".format(str(seed))), returns)

    if evaluation_episodes > 0:
        print("Evaluating agent over {} episodes".format(evaluation_episodes))
        agent.epsilon_final = 0.
        agent.epsilon = 0.
        agent.noise = None
        evaluation_returns = evaluate(env, agent, evaluation_episodes, max_reward=max_reward, ran_reward=ran_reward)
        plt.plot(evaluation_returns, label='Evaluate Reward')
        plt.axhline(y=max_reward, color='green', label='Max Reward')
        plt.xlabel('episode')
        plt.ylabel('evaluate reward')
        plt.title('final evaluate reward')
        plt.legend()
        plt.show()
        # plt.savefig('showandevaluate.pdf')
        print("Ave. evaluation return =", sum(evaluation_returns) / len(evaluation_returns))
        # np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_returns)


if __name__ == '__main__':
    run()
