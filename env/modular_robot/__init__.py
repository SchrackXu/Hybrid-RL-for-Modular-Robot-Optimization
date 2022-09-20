from gym.envs.registration import register

register(
    id='Modular_robot-v0',
    entry_point='mpdqn.env.modular_robot.envs:BasicEnv',
    max_episode_steps=200,
    # TODO: max_episode_steps=200,
    # TODO: reward_threshold=1.0? maybe 0.8 or 0.9
)
