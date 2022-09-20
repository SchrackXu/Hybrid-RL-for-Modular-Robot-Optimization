# This is a warapper to flatten the action space of modular robot env
# by Yingjie Xu, 31.08.2022
import gym
import numpy as np


class RobotFlattenedActionWrapper(gym.ActionWrapper):
    """
    Changes the format of the parameterised action space to conform to that of modular robot
    """
    def __init__(self, env):
        super(RobotFlattenedActionWrapper, self).__init__(env)
        old_as = env.action_space
        num_actions = old_as.spaces[0].n
        self.action_space = gym.spaces.Tuple((
            old_as.spaces[0],  # actions
            gym.spaces.Box(old_as.spaces[1].low[0], old_as.spaces[1].high[0], (1,), dtype=np.float64),
            # gym.spaces.Box(old_as.spaces[1].low, old_as.spaces[1].high, dtype=np.float32)
            *(gym.spaces.Box(old_as.spaces[1].low, old_as.spaces[1].high,
                             dtype=np.float64) for i in range(0, num_actions - 1))
        ))

    def action(self, action):
        return action
