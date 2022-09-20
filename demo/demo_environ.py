# This is to test with a small demo of reset and step func of env
# by Yizhen Li, 05,07,2022
from mpdqn.env.modular_robot.envs.environment import BasicEnv

b_env = BasicEnv()
obs_space = b_env.observation_space
action_space = b_env.action_space
print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))

obs = b_env.reset()
print("The initial observation (the first module) is {}".format(obs))

# ------------------------- settled action------------------------------------------
random_param = b_env.action_space.sample()
random_action = {"module": 0, "params": random_param}
print("The second module to be added is randomly chosen as {}".format(random_action))

# -------------------------- random action -------------------------------------------
b_env.action_space.spaces[0].n -= 1  # not choosing base module again
print("action space after not choosing base module again is {}".format(b_env.action_space))
random_action2 = b_env.action_space.sample()
print("random action to be taken is {}".format(random_action2))
