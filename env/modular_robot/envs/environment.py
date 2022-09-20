# This is the file to build the environment of implementing mpdqn algo
# As the very first implementation, connection btw module and self-conflict are ignored
# by Yizhen Li, 15,08,2022
import gym
import numpy as np
from gym import spaces
from numpy.linalg import norm

from Robot import PinRobot
from mpdqn.env.modular_robot.envs.robot_assemble import ParamModAss, RobAss
from mpdqn.env.modular_robot.envs.space_with_obs import TestObsScenario
from mpdqn.parametrized_module.parametrized_mod_gen import db


class BasicEnv(gym.Env):
    """
    This is the first step env
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, size: int = 1.5, max_mod_num: int = 10, database=db):
        """
        initialize the environment
        size: space size of the 3-dim space the robots is in, init 2*2*2
        mod_num: the number of modules in database, init 2
        max_param_num: max. parameters number for a single module
        max_mod_num: max. module you can add in an episode
        mod_params_space: the shape of all modules' params, mod_num*max_param_num
        (auto 0-appended should be done before using the env), init see below
        """
        self.assembly = ParamModAss  # add rob & ass to self if we want 3d visualization of the robot and assembly
        self.robot = PinRobot
        self.max_mod_num = max_mod_num
        max_param_num = db.max_param_num
        self.max_param_num = max_param_num
        mod_num = len(database.all_module_ids) - 1
        self.mod_num = mod_num
        self.module_sequence = []
        self.module_sequence_int = []
        self._base_location = None
        self._target_location = None
        self._EE_location = None
        self.db = database
        self.obstacle, self.scene = TestObsScenario.generate_obs_scene(head='Two Sphere obstacles')
        x = np.zeros(max_mod_num) + mod_num  # num of generated module type in action space,
        # each module in sequence has mod_num type
        self.size = size  # The size of the 3d space, aka the biggest range of robot size
        self.observation_space = spaces.Dict(
            {
                "base_point": spaces.Box(0, size, shape=(3,), dtype=np.float32),
                # base_point's (x,y,z) in 3d space
                "robot": spaces.Tuple(
                    (spaces.MultiDiscrete(x), spaces.Box(0.5, size, shape=(max_mod_num, max_param_num),
                                                         dtype=np.float_))),
                # every existing module's choice and params in sequence (modules assembled till the present step)
                "target": spaces.Box(0, size, shape=(3,), dtype=np.float32),  # target's (x,y,z) in 3d space
            }
        )
        self.action_space = spaces.Tuple((spaces.Discrete(mod_num),
                                          spaces.Box(0.5, size, shape=(max_param_num,), dtype=np.float_)))

    def reset(self):
        """
        The reset method will be called to initiate a new episode.
        You may assume that the step method will not be called before reset has been called.
        reset should be called whenever a done signal has been issued.
        """
        # Choose the robot's base location uniformly at random
        # self._base_location = np.random.random_integers(0, 1, size=(3,))
        self._base_location = np.array([0, 0, 0])
        # We will sample the target's location randomly until it does not coincide with the robot's base location
        # self._target_location = np.array([1, 0, 1])
        self._target_location = np.array([np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(1, 2)])
        # self._target_location = self._base_location
        # while np.array_equal(self._target_location, self._base_location):
        #     self._target_location = np.random.random_integers(0, 1, size=(3,))
        # initial only one module now: from obs_space, output E.g. [(1,(30,20))]
        # ATTENTION: dim of param is all set as max_param_num
        # ---------------------------random sample-------------------------------------------------------
        # self.module_sequence_int.append(
        #     (str(self.observation_space["robot"]["module"].sample()),  # suppose moduleID is "0" "1"
        #      self.observation_space["robot"]["params"].sample())
        # )
        # -----------------------------------------------------------------------------------------------
        # need to sample module with base connector in reset to guarantee that there will always be one & the only one
        # base module. Parameters of base module is set as 0: the base module's param has no influence in robot config
        self.module_sequence = []
        self.module_sequence_int = []
        self.module_sequence.append((str(self.mod_num), np.array([0.0, 0.0])))
        self.module_sequence_int.append((self.mod_num, np.array([0.0, 0.0])))
        # output: dict with "base_point": (3,), "robot": [int,(2,)], "target": (3,)
        observation = self._get_obs()
        return observation

    def step(self, action: dict):
        """
        The step method usually contains most of the logic of your environment. It accepts an action
        computes the state of the environment after applying that action and returns the 4-tuple
        (observation, reward, done, info). Once the new state of the environment has been computed,
        we can check whether it is a terminal state, and we set done accordingly.
        Since we are using sparse binary rewards in GridWorldEnv, computing reward is trivial once we know done
        INPUT: action: tuple with [0]="module": discrete,
        [1]="params" with params[module]=params of the chosen module by agent
        return: observation: all attached module type and config
        reward: reward of this step
        done: return true if reached max_step or if target location can be reached with current config
        info: location of end effector after ik
        """
        mod_add = action[0]  # transfer modID form int in action_space to string to use func RobAss.solve
        param_add = action[1][mod_add]
        param_add = np.clip(param_add, np.ones(self.mod_num)[mod_add]*0.5, (np.ones(self.mod_num) * self.size)[mod_add])
        if mod_add == 0:
            param_add = param_add[0]
        self.module_sequence.append((str(mod_add), param_add))
        self.module_sequence_int.append((mod_add, param_add))
        # ------ use below to replace the solve function to test env (if solve has problems)------------
        # solves = [False]
        # q = 0
        # cost = 0
        rob_ass = RobAss(db=self.db)
        q, solves, assembly, robot = rob_ass.solve(target_location=self._target_location,
                                                   base_location=self._base_location,
                                                   observation=self.module_sequence, method="ik_scipy")
        # -------------------------------------------------------------------------------------
        done = (all(solves) or assembly.nModules == self.max_mod_num)
        # only return true if all elements of solves are true, also return true/or when get to max_mod_num
        # reward = 1 if all(solves) else 0 # sparse reward
        self._EE_location = robot.fk(q)
        self.collides = robot.has_collisions(self.scene)
        print(self.collides)
        observation = self._get_obs()
        info = self._EE_location
        self.robot = robot
        self.assembly = assembly
        reward = self._get_reward()
        # print("reward:{}\ninfo(end effector position):\n{}".format(reward, info))
        return observation, reward, done, info

    def render(self, mode="human"):
        """
        visualize result
        """
        vis = self.robot.plot(coordinate_systems='joints')
        self.obstacle.visualize(vis)
        # self.assembly.plot_graph(which="verbose")

    def _get_obs(self):
        """
        translates the environment’s state into an observation
        return: dictionary with "base_point" as base point of this episode, "target_point" as target of this episode
        "robot" as array of (num of max module to be attached in one robot * 1,
        num of max module to be attached in one robot * num of max parameters each module)
        first line as already chosen module type, 2nd line as params of each
        """
        mod_ids = []
        mod_params = []
        i = 0
        for mod_inst in self.module_sequence_int:
            # if mod_inst[0] == 0:
            #     mod_inst[1]
            if i == 0:
                mod_ids = np.array([mod_inst[0]])
                mod_params = mod_inst[1].reshape(1, 2)
            else:
                if mod_inst[0] == 0:
                    mod_para = np.concatenate((np.array([mod_inst[1]]), np.zeros(1, )))
                else:
                    mod_para = mod_inst[1]
                mod_params = np.concatenate((mod_params, mod_para.reshape(1, 2)), axis=0)
                mod_ids = np.append(mod_ids, mod_inst[0])

            i += 1
        pad_mod_params = np.zeros((self.max_mod_num, self.max_param_num))  # places where no input is filled as 0
        # to ensure the input size (observation aka output of env) to agent is always settled
        pad_mod_params[: mod_params.shape[0], :mod_params.shape[1]] = mod_params
        pad_mod_ids = np.zeros((self.max_mod_num,))
        pad_mod_ids[: mod_ids.shape[0]] = mod_ids/self.mod_num
        self.state = (pad_mod_ids, pad_mod_params)
        return {"base_point": self._base_location, "robot": self.state, "target": self._target_location}

    def _get_reward(self):
        """
        provide the reward, need tuning to pass to the situation
        """
        # dist = -norm(self._EE_location[:3, 3] - self._target_location)
        if (norm(self._EE_location[:3, 3] - self._target_location, ord=2)) <= 1e-3:
            reward = (50 - (norm(self._EE_location[:3, 3] - self._target_location, ord=2)) / (
                norm(self._base_location - self._target_location, ord=2)))
            reward = reward + 300/(2 ** (self.assembly.nModules - 2))
            # if self.assembly.nModules == 2:
            #     reward = reward + 200
            # elif self.assembly.nModules == 3:
            #     reward = reward +100
        else:
            reward = - (norm(self._EE_location[:3, 3] - self._target_location, ord=2)) / (
                norm(self._base_location - self._target_location, ord=2))
        reward = reward - (self.assembly.nModules - 2) * 2          # multi-config reward
        reward = reward - self.assembly.get_mass * 5
        if self.collides:
            print(self.robot.collisions(self.scene))
            reward = reward - 30

        # m_id = tuple(mod.id[0] for mod in assembly.module_instances)        # check same module in conti
        # for i in range(len(m_id)):
        #     if m_id[i] == m_id[i - 1]:
        #         reward += 0.1
        return reward
