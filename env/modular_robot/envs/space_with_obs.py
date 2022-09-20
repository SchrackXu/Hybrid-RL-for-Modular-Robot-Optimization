# this is to generate a working scenario with obstacles for the assembled robot
# by Yingjie Xu & Yizhen Li, 29,08,2022
from Geometry import Box, ComposedGeometry, Sphere
from scenario import Goals, Obstacle
from scenario.Scenario import Scenario, ScenarioHeader
from scenario.ScenarioPlanner import SimpleHierarchicalPlanner
from utilities import spatial


class TestObsScenario:

    @classmethod
    def generate_obs_scene(cls, head, tags=['test_env'], author='Group 4'):
        header = ScenarioHeader(head, tags=tags, author=author)
        sphere = Sphere(dict(r=0.3), spatial.homogeneous([-0.5, 0.5, 1.2]))
        # sphere2 = Sphere(dict(r=1), spatial.homogeneous([0.5, -0.5, 0]))
        box = Box(dict(x=.3, y=.4, z=.5), spatial.homogeneous([.7, .6, 0.5]))
        collision = ComposedGeometry([sphere, box])
        obstacle = Obstacle.Obstacle(head, collision=collision, name='test_obs')
        return obstacle, Scenario(header, obstacles=[obstacle])

    @classmethod
    def generate_scenario(cls, target_location, target_rotation=None):
        # set target as up-straight if there's no restriction of orientation of end effector
        if target_rotation is None:
            reach_goal = Goals.Reach('Reach goal', spatial.homogeneous(target_location))
        else:
            reach_goal = Goals.Reach('Reach goal', spatial.homogeneous(target_location, target_rotation))
        box = Box(dict(x=.1, y=.1, z=.1), spatial.homogeneous([.2, .4, .2]))
        # box = Box('Box 2', .1, .1, .1, np.array([.2, .4, .2]))
        scenario_header = ScenarioHeader('test_case', 'py', ['debug', 'test'])
        scenario = Scenario(header=scenario_header, obstacles=[box], goals=[reach_goal])
        planner = SimpleHierarchicalPlanner(scenario=scenario)
        return reach_goal, scenario, planner
