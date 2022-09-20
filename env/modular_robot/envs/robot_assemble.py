# This file assembles a robot from given modules and calculate if the end effector reached goal or not
# by Yizhen Li, 29,08,2022
from datetime import datetime
from typing import Collection, Dict, List, Set, Tuple

import utilities.errors as err
from Bodies import Connector
from Module import ModuleAssembly
from mpdqn.parametrized_module.Parametrized_module import ParametrizedModDB, ParametrizedModule
from scenario import Goals
from scenario.CostFunctions import CostFunctionBase
from scenario.Scenario import Scenario
from scenario.ScenarioPlanner import ScenarioPlannerBase
from scenario.Solution import SolutionHeader, SolutionTrajectory
from scenario.Tolerance import CartesianXYZ
from utilities import spatial, tolerated_pose


class RobAss:
    """
    This is the class to assemble a robot from given modules and calculate if the end effector reached goal or not

    """

    def __init__(self, db):
        """
        initialize the environment
        db: module database
        module_list: This list defines which modules are within the assembly and
        also defines their index position when referencing single modules.
        param_list: list of parameters coming from 2nd part of observation,
        each set of parameters refers to corresponding instances in module list
        """
        self.db = db
        self.module_list = []
        self.param_list = []

    def get_robot(self, observation: list = None, base_location=None):
        """
        build robot out of given observation list
        base_location: base point, tuple/list/array 3X1
        assembly: robot assembly of class ParamModAss with input module type & params (obs)
        robot: robot made out of assembly
        """
        self.module_list = []
        self.param_list = []
        if observation is not None:
            for i in range(len(observation)):
                self.module_list.append(observation[i][0])  # module list with only module ID
                self.param_list.append(observation[i][1])  # param list with only params [(),()]
        else:
            raise ValueError("observations are not given")

        assembly = ParamModAss(self.db, self.module_list, obs=self.param_list)
        base_pose = spatial.homogeneous(base_location)
        robot = assembly.to_pin_robot(base_pose)
        return robot, assembly

    def solve(self, target_location, target_rotation=None, base_location=None, method="ik_scipy",
              observation: list = None):
        """
        set given goals as reach_goal (only one now for first dev), build robot base on given observations
        evaluate if the robot can reach the given target location (only one now for first dev)
        INPUT: target_location: translation matrix, tuple/list/array 3X1
        base_location: base point, tuple/list/array 3X1
        observation: list of [("ModID",Params),("ModID",Params)...]
        OUTPUT: q: config of robot when achieving the final position (if cannot achieving it's the output of ik func)
        solves: True/False: can reach the goal or not
        """
        if base_location is None:
            base_location = [0, 0, 0]
        solves = []
        # goals(target), no constraint etc.
        # set target as up-straight if there's no restriction of orientation of end effector
        if target_rotation is None:
            reach_goal = Goals.Reach('Reach goal', tolerated_pose.ToleratedPose(spatial.homogeneous(target_location),
                                                                                CartesianXYZ.default()))
        else:
            reach_goal = Goals.Reach('Reach goal', tolerated_pose.ToleratedPose(spatial.homogeneous(
                target_location, target_rotation), CartesianXYZ.default()))
        robot, assembly = self.get_robot(observation=observation, base_location=base_location)
        q, solved = robot.ik_scipy(reach_goal.goal_pose)
        # only considered end pose, but no rotation for default
        if method == "ik":
            # q, solved = robot.ik(reach_goal.goal_pose, tolerance=CartesianXYZ.default())
            q, solved = robot.ik(reach_goal.goal_pose)
        solves.append(solved)
        return q, solves, assembly, robot

    def solve_scenario(self, trajectory_planner: 'ScenarioPlannerBase', scenario: Scenario,
                       cost_function: CostFunctionBase, base_location=None, observation: list = None):
        """
        solve given scenario using built robot, it uses ik but for simple robot ik may easily fail into singularity
        """
        if base_location is None:
            base_location = [0, 0, 0]
        robot, assembly = self.get_robot(observation=observation, base_location=base_location)
        _, solved = robot.ik(scenario.goals[0].goal_pose, tolerance=CartesianXYZ.default())
        if solved:
            trajectory = trajectory_planner.solve(robot=robot)
            header = SolutionHeader(scenarioID=scenario.id, date=datetime.now())
            solution = SolutionTrajectory(
                trajectory=trajectory,
                header=header,
                scenario=scenario,
                robot=robot,
                cost_function=cost_function)
            return solved, solution.cost, solution.valid, assembly, robot
        else:
            return solved, None, False, assembly, robot  # cannot reach target even without obstacles


class ParamModAss(ModuleAssembly):
    """
    integrate params to ModuleAssembly class, return a ModuleAssembly class with integrated and re-indexed modules
    and connections, mostly copied from ModuleAssembly class, modified init(mainly setting connection part),
    modified _add_module from "copy and re-index a new module_instance when facing same module type" to "copy and
    re-index and parametrize a new module_index when facing same module type"
    """

    connection_type = Tuple[int, str, int, str]

    def __init__(self,
                 database: ParametrizedModDB,
                 assembly_modules: Collection[str] = (),
                 base_connector: Tuple[int, str] = None,
                 obs: list = None):
        """
        inherit from ModAss
        assembly_module: list of module type
        obs: list of params in sequence of module types corresponding to assembly module
        """
        self.db = database
        self.obs = obs
        self.module_instances: List[ParametrizedModule] = []
        self._module_copies: Dict[str, str] = dict()
        self.connections: Set[Tuple[ParametrizedModule, Connector, ParametrizedModule, Connector]] = set()
        # add module to module_instances according to given module type & params:
        i = 0
        for module_id in assembly_modules:
            self._add_module(module_id, obs=self.obs[i])
            i += 1
        # set base module:
        self._base_connector = None
        if base_connector is None:
            con2base = [con for con in self.free_connectors.values() if con.type == 'base']
            if len(con2base) == 1:
                self._base_connector = con2base[0]
            elif len(self.module_instances) > 0:
                raise ValueError(
                    "There are {} candidates for a base connector in the assembly.".format(len(con2base)))

        # set connection as: searching all possible connections btw 2 modules,
        # if more than 2 possible then choose the one with smaller module_instance index
        for i in range(len(assembly_modules)):
            if i == 0:
                pass
            else:
                module_instance = self.module_instances[i]
                previous_module = self.module_instances[i - 1]
                # TODO: I copied it from the module part but it can only act like chain-structure now
                possible_connections = [(module_instance, con_id[-1], previous_module, to_con_id)
                                        for con_id, mod_con in module_instance.available_connectors.items()
                                        for to_con_id, to_con in self.free_connectors.items()
                                        if mod_con.connects(to_con) and previous_module.id == to_con_id[0]]
                if len(possible_connections) != 1:
                    print("There is no unique way to add module" + module_instance.id + " to the robot.")
                connection = possible_connections[0]
                new = connection[0]
                new_connector = new.connectors_by_own_id[connection[1]]
                to = connection[2]
                to_connector = to.connectors_by_own_id[connection[3][-1]]
                self.connections.add((new, new_connector, to, to_connector))
        # -------------------------check valid: not in db anymore ------------------------------------------------
        # for module in self.module_instances:
        #     if module.id not in self._module_copies:
        #         X = (module in self.db)
        #         assert module in self.db
        # try:
        #     self._assert_is_valid()
        # except AssertionError:
        #     raise err.InvalidAssemblyError("Robot Assembly seems to be invalid. Check the input arguments.")

    def _add_module(self, module_id: str, obs: list = None, set_base: bool = False) -> int:
        """
        modified _add_module from "copy and re-index a new module_instance when facing same module type" to "copy and
        re-index and parametrize a new module_index when facing same module type". re-index: from "1" to "1-1"
        obs: params of one module
        """
        # observation: [ModID,Params]
        if isinstance(module_id, ParametrizedModule):
            raise ValueError("You provided a Module, but an ID was expected.")
        db_by_id = self.db.by_id
        previous_identical = module_id in self.internal_module_ids
        # -------------- define base_connector, copied from original class, kept for backup --------------------------
        if set_base:
            module = db_by_id[module_id]
            if len(self.internal_module_ids) != 0:
                raise NotImplementedError("Cannot reset the base after having set one already.")
            base_connectors = [c for c in module.available_connectors.values() if c.type == 'base']
            if len([c for c in module.available_connectors.values() if c.type == 'base']) != 1:
                raise err.InvalidAssemblyError(
                    "A base module needs to have exactly one base connector if added this way.")
            self._base_connector = base_connectors[0]
        # -------------------------------------------------------------------------------------------------------------
        # below: re-indexing
        if not previous_identical:
            module = db_by_id[module_id]
            module = module.from_parameters(params=obs)
            self.module_instances.append(module)
        else:
            i = 1
            while True:
                # Find the first free module id + suffix id and add the module as such
                suffix = f'_{i}'
                if (module_id + suffix) not in self.internal_module_ids:
                    new_module = db_by_id[module_id]
                    new_module = new_module.from_parameters(params=obs, suffix_id=i)
                    self._module_copies[new_module.id] = module_id
                    self.module_instances.append(new_module)
                    break
                i += 1
        return len(self.module_instances)

    @property
    def get_mass(self) -> float:
        return sum([mod.get_mass for mod in self.module_instances])
