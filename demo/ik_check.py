# This should be inferring if ik can be used for the given simple scenario (?use ik_scipy instead if not)
# by Yizhen Li, 15,07,2022
import numpy as np

from mpdqn.env.modular_robot.envs.robot_assemble import ParamModAss, RobAss
from mpdqn.parametrized_module.Parametrized_module import ParametrizedCylMod, ParametrizedJntLMod, ParametrizedModDB
from scenario import Goals
from scenario.Tolerance import CartesianXYZ
from utilities import tolerated_pose
from utilities.spatial import homogeneous

# ---------------------------set module db --------------------------------------------------
db = ParametrizedModDB()

mod_i = ParametrizedCylMod(header={"moduleName": "I_shape", "moduleID": 0}, body_id="cyl_b0",
                           h=1.0, con_id=["con_cyl_01", "con_cyl_02"])
db.add(mod_i)  # add I shape module to db

jnt_L_better = ParametrizedJntLMod(header={"moduleName": "joint_L_better", "moduleID": 1},
                                   body_id=["proximal", "distal"], params=[1.5, 2],
                                   con_id=["proximal_con", "distal_con"],
                                   joint_id="joint_1")
db.add(jnt_L_better)  # add a better L shape joint module to db (param are length of 2 directions)

mod_i_base = ParametrizedCylMod(header={"moduleName": "I_shape_base", "moduleID": 2}, body_id="cyl_b2",
                                h=1.0, con_id=["con_cyl_b_01", "con_cyl_b_02"], set_base=True)
db.add(mod_i_base)  # set I-shaple as base module
# ---------------------------------------------------------------------------------------------

# below: reproducing RobAss.solve:
print("below is result from reproducing Robass.solve:")
module_list = ['2', '1', '0']
param_list = [0.0, (0.2, 0.3), 0.4]
assembly = ParamModAss(db, module_list, obs=param_list)
robot = assembly.to_pin_robot(homogeneous(np.array([0, 0, 0])))
reach_goal0 = Goals.Reach('Reach goal', tolerated_pose.ToleratedPose(homogeneous(np.array([0, 0, 0])),
                                                                     CartesianXYZ.default()))
q, solved = robot.ik_scipy(reach_goal0.goal_pose)
print(robot.fk(q))
reach_goal = Goals.Reach('Reach goal', tolerated_pose.ToleratedPose(robot.fk(q), CartesianXYZ.default()))
q, solved = robot.ik(reach_goal.goal_pose)
print("solved by ik?", solved, q)
q, solved = robot.ik_scipy(reach_goal.goal_pose)
print("solved by ik_scipy?", solved, q)

# repeat the same process with test.solve:
print("below is result from Robass.solve:")
obs = [('2', 0.0), ('1', (0.2, 0.3)), ('0', 0.4)]
test = RobAss(db=db)
q, solves, assembly2, robot2 = test.solve(target_location=reach_goal.goal_pose[:3, 3],
                                          target_rotation=reach_goal.goal_pose[:3, :3], observation=obs, method="ik")
print(robot.fk(q))
print("solved by ik?", solves, q)

# but if changing ik to ik_scipy in RobAss.solve (line 65 in robot_assemble.py) it works again
q, solves, assembly2, robot2 = test.solve(target_location=reach_goal.goal_pose[:3, 3],
                                          target_rotation=reach_goal.goal_pose[:3, :3], observation=obs)
print(robot.fk(q))
print("solved by ik_scipy?", solves, q)
