# this is to test if parametrized_mod_gen (and robot_assemble) works properly
# by Yizhen Li, 28,08,2022
import numpy as np

from mpdqn.env.modular_robot.envs.robot_assemble import RobAss
from mpdqn.parametrized_module.parametrized_mod_gen import db_full

# -------------------------test solve----------------------------------------------
test = RobAss(db=db_full)
# # 0: mod_i, 1: jnt_L, 2: jnt_L_better, 3: jnt_I, 4: mod_L, 5: jnt_rot, 6: base
# # obs = [("6", (0.1, 0.1)), ("1", (0.3, 0.1)),("4", (0.4, 0.2)), ("3", (0.1, 0.2)), ("0", (0.3)),
# #        ("4", (0.3, 0.5)),("1", (0.1, 0.2)),("0", (0.1)),("3", (0.05, 0.1)), ("0", (0.2)),("2", (0.7, 0.2))]
# # first one has to be the base_connector
# # (p(m), d(f)) for joint

# obs = [("6", (0.1, 0.1)), ("0", 0.3), ("2", (0.4, 0.2)), ("0", 0.1), ("2", (1.0, 0.5)), ("2", (2.0, 0.9)),
# ("2", (0.7, 0.2))]  # add better L joint
obs = [("6", (0.0, 0.0)), ("0", 0.3), ("5", (0.4, 0.2)), ("0", 0.1), ("5", (0.5, 0.5)), ("0", 0.3)]
q, solves, assembly, robot = test.solve(target_location=np.array([1, 1, 1]),
                                        target_rotation=np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), observation=obs)
print(solves, q)
# target = robot.fk(q)
# print(target)
# q2, solves2, assembly2, robot2 = test.solve(target_location=target[:3, 3], target_rotation=target[:3, :3],
#                                             observation=obs)
# print(solves2, q2)
# robot2.plot(coordinate_systems='joints')
# # assembly2.plot_graph(which="verbose")
# input('Press ENTER to continue...')
