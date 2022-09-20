# This is a small module DB to test the robot_assemble class. All modules are set as very simple cylinder
# with default(unuseful) r_cyl and h_cyl which are params to be changed by optimizer, here by manually setting obs
# original hardcoded modules without using parametrized class. compare it with folder parametrized_module to see
# what is parametrized_module be using for
# by Yizhen Li, 28,06,2022
import numpy as np
import pinocchio as pin
from numpy.linalg import norm

from Bodies import Body, Connector, ConnectorSet
from Geometry import Cylinder
from Joints import Joint
from Module import AtomicModule, ModulesDB
from mpdqn.env.modular_robot.envs.robot_assemble import RobAss
from utilities.spatial import homogeneous, rotX

# -----------------------------define module database -------------------------------------------------------
db = ModulesDB()
mass = 0
r_com = [0, 0, 0]
iner = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
inertia = pin.Inertia(float(mass), np.array(r_com, dtype=float), np.asarray(iner, dtype=float))

header = {"moduleName": "I_shape", "moduleID": 0}
r_cyl = 0.055
h_cyl = 0.1252
body_id_I = "cy_0"
collision = Cylinder(dict(r=r_cyl, z=h_cyl))
con_set01 = ConnectorSet()
connector0_1 = Connector(connector_id="con_01_01",
                         body2connector=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.0626], [0, 0, 0, 1]], gender="f")
con_set01.add(connector0_1)
connector0_2 = Connector(connector_id="con_01_02",
                         body2connector=[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, -0.0626], [0, 0, 0, 1]], gender="m")
con_set01.add(connector0_2)
body_I = Body(body_id="cylinder_b01", inertia=inertia, collision=collision, connectors=con_set01)
mod_test_01 = AtomicModule(header=header, bodies=[body_I])
db.add(mod_test_01)

header2 = {"moduleName": "I_shape_base", "moduleID": 2}
body_id_Ib = "cy_1"
collision2 = Cylinder(dict(r=r_cyl, z=h_cyl))
con_set02 = ConnectorSet()
connector2_1 = Connector(connector_id="con_02_01", connector_type='base',
                         body2connector=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.0626], [0, 0, 0, 1]], gender="m")
con_set02.add(connector2_1)
connector2_2 = Connector(connector_id="con_02_02",
                         body2connector=[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, -0.0626], [0, 0, 0, 1]], gender="f")
con_set02.add(connector2_2)
body_Ib = Body(body_id="cylinder_b02", inertia=inertia, collision=collision2, connectors=con_set02)
mod_test_02 = AtomicModule(header=header2, bodies=[body_Ib])
db.add(mod_test_02)

header1 = {"moduleName": "joint", "moduleID": 1}
mass1 = 2.795
r_com1 = [0.00018, 0.00013, 0.0543]
ine = [[0.0057589232145, 2.145403E-6, 3.065833E-5], [2.145403E-6, 0.005758879892, 1.4879905E-5],
       [3.065833E-5, 1.4879905E-5, 0.0039998622065]]
inertia1 = pin.Inertia(float(mass1), np.array(r_com1, dtype=float), np.asarray(ine, dtype=float))
connector_p = Connector(connector_id="proximal_con",
                        body2connector=[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], gender="m")
h_cyl_p = 0.1252
R_cyl = 0.055
pos = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, h_cyl_p / 2], [0, 0, 0, 1]]
pose1 = np.asarray(pos)
translation1 = pose1[:3, 3]
rotation1 = pose1[:3, :3]
collision3 = Cylinder(dict(r=R_cyl, z=h_cyl_p), homogeneous(translation=translation1, rotation=rotation1))
body_p = Body(body_id="proximal_b", inertia=inertia1, collision=collision3, connectors=[connector_p])

mass2 = 0.588
r_com2 = [0.00063, 0.00019, -0.00966]
ine2 = [[0.00078510920040000007, -3.766164E-7, 7.4215496E-6], [-3.766164E-7, 0.00077789705, -2.2621520000000003E-7],
        [7.4215496E-6, -2.2621520000000003E-7, 0.000731745396]]
inertia2 = pin.Inertia(float(mass2), np.array(r_com2, dtype=float), np.asarray(ine2, dtype=float))

connector_d = Connector(connector_id="distal", body2connector=rotX(np.pi / 2), gender="f")

h_cyl_d = 0.01
pose2 = np.asarray([[1, 0, 0, 0], [0, -1, -1.2246467991473532E-16, 0],
                    [0, 1.2246467991473532E-16, -1, -h_cyl_d / 2], [0, 0, 0, 1]])
translation2 = pose2[:3, 3]
rotation2 = pose2[:3, :3]
collision4 = Cylinder(dict(r=R_cyl, z=h_cyl_d), homogeneous(translation=translation2, rotation=rotation2))
body_d = Body(body_id="distal_b", inertia=inertia2, collision=collision4, connectors=[connector_d])

joint = Joint(joint_id="joint_1", joint_type="revolute", parent_body=body_p, child_body=body_d,
              gear_ratio=160, motor_inertia=0.00037, friction_coulomb=36.56, friction_viscous=51.46,
              velocity_limit=1.89, torque_limit=205.92,
              joint2child=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, h_cyl_p + h_cyl_d], [0, 0, 0, 1]],
              q_limits=(-3.141592654, 3.141592654))

mod_test_03 = AtomicModule(header=header1, bodies=[body_p, body_d], joints=[joint])
db.add(mod_test_03)
# -------------------------------- test RobAss--------------------------------------------------------------
test = RobAss(db=db)
obs = [("2", (0.2, 0.055)), ("1", (0.1, 0.2)), ("0", (0.4, 0.1)), ("0", (0.2, 0.055)), ("1", (0.2, 0.4)),
       ("0", (0.15, 0.055))]
# (modID, (h_cyl, r_cyl)) # first one has to be the base_connector
q, solves, _, robot = test.solve(target_location=[0.75, 0, 0.5], target_rotation=[[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                                 observation=obs)
EE_location = robot.fk(q)
reward = 1 - (norm(EE_location[:3, 3] - [0.75, 0, 0.5], ord=2)) / (norm([0.75, 0, 0.5], ord=2))  # check dist as reward
print(solves, q, reward)
t = robot.fk(q + np.pi / 3)[:3, 3]  # do fk to see if solve-func is working
q, solves, assembly, robot = test.solve(target_location=t, observation=obs)
print(test.solve(target_location=[0, -0.75, 0.5], target_rotation=[[1, 0, 0], [0, 0, -1], [0, -1, 0]], observation=obs))
# ---------------------------- check same module in conti-----------------------------------
print(solves, q, reward)
count = 0
m_id = tuple(mod.id[0] for mod in assembly.module_instances)
for i in range(len(m_id)):
    if m_id[i] == m_id[i - 1]:  # check how many module in module-instances is in conti
        count += 1
print(count)
# -------------------------------------------------------------------------------------------

# assembly.plot_graph(which="verbose")
robot.plot(coordinate_systems='tcp')
# input('Press ENTER to continue...')
