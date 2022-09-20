# This is to generate parametrized body and module class
# by Yizhen LI, 29,08,2022
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pinocchio as pin

from Bodies import Body, BodySet, Connector, ConnectorSet
from Geometry import ComposedGeometry, Cylinder
from Joints import Joint, JointSet
from Module import AtomicModule, ModuleHeader, ModulesDB
from utilities.dtypes import SingleSet
from utilities.spatial import homogeneous, rotX


class ParametrizedBody(Body, ABC):
    """
    This class is inherited from class body with self_defining parameter. skeleton of parametrized class
    """

    # def __init__(self,
    #              body_id: str,
    #              connectors: Iterable[Connector] = None,
    #              inertia: pin.Inertia = pin.Inertia(0, np.zeros(3), np.zeros((3, 3))),
    #              collision: ObstacleBase = None,
    #              visual: ObstacleBase = None,
    #              in_module: 'AtomicModule' = None
    #              ):
    #     """
    #     same as in class Body except for init self.rotate
    #     """
    #     super().__init__(body_id, connectors, inertia, collision, visual, in_module)
    #     self.rotate = None

    @property
    @abstractmethod
    def parameters(self) -> Tuple[float]:
        """implemented by children: return params of body"""

    @abstractmethod
    def from_parameters(self, params, one_side, rotate, suffix_id=None):
        """implemented by children: create body from given parameters"""


class ParametrizedCylinder(ParametrizedBody):
    """
    This class is for creating a parametrized cylinder-shape body: Z-axis points along height, r_com at center,
    initial 2 connector at the end of cylinder, pointing along Z-axis to outside, gender as female and male separately
    """

    def __init__(self, body_id: str, h_cyl: float = 0.1252, r_cyl: float = 0.055, connectors_id: List[str] = None,
                 mass: float = .0, con_gender=None, r_com: np.ndarray = np.array((0, 0, 0), dtype=float),
                 one_side=None, rotate: bool = False, set_base=False):
        """
        h_cyl: height of the cylinder, can be set in from_parameters
        r_cyl: radius of cylinder, cannot be set in from_parameters
        connectors_id: list of connector's id in body
        one_side: false if this side of connector is not needed, [True, True] for both connector activated
        rotate: true if rotate second connector's direction with 90 grad (along radius of cylinder)
        """
        if mass == .0:
            mass = h_cyl  # mass is proportional to length of cylinder
        if con_gender is None:
            con_gender = ["m", "f"]
        if one_side is None:
            one_side = [True, True]
        self.h_cyl = h_cyl
        self.r_cyl = r_cyl
        self.rotate = rotate
        self.set_base = set_base
        inertia_tensor = np.zeros((3, 3), dtype=float)
        inertia = pin.Inertia(float(mass),
                              np.array(r_com, dtype=float),
                              np.asarray(inertia_tensor, dtype=float))
        self.inertia = inertia
        collision = Cylinder(dict(r=r_cyl, z=h_cyl))
        # collision = Cylinder(body_id, r_cyl, h_cyl)
        connectors = ConnectorSet()
        con_idx = 0
        if one_side[0] is True:
            pose = (np.eye(4, dtype=float)) @ rotX(np.pi)
            pose[2, 3] = -h_cyl / 2
            if rotate is True:
                pose = pose @ rotX(np.pi / 2)
            if set_base is not True:
                connector = Connector(connector_id=connectors_id[con_idx], body2connector=pose, gender=con_gender[0])
            else:
                connector = Connector(connector_id=connectors_id[con_idx],
                                      body2connector=pose, gender=con_gender[0],
                                      connector_type="base")  # base is always the first con aka m con
            connectors.add(connector)
            con_idx += 1
        if one_side[1] is True:
            pose = np.eye(4, dtype=float)
            pose[2, 3] = h_cyl / 2
            if rotate is True:
                pose = pose @ rotX(np.pi / 2)
            connector = Connector(connector_id=connectors_id[con_idx],
                                  body2connector=pose, gender=con_gender[1])
            connectors.add(connector)
        super().__init__(body_id,  collision, collision, connectors, inertia)

    @property
    def parameters(self) -> Tuple[float]:
        """return params of cylinder: height"""
        return [self.h_cyl]

    def from_parameters(self, params, suffix_id=None, one_side=None, rotate=False):
        """
        create cylinder shape body from given param: h_cyl and add suffix to body_id
        """
        if one_side is None:
            one_side = [True, True]
        if suffix_id is None:
            suffix = ""
        else:
            suffix = f'_{suffix_id}'
        new_id = self._id + suffix
        new_con_id = [con.own_id + suffix for con in self.connectors]
        set_base = self.set_base
        return self.__class__(body_id=new_id, h_cyl=params, connectors_id=new_con_id, one_side=one_side, rotate=rotate,
                              set_base=set_base)


class ParametrizedL(ParametrizedBody):
    """
        This class is for creating a parametrized L-shape body: Z-axis points along height of the first cylinder,
        init 2 connector at the end of the L shape, pointing along heights to outside, gender as f and m separately
        """

    def __init__(self, body_id: str, cyl_id: List[str], h=None, r_cyl: float = 0.055, connectors_id: List[str] = None,
                 mass: float = .0, con_gender=None, r_com: np.ndarray = np.array((0, 0, 0), dtype=float),
                 one_side=None, set_base=False):
        if h is None:
            h = [0.1252, 0.1252]
        if mass == 0:
            mass = sum(h)
        if con_gender is None:
            con_gender = ["m", "f"]
        if one_side is None:
            one_side = [True, True]
        self.h_1 = h[0]
        self.h_2 = h[1]
        self.r_cyl = r_cyl
        self.set_base = set_base
        inertia_tensor = np.zeros((3, 3), dtype=float)
        inertia = pin.Inertia(float(mass),
                              np.array(r_com, dtype=float),
                              np.asarray(inertia_tensor, dtype=float))
        self.inertia = inertia
        rotation = (np.eye(4, dtype=float) @ rotX(np.pi / 2))[:3, :3]
        translation = np.array([0.0, -self.r_cyl + self.h_2 / 2, self.r_cyl + self.h_1 / 2])
        self.obstacle = [Cylinder(dict(r=r_cyl, z=h[0])),
                         Cylinder(dict(r=r_cyl, z=h[1]), homogeneous(translation, rotation))]
        # self.obstacle = [Cylinder(cyl_id[0], r_cyl, h[0]), Cylinder(cyl_id[1], r_cyl, h[1], rotation=rotation,
        # translation=translation)]
        # L shape contains of 2 cylinder, with h[0] as center,
        # rotation/translation is the position of h[1] refering to h[0]
        collision = ComposedGeometry(self.obstacle)
        # collision = ComposedObstacle(body_id,
        # obstacles=self.obstacle, translation=NO_TRANSLATION, rotation=NO_ROTATION)
        # -------------------use this lines if you want to see the visualization of L-body------------------------
        # header = ScenarioHeader('test', 'PyTest')
        # scen = Scenario(header)
        # scen.add_obstacle(collision)
        # viz = scen.visualize()
        # collision.visualize(viz)
        connectors = ConnectorSet()
        con_idx = 0
        if one_side[0] is True:
            pose = (np.eye(4, dtype=float)) @ rotX(np.pi)
            pose[2, 3] = -self.h_1 / 2
            if set_base is not True:
                connector = Connector(connector_id=connectors_id[con_idx], body2connector=pose, gender=con_gender[0])
            else:
                connector = Connector(connector_id=connectors_id[con_idx],
                                      body2connector=pose, gender=con_gender[0],
                                      connector_type="base")  # base is always m con
            connectors.add(connector)
            con_idx += 1
        if one_side[1] is True:
            pose = np.eye(4, dtype=float) @ rotX(-np.pi / 2)
            pose[1, 3] = self.h_2 - self.r_cyl
            pose[2, 3] = self.h_1 / 2 + self.r_cyl
            connector = Connector(connector_id=connectors_id[con_idx],
                                  body2connector=pose, gender=con_gender[1])
            connectors.add(connector)
        super().__init__(body_id,  collision, collision, connectors, inertia)

    @property
    def parameters(self) -> Tuple[float]:
        """return params of 2 cylinder: height (radius are the same)"""
        return [self.h_1, self.h_2]

    def from_parameters(self, params, suffix_id=None, one_side=None, rotate=False):
        """
        create L shape body from given param: h_cyl and add suffix to body_id
        """
        if one_side is None:
            one_side = [True, True]
        if suffix_id is None:
            suffix = ""
        else:
            suffix = f'_{suffix_id}'
        new_id = self._id + suffix
        new_con_id = [con.own_id + suffix for con in self.connectors]
        new_cyl_id = [cyl.id + suffix for cyl in self.obstacle]
        set_base = self.set_base
        return self.__class__(body_id=new_id, cyl_id=new_cyl_id, h=params, connectors_id=new_con_id,
                              set_base=set_base, one_side=one_side)


class ParametrizedModule(AtomicModule, ABC):
    """
    inherited from class AtomicModule with self_defining parameter. skeleton of parametrized class
    """

    @property
    @abstractmethod
    def parameters(self) -> Tuple[float]:
        """implemented by children: return params of module"""

    @abstractmethod
    def from_parameters(self, params, suffix_id=None):
        """implement by children: a method to parametrize different modules"""

    @property
    def bodies(self) -> BodySet[ParametrizedBody]:
        """Returns all bodies contained in this module."""
        return self._bodies

    @property
    @abstractmethod
    def get_mass(self) -> float:
        """returns total mass of all body in module"""


class ParametrizedCylMod(ParametrizedModule):
    """
    This class is for creating a parametrized cylinder-shape module: contains only one cylinder shape body and no joint
    """

    def __init__(self,
                 header: Union[Dict, ModuleHeader],
                 body_id: str = None, h: float = None, con_id: List[str] = None, set_base=False,
                 bodies: Iterable[Body] = (),
                 joints: Iterable[Joint] = ()
                 ):
        """
        mainly the same as in AtomicModule
        redefining bodies for updating the initial parameter bodies from [Body] to [ParametrizedBody]
        """
        if bodies == ():
            bodies = [ParametrizedCylinder(body_id=body_id, h_cyl=h, connectors_id=con_id, set_base=set_base)]
        super().__init__(header, bodies, joints)
        self._bodies: BodySet[ParametrizedCylinder] = BodySet(bodies)

    @property
    def parameters(self) -> Tuple[float]:
        """return params of cylinder body: height"""
        return [body.parameters for body in self._bodies]

    def from_parameters(self, params, suffix_id=None):
        """
        create cylinder shape module from given param: h_cyl for body and add suffix to body_id & module_id
        """
        if suffix_id is None:
            suffix = ""
        else:
            suffix = f'_{suffix_id}'
        new_header = ModuleHeader(moduleID=self.id + suffix, moduleName=self.name + suffix)
        new_bodies = BodySet([body.from_parameters(params, suffix_id) for body in self._bodies])
        for body in new_bodies:
            body.in_module = self
        new_joints = self.joints
        return self.__class__(header=new_header, bodies=new_bodies, joints=new_joints)

    @property
    def bodies(self) -> BodySet[ParametrizedBody]:
        """Returns all bodies contained in this module."""
        return self._bodies

    @property
    def get_mass(self) -> float:
        return sum([body.mass for body in self._bodies])


class ParametrizedLMod(ParametrizedModule):
    """
    This class is for creating a parametrized L-shape module: contains one L shape body and no joint
    """

    def __init__(self,
                 header: Union[Dict, ModuleHeader], cyl_id: List[str] = None,
                 body_id: List[str] = None, h: List[float] = None, con_id: List[str] = None, set_base=False,
                 bodies: Iterable[Body] = (),
                 joints: Iterable[Joint] = ()
                 ):
        """
        need it for updating the initial parameter bodies from [Body] to [ParametrizedBody]
        """
        if bodies == ():
            bodies = [ParametrizedL(body_id=body_id, cyl_id=cyl_id, h=h, connectors_id=con_id, set_base=set_base)]
        super().__init__(header, bodies, joints)
        self._bodies: BodySet[ParametrizedCylinder] = BodySet(bodies)

    @property
    def parameters(self) -> Tuple[float]:
        """return params of the 2 cylinder in the L body: 2 heights"""
        return [body.parameters for body in self._bodies]

    def from_parameters(self, params, suffix_id=None):
        """
        create cylinder shape module from given param:
        params as h for 2 cylinder in body and add suffix to body_id & module_id
        """
        if suffix_id is None:
            suffix = ""
        else:
            suffix = f'_{suffix_id}'
        new_header = ModuleHeader(moduleID=self.id + suffix, moduleName=self.name + suffix)
        new_bodies = BodySet([body.from_parameters(params, suffix_id) for body in self._bodies])
        for body in new_bodies:
            body.in_module = self
        new_joints = self.joints
        return self.__class__(header=new_header, bodies=new_bodies, joints=new_joints)

    @property
    def bodies(self) -> BodySet[ParametrizedBody]:
        """Returns all bodies contained in this module."""
        return self._bodies

    @property
    def get_mass(self) -> float:
        return sum([body.mass for body in self._bodies])


class ParametrizedJntMod(ParametrizedModule):
    """
    This class is for creating a parametrized joint module: contains two I shape bodies and one rotational joint,
    each body has one connector, connectors are at the end of both bodies, pointing towards Z-axis and outside.
    by defining one of the connector can be rotated 90 grad (see parametrized cylinder body)
    """

    def __init__(self,
                 header: Union[Dict, ModuleHeader],
                 body_id: List[str] = None, params: List[float] = None, con_id: List[str] = None,
                 joint_id: str = None, joint_type="revolute", q_limits=(-3.141592654, 3.141592654),
                 gear_ratio=160, motor_inertia=0.00037, friction_coulomb=36.56, friction_viscous=51.46,
                 velocity_limit=1.89, torque_limit=205.92,
                 bodies: Iterable[Body] = (),
                 joints: Iterable[Joint] = (),
                 set_base=False, l_shape_jnt=True
                 ):
        """
        l_shape_jnt: true if one con is rotate to make the joint seems as L shape
        if in from_parameters bodies & joints are defined, use them for build an original AtomicModule, if not defined,
        set bodies as 2 cylinder and joint position same as r_com of the parent body
        """
        self.l_shape_jnt = l_shape_jnt
        if bodies == ():
            bodies = [ParametrizedCylinder(body_id=body_id[0], h_cyl=params[0], connectors_id=[con_id[0]],
                                           one_side=[True, False], set_base=set_base),
                      ParametrizedCylinder(body_id=body_id[1], h_cyl=params[1], connectors_id=[con_id[1]],
                                           one_side=[False, True], rotate=l_shape_jnt)]
        if joints == ():
            j2c = np.eye(4, dtype=float)
            j2c[2, 3] = (params[0] + params[1]) / 2
            joints = [Joint(joint_id=joint_id, joint_type=joint_type, parent_body=bodies[0], child_body=bodies[1],
                            joint2child=j2c, q_limits=q_limits, gear_ratio=gear_ratio, motor_inertia=motor_inertia,
                            friction_viscous=friction_viscous, friction_coulomb=friction_coulomb,
                            velocity_limit=velocity_limit, torque_limit=torque_limit)]
        super().__init__(header, bodies, joints)
        self._bodies: BodySet[ParametrizedCylinder] = BodySet(bodies)

    @property
    def parameters(self) -> Tuple[float]:
        """return params of the 2 cylinder body: 2 heights"""
        return [body.parameters for body in self._bodies]

    def from_parameters(self, params: List[float], suffix_id=None):  # params: [parent_h(p), child_h(d)]
        """
        create joint module from given param:
        params as h for 2 cylinder body and add suffix to body_id & module_id
        """
        if suffix_id is None:
            suffix = ""
        else:
            suffix = f'_{suffix_id}'
        new_header = ModuleHeader(moduleID=self.id + suffix, moduleName=self.name + suffix)
        new_bodies = BodySet()
        bodies_by_original_id = {}
        if self.l_shape_jnt is True:
            # l_shape_jnt is true, means the module is "L-shape joint", rotate cons as in selves bodies
            i = 0
            for body in self._bodies:
                original_id = body.id
                rotate = body.rotate
                inv_r = not rotate
                one_side = [inv_r, rotate]  # cylinder body only have one side con: here set as the rotate side
                body = body.from_parameters(params[i], suffix_id, one_side=one_side, rotate=rotate)
                body.in_module = self
                new_bodies.add(body)
                bodies_by_original_id.update({original_id: body})
                i += 1
        else:
            for body in self.bodies:
                cons = [cons for cons in body.connectors]
                gender = [con.gender for con in cons]
                if gender[0].name == 'm':  # cylinder body only have one side con: set as the same in self
                    one_side = [True, False]
                    i = 0
                else:
                    one_side = [False, True]
                    i = 1
                original_id = body.id
                rotate = body.rotate
                body = body.from_parameters(params=params[i], suffix_id=suffix_id, one_side=one_side, rotate=rotate)
                body.in_module = self
                new_bodies.add(body)
                bodies_by_original_id.update({original_id: body})  # dict of original module id and updated_body
        new_joints = JointSet()
        for jnt in self.joints:
            joint_id = jnt.id[1] + suffix
            # link joint to the updated bodies coming from the original parent&child bodies
            parent_body = bodies_by_original_id[jnt.parent_body.id]
            child_body = bodies_by_original_id[jnt.child_body.id]
            joint_type = jnt.type
            j2c = np.eye(4, dtype=float)
            j2c[2, 3] = (params[0] + params[1]) / 2
            new_joints.add(
                Joint(joint_id=joint_id, joint_type=joint_type, parent_body=parent_body, child_body=child_body,
                      joint2child=j2c, q_limits=jnt.limits, gear_ratio=jnt.gear_ratio, motor_inertia=jnt.motor_inertia,
                      friction_viscous=jnt.friction_viscous, friction_coulomb=jnt.friction_coulomb,
                      velocity_limit=jnt.velocity_limit, torque_limit=jnt.torque_limit))
        return self.__class__(header=new_header, bodies=new_bodies, joints=new_joints)

    @property
    def bodies(self) -> BodySet[ParametrizedBody]:
        """Returns all bodies contained in this module."""
        return self._bodies

    @property
    def get_mass(self) -> float:
        return sum([body.mass for body in self._bodies])


class ParametrizedJntLMod(ParametrizedModule):
    """
        This class is for creating a better parametrized L shape joint module: shape as L body but have joint between
        the two cylinder bodies of L. contains two I shape bodies and one rotational joint, each body has one connector,
        connectors are at the end of both bodies, pointing towards Z-axis and outside.
        """

    def __init__(self,
                 header: Union[Dict, ModuleHeader],
                 body_id: List[str] = None, params: List[float] = None, con_id: List[str] = None,
                 joint_id: str = None, joint_type="revolute", q_limits=(-3.141592654, 3.141592654),
                 gear_ratio=160, motor_inertia=0.00037, friction_coulomb=36.56, friction_viscous=51.46,
                 velocity_limit=1.89, torque_limit=205.92,
                 bodies: Iterable[Body] = (),
                 joints: Iterable[Joint] = (),
                 set_base=False, r_cyl: float = 0.055
                 ):
        """
        if in from_parameters bodies & joints are defined, use them for build an original AtomicModule, if not defined,
        set bodies as 2 cylinder and joint position same as r_com of the parent body
        """
        self.r_cyl = r_cyl
        if bodies == ():
            bodies = [ParametrizedCylinder(body_id=body_id[0], h_cyl=params[0], connectors_id=[con_id[0]],
                                           one_side=[True, False], set_base=set_base),
                      ParametrizedCylinder(body_id=body_id[1], h_cyl=params[1], connectors_id=[con_id[1]],
                                           one_side=[False, True])]
        if joints == ():
            rotation = (np.eye(4, dtype=float) @ rotX(-np.pi / 2))[:3, :3]
            translation = np.array([0.0, self.r_cyl - params[1] / 2, self.r_cyl + params[0] / 2])
            # rotation/translation is similar as in CompactObstacle in L shape body
            joints = [Joint(joint_id=joint_id, joint_type=joint_type, parent_body=bodies[0], child_body=bodies[1],
                            joint2child=homogeneous(translation=translation, rotation=rotation),
                            q_limits=q_limits, gear_ratio=gear_ratio, motor_inertia=motor_inertia,
                            friction_viscous=friction_viscous, friction_coulomb=friction_coulomb,
                            velocity_limit=velocity_limit, torque_limit=torque_limit)]
        super().__init__(header, bodies, joints)
        self._bodies: BodySet[ParametrizedCylinder] = BodySet(bodies)

    @property
    def parameters(self) -> Tuple[float]:
        """return params of the 2 cylinder body: 2 heights"""
        return [body.parameters for body in self._bodies]

    def from_parameters(self, params: List[float], suffix_id=None):  # params: [parent_h(p), child_h(d)]
        """
        create joint module from given param:
        params as h for 2 cylinder body and add suffix to body_id & module_id
        """
        if suffix_id is None:
            suffix = ""
        else:
            suffix = f'_{suffix_id}'
        new_header = ModuleHeader(moduleID=self.id + suffix, moduleName=self.name + suffix)
        new_bodies = BodySet()
        bodies_by_original_id = {}
        for body in self.bodies:
            cons = [cons for cons in body.connectors]
            gender = [con.gender for con in cons]
            if gender[0].name == 'm':  # cylinder body only have one side con: set as the same in self
                one_side = [True, False]
                i = 0
            else:
                one_side = [False, True]
                i = 1
            original_id = body.id
            rotate = body.rotate
            body = body.from_parameters(params=params[i], suffix_id=suffix_id, one_side=one_side, rotate=rotate)
            body.in_module = self
            new_bodies.add(body)
            bodies_by_original_id.update({original_id: body})
        new_joints = JointSet()
        for jnt in self.joints:
            joint_id = jnt.id[1] + suffix
            # link joint to the updated bodies coming from the original parent&child bodies
            parent_body = bodies_by_original_id[jnt.parent_body.id]
            child_body = bodies_by_original_id[jnt.child_body.id]
            joint_type = jnt.type
            rotation = (np.eye(4, dtype=float) @ rotX(np.pi / 2))[:3, :3]
            translation = np.array([0.0, self.r_cyl - params[1] / 2, self.r_cyl + params[0] / 2])
            new_joints.add(
                Joint(joint_id=joint_id, joint_type=joint_type, parent_body=parent_body, child_body=child_body,
                      joint2child=homogeneous(translation=translation, rotation=rotation),
                      q_limits=jnt.limits, gear_ratio=jnt.gear_ratio, motor_inertia=jnt.motor_inertia,
                      friction_viscous=jnt.friction_viscous, friction_coulomb=jnt.friction_coulomb,
                      velocity_limit=jnt.velocity_limit, torque_limit=jnt.torque_limit))
        return self.__class__(header=new_header, bodies=new_bodies, joints=new_joints)

    @property
    def bodies(self) -> BodySet[ParametrizedBody]:
        """Returns all bodies contained in this module."""
        return self._bodies

    @property
    def get_mass(self) -> float:
        return sum([body.mass for body in self._bodies])


class ParametrizedJntRotMod(ParametrizedModule):
    """
        This class is for creating a parametrized rotational joint module: rotate in the plate of Z axis of 2 cylinder.
        contains two I shape bodies and one rotational joint, each body has one connector,
        connectors are at the end of both bodies, pointing towards Z-axis and outside.
        """

    def __init__(self,
                 header: Union[Dict, ModuleHeader],
                 body_id: List[str] = None, params: List[float] = None, con_id: List[str] = None,
                 joint_id: str = None, joint_type="revolute", q_limits=(-3.141592654, 3.141592654),
                 gear_ratio=160, motor_inertia=0.00037, friction_coulomb=36.56, friction_viscous=51.46,
                 velocity_limit=1.89, torque_limit=205.92,
                 bodies: Iterable[Body] = (),
                 joints: Iterable[Joint] = (),
                 set_base=False, r_cyl: float = 0.055
                 ):
        """
        if in from_parameters bodies & joints are defined, use them for build an original AtomicModule, if not defined,
        set bodies as 2 cylinder and joint position same as r_com of the parent body
        """
        self.r_cyl = r_cyl
        if bodies == ():
            bodies = [ParametrizedCylinder(body_id=body_id[0], h_cyl=params[0], connectors_id=[con_id[0]],
                                           one_side=[True, False], set_base=set_base),
                      ParametrizedCylinder(body_id=body_id[1], h_cyl=params[1], connectors_id=[con_id[1]],
                                           one_side=[False, True])]
        if joints == ():
            p2j_rotation = (np.eye(4, dtype=float) @ rotX(np.pi / 2))[:3, :3]
            p2j_translation = np.array([0.0, 0.0, params[0] / 2])
            j2c_rotation = (np.eye(4, dtype=float) @ rotX(-np.pi / 2))[:3, :3]
            j2c_translation = np.array([0.0, params[1] / 2, 0.0])
            # joint is at the end of parent body,
            # rotate to pi/2 towards outside of the plate forming by z-axis of 2 cylinder
            joints = [Joint(joint_id=joint_id, joint_type=joint_type, parent_body=bodies[0], child_body=bodies[1],
                            joint2child=homogeneous(translation=j2c_translation, rotation=j2c_rotation),
                            parent2joint=homogeneous(translation=p2j_translation, rotation=p2j_rotation),
                            q_limits=q_limits, gear_ratio=gear_ratio, motor_inertia=motor_inertia,
                            friction_viscous=friction_viscous, friction_coulomb=friction_coulomb,
                            velocity_limit=velocity_limit, torque_limit=torque_limit)]
        super().__init__(header, bodies, joints)
        self._bodies: BodySet[ParametrizedCylinder] = BodySet(bodies)

    @property
    def parameters(self) -> Tuple[float]:
        """return params of the 2 cylinder body: 2 heights and 2 radius"""
        return [body.parameters for body in self._bodies]

    def from_parameters(self, params: List[float], suffix_id=None):  # params: [parent_h(p), child_h(d)]
        """
        create joint module from given param:
        params as h for 2 cylinder body and add suffix to body_id & module_id
        """
        if suffix_id is None:
            suffix = ""
        else:
            suffix = f'_{suffix_id}'
        new_header = ModuleHeader(moduleID=self.id + suffix, moduleName=self.name + suffix)
        new_bodies = BodySet()
        bodies_by_original_id = {}
        for body in self.bodies:
            cons = [cons for cons in body.connectors]
            gender = [con.gender for con in cons]
            if gender[0].name == 'm':  # cylinder body only have one side con: set as the same in self
                one_side = [True, False]
                i = 0
            else:
                one_side = [False, True]
                i = 1
            original_id = body.id
            rotate = body.rotate
            body = body.from_parameters(params=params[i], suffix_id=suffix_id, one_side=one_side, rotate=rotate)
            body.in_module = self
            new_bodies.add(body)
            bodies_by_original_id.update({original_id: body})
        new_joints = JointSet()
        for jnt in self.joints:
            joint_id = jnt.id[1] + suffix
            # link joint to the updated bodies coming from the original parent&child bodies
            parent_body = bodies_by_original_id[jnt.parent_body.id]
            child_body = bodies_by_original_id[jnt.child_body.id]
            joint_type = jnt.type
            p2j_rotation = (np.eye(4, dtype=float) @ rotX(np.pi / 2))[:3, :3]
            p2j_translation = np.array([0.0, 0.0, params[0] / 2])
            j2c_rotation = (np.eye(4, dtype=float) @ rotX(-np.pi / 2))[:3, :3]
            j2c_translation = np.array([0.0, params[1] / 2, 0.0])
            new_joints.add(
                Joint(joint_id=joint_id, joint_type=joint_type, parent_body=parent_body, child_body=child_body,
                      joint2child=homogeneous(translation=j2c_translation, rotation=j2c_rotation),
                      parent2joint=homogeneous(translation=p2j_translation, rotation=p2j_rotation),
                      q_limits=jnt.limits, gear_ratio=jnt.gear_ratio, motor_inertia=jnt.motor_inertia,
                      friction_viscous=jnt.friction_viscous, friction_coulomb=jnt.friction_coulomb,
                      velocity_limit=jnt.velocity_limit, torque_limit=jnt.torque_limit))
        return self.__class__(header=new_header, bodies=new_bodies, joints=new_joints)

    @property
    def bodies(self) -> BodySet[ParametrizedBody]:
        """Returns all bodies contained in this module."""
        return self._bodies

    @property
    def get_mass(self) -> float:
        return sum([body.mass for body in self._bodies])


class ParametrizedModDB(ModulesDB, SingleSet[ParametrizedModule]):
    """
    This class is used to build ModuleDB from parametrized modules, inherited from ModulesDB
    """
    connection_type = Tuple[ParametrizedModule, Connector, ParametrizedModule, Connector]

    def __contains__(self, item: ParametrizedModule) -> bool:
        """see ModulesDB"""
        super().__contains__(item)

    def add(self, element: ParametrizedModule) -> None:
        """see ModulesDB"""
        super().add(element)

    @property
    def by_id(self) -> Dict[str, ParametrizedModule]:
        """Returns this DB as a dictionary, mapping the Module ID to the module"""
        return {mod.id: mod for mod in self}

    @property
    def by_name(self) -> Dict[str, ParametrizedModule]:
        """Returns this DB as a dictionary, mapping the Module Name to the module"""
        return {mod.name: mod for mod in self}

    @property
    def get_param_num(self) -> int:
        """returns the num of all parameters in this module DB"""
        param_num = 0
        for mod in self:
            param_num += sum([len(listElem) for listElem in mod.parameters])
        return param_num - 2

    @property
    def get_zero_param_list(self) -> list:
        """get a zero list of all params in the db"""
        zero_list = []
        for i in range(len(self)-1):
            mod_param_num = sum([len(listElem) for listElem in self.by_id[str(i)].parameters])
            zero_list.append(np.zeros(mod_param_num, ))
        return zero_list

    @property
    def max_param_num(self) -> int:
        """get the biggest num of parameters individual module can have, among all modules in DB"""
        param_num = []
        for mod in self:
            param_num.append(sum([len(listElem) for listElem in mod.parameters]))
        return max(param_num)
