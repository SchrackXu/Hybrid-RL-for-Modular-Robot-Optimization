# this file is used to generate module database
# by Yizhen LI, 29,08,2022
from mpdqn.parametrized_module.Parametrized_module import ParametrizedCylMod, ParametrizedJntLMod, ParametrizedJntMod, \
    ParametrizedJntRotMod, ParametrizedLMod, ParametrizedModDB

db_full = ParametrizedModDB()
db = ParametrizedModDB()
# below are instances of all classes in parametrized_module, but must not all be used in one application.
# choose useful ones for your applications
mod_i = ParametrizedCylMod(header={"moduleName": "I_shape", "moduleID": 0}, body_id="cyl_b0",
                           h=1.0, con_id=["con_cyl_01", "con_cyl_02"])
db.add(mod_i)
db_full.add(mod_i)  # add I shape module to db
jnt_rot = ParametrizedJntRotMod(header={"moduleName": "joint_rot", "moduleID": 1},
                                body_id=["proximal", "distal"], params=[1.5, 2],
                                con_id=["proximal_con", "distal_con"],
                                joint_id="joint_rot")
db.add(jnt_rot)
db_full.add(jnt_rot)  # add rotational joint in other z axis to db
jnt_L_better = ParametrizedJntLMod(header={"moduleName": "joint_L_better", "moduleID": 2},
                                   body_id=["proximal", "distal"], params=[1.5, 2],
                                   con_id=["proximal_con", "distal_con"],
                                   joint_id="joint_L_better")
db_full.add(jnt_L_better)  # add a better L shape joint module to db (param are length of 2 directions)
jnt_I = ParametrizedJntMod(header={"moduleName": "joint_I", "moduleID": 3},
                           body_id=["proximal", "distal"], params=[1.5, 2], con_id=["proximal_con", "distal_con"],
                           joint_id="joint_I", l_shape_jnt=False)
db_full.add(jnt_I)  # add I shape joint module to db
mod_L = ParametrizedLMod(header={"moduleName": "L_shape", "moduleID": 4},
                         body_id="L_b0", cyl_id=["L_cyl_1", "L_cyl_2"], h=[1.5, 2], con_id=["con_L_01", "con_L_02"])
db_full.add(mod_L)  # add L shape no joint module to db
jnt_L = ParametrizedJntMod(header={"moduleName": "joint_L", "moduleID": 5},
                           body_id=["proximal", "distal"], params=[1.5, 2], con_id=["proximal_con", "distal_con"],
                           joint_id="joint_L")
db_full.add(jnt_L)  # add L shape joint module to db (only rotate one connector's direction of I shape joint)
mod_L_base = ParametrizedLMod(header={"moduleName": "L_shape", "moduleID": 6},
                              body_id="L_b0", cyl_id=["L_cyl_1", "L_cyl_2"], h=[1.5, 2],
                              con_id=["con_L_01", "con_L_02"], set_base=True)
# db.add(mod_L_base) #set L shape module as base,
# only one base nodule is allowed in db, base module always with last ID-index
mod_i_base = ParametrizedCylMod(header={"moduleName": "I_shape_base", "moduleID": 6}, body_id="cyl_b2",
                                h=1.0, con_id=["con_cyl_b_01", "con_cyl_b_02"], set_base=True)
# db.add(mod_i_base) # set I-shape as base module,
# only one base nodule is allowed in db, base module always with last ID-index
jnt_base = ParametrizedJntMod(header={"moduleName": "joint", "moduleID": 6},
                              body_id=["proximal", "distal"], params=[1.5, 2], con_id=["proximal_con", "distal_con"],
                              joint_id="joint_0", set_base=True, l_shape_jnt=False)
db.add(jnt_base)
db_full.add(jnt_base)  # set joint as base module,
# only one base module is allowed in db, base module always with last ID-index
max_param = db.max_param_num
# --------------------test copying -----------------------------------------------
# mod_i_1 = mod_i.from_parameters(params=2.0, suffix_id=1)
# mod_i_2 = mod_i.from_parameters(params=3.0, suffix_id=2)
# jnt_1 = jnt_I.from_parameters(params=[3,4], suffix_id=1)
# available_con = {c.id: c for c in itertools.chain.from_iterable(b.connectors for b in mod_i.bodies)}
