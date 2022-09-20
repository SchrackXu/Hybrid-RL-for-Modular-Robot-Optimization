*Hybrid RL for modular robot*

This software pack contains an implementation of reinforcement learning algorithms into a self-developed environment for modular robots. 


Code for modular robot building is developed based on the work of cRoK benchmark and mcs repository from Matthias Mayer, Jonathan Kulz, and the others. [mcs](https://gitlab.lrz.de/cps-robotics)


Code for agents network and algorithms structure is mainly developed based on work of Multi-pass q-networks for deep reinforcement learning with parameterised action spaces from Craig J Bester and the others. [MP-DQN](https://github.com/cycraig/MP-DQN) 


# Code Structure
The code is made up of 6 parts:

**Environment**
Environment: In which the environment of reinforcement learning is developed. Use module_assemble.py to transfer action and state signal of reinforcement learning to robot-related info.
robot_assemble: In which a robot is built and calculated if it can reach the given task or not, given module type index and parameters. 
space_with_obs: generate a scienario with obstacles that is used in environment for experiments

**Agents**
Mostly the same as in the original open source code of MP-DQN work, including P-DQN, MP-DQN, SP-DQN, PA-DDPG etc. 

**common**
This contains wrappers for flatten action & state spaces and wrappers for timing etc.

**Training and testing process (run-files)**
run: run-file for P-DQN, MP-DQN, SP-DQN algorithms, including the interface between the environment and the agent. The training parameters and agent type and parameters are set manually via Click package. Graph visualizations are also implemented in this file for both matplotlib and Tensorboard.
random_run: run-file for an agent with random policy. Used for getting a lower bound of an environment.
run_ddpg: run-file for PA-DDPG algorithm. Different interface to agent than the P-DQN family because of the different algorithm structure.

**parametrized_module**
Parametrized_module: classes for generating atomic module and module database, based on the module file of mcs. Only implement the required shapes for experiments.
Paramstrized_mod_gen: generate a module database instance for all the experiments.

**demo**
demo_environ: check whether environment is set up proper
demo_rob_ass: check whether the robot_assemble file is set up proper
mod_db_test: check whether the parametrized process is done proper
ik_check: check if the scenario can use ik or ik_scipy

# Dependencies
Gym 0.10.5
Tensorboard
Click
and all other dependencies for mcs repository

#Quick guide
put the folder mpdqn under mcs/mcs 
Run test demos to check how environment is settled and how parametrized module is established. 
Run run_fuile to check to training/evaluating result of project experiments. 
Change parameters in click part of run_file to change network structure, training hyperparameter etc..