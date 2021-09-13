#! /usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()



from gym import utils
import math
import rospy
from src.moveo_training.src.goal_position import Obj_Pos
from gym import spaces
from src.moveo_training.src import moveo_env
from gym.envs.registration import register
import numpy as np

max_episode_steps = 100 # Can be any Value

register(
        id='MoveoIK-v0',
        entry_point='src.moveo_training.src.moveo_ik_env:MoveoIKEnv',
        max_episode_steps=max_episode_steps,
    )


class MoveoIKEnv(moveo_env.MoveoEnv, utils.EzPickle):
    def __init__(self):
    
        self.goal_pose = Obj_Pos(object_name="goalPoint")
       
        moveo_env.MoveoEnv.__init__(self)
        utils.EzPickle.__init__(self)
        self.get_params()
        self.gazebo.unpauseSim()

        self.action_space = spaces.Box(
            low=self.joint_min_angle,
            high=self.joint_max_angle, shape=(self.number_of_actions,),
            dtype=np.float32
        )
        
        max_possible_distance_to_goal = np.array([2])
        min_possible_distance_to_goal = np.array([0])
        max_possible_position_x =np.array([1.0])
        max_possible_position_y =np.array([1.0])
        max_possible_position_z =np.array([1.0])
        min_possible_position_x =np.array([-1.0])
        min_possible_position_y =np.array([-1.0])
        min_possible_position_z =np.array([-1.0])

        high = np.concatenate([ max_possible_distance_to_goal,max_possible_position_x,max_possible_position_y,max_possible_position_z])
        low = np.concatenate([min_possible_distance_to_goal, min_possible_position_x,min_possible_position_y,min_possible_position_z])

        self.observation_space = spaces.Box(low, high)

    def get_params(self):
        self.sim_time = rospy.get_time()
        self.number_of_actions = 5
        self.number_of_observations = 3
        self.joint_max_angle = 2.356
        self.joint_min_angle = -2.356
        self.init_pos ={}
        for key in self.joint_names:
            self.init_pos[key]= 0.0
        # self.init_pos = {
        #         "Joint_1": 0.0,
        #         "Joint_2": 0.0,
        #         "Joint_3": 0.0,
        #         "Joint_4": 0.0,
        #         "Joint_5": 0.0,
        #         }
        
        self.setup_ee_pos = {"x": 0,
                            "y": 0,
                            "z": 0}
        self.impossible_movement_punishement = -500
        self.max_distance_to_Goal= 0.05
   

    def _set_init_pose(self):
    
        self.gazebo.unpauseSim()
        if not self.set_trajectory_joints(self.init_pos):
            assert False, "Initialisation is failed...."

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        :return:
        """
        # rospy.logdebug("Init Env Variables...")
        # rospy.logdebug("Init Env Variables...END")

    def _set_action(self, action):
        self.new_pos = {}
        for index,key in enumerate(self.joint_names):
            self.new_pos[key] = float(action[index])
        self.movement_result = self.set_trajectory_joints(self.new_pos)


    def _get_obs(self):
        self.gazebo.unpauseSim()
        grip_pose = self.get_ee_pose()
        ee_array_pose = [grip_pose.position.x, grip_pose.position.y, grip_pose.position.z]
        # the pose of the goal
        goal_data = self.goal_pose.get_states()
        # position Goal
        object_pos = goal_data[:3]
        distance_from_goal = self.calc_dist(object_pos,ee_array_pose)

        observations_obj = np.array([distance_from_goal,object_pos[0],object_pos[1],object_pos[2]])
        return  observations_obj
    
    def calc_dist(self,p1,p2):
        """
        d = ((2 - 1)2 + (1 - 1)2 + (2 - 0)2)1/2
        """
       
        x_d = math.pow(p1[0] - p2[0],2)
        y_d = math.pow(p1[1] - p2[1],2)
        z_d = math.pow(p1[2] - p2[2],2)
        d = math.sqrt(x_d + y_d + z_d)

        return d

    
    def get_elapsed_time(self):
        """
        Returns the elapsed time since the beginning of the simulation
        Then maintains the current time as "previous time" to calculate the elapsed time again
        """
        current_time = rospy.get_time()
        dt = self.sim_time - current_time
        self.sim_time = current_time
        return dt

    def _is_done(self, observations):
        """
        If the latest Action didnt succeed, it means that tha position asked was imposible therefore the episode must end.
        It will also end if it reaches its goal.
        """

       
        # distance = observations[0]
        distance = observations[0]
        # Did the movement fail in set action?
        done_fail = not(self.movement_result)

        done_sucess = distance <= self.max_distance_to_Goal

        # print(">>>>>>>>>>>>>>>>done_fail="+str(done_fail)+",done_sucess="+str(done_sucess))
        # If it moved or the arm couldnt reach a position asced for it stops
        done = done_fail or done_sucess
        
        return done

    def _compute_reward(self, observations, done):
        """
        Reward moving the cube
        Punish movint to unreachable positions
        Calculate the reward: binary => 1 for success, 0 for failure
        """
        distance =  observations[0]


        # Did the movement fail in set action?
        done_fail = not(self.movement_result)
        
        done_sucess = distance <= self.max_distance_to_Goal
        # print("Distanz zum Ziel= "+ str(distance))
        if done_fail:
            
            # We punish that it tries sto move where moveit cant reach
            reward = self.impossible_movement_punishement

            # print("Bestrafung für unmögliche Kombination an Gelenkwinkel! Reward= ", reward)
        else:
            if done_sucess:
                #It reached the goal
                reward = -1*self.impossible_movement_punishement

                # print("Ziel wurde erreicht Reward= ", reward)
            else:
            
                reward =-distance
                # print("Ziel wurde nicht erreicht, Reward= ", reward)
        return reward