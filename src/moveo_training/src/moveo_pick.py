#! /usr/bin/env python

from re import S
from gym import utils
import math
import rospy
from src.moveo_training.src.goal_position import Obj_Pos
from gym import spaces
from src.moveo_training.src.moveo_env import MoveoEnv
from gym.envs.registration import register
import numpy as np
import moveit_commander
from geometry_msgs.msg import PoseStamped

max_episode_steps = 100 # Can be any Value

register(
        id='MoveoPick-v0',
        entry_point='src.moveo_training.src.moveo_pick:MoveoPickEnv',
        max_episode_steps=max_episode_steps,
    )


class MoveoPickEnv(MoveoEnv, utils.EzPickle):
    def __init__(self):
        scene = moveit_commander.PlanningSceneInterface()
        robot = moveit_commander.RobotCommander()

        rospy.sleep(2)

        p = PoseStamped()
        p.header.frame_id = robot.get_planning_frame()
        p.pose.position.x = 0.
        p.pose.position.y = 0.
        p.pose.position.z = -0.001
        scene.add_box("table", p, (1.5, 1.5, 0))
        print ("Entered Push Env")
        self.obj_positions = Obj_Pos(object_name="cube")

        self.get_params()

        MoveoEnv.__init__(self)
        utils.EzPickle.__init__(self)

        self.gazebo.unpauseSim()

        # self.action_space = spaces.Discrete(self.n_actions)
        self.action_space = spaces.Box(
            low=self.position_joints_min,
            high=self.position_joints_max, shape=(self.n_actions,),
            dtype=np.float32
        )

        observations_high_dist = np.array([self.max_distance])
        observations_low_dist = np.array([0.0])

        observations_high_speed = np.array([self.max_speed])
        observations_low_speed = np.array([0.0])

        observations_ee_z_max = np.array([self.ee_z_max])
        observations_ee_z_min = np.array([self.ee_z_min])

        high = np.concatenate([observations_high_dist, observations_high_speed, observations_ee_z_max])
        low = np.concatenate([observations_low_dist, observations_low_speed, observations_ee_z_min])

        self.observation_space = spaces.Box(low, high)

        obs = self._get_obs()
        

    def get_params(self):
        """
        get configuration parameters

        """
        self.sim_time = rospy.get_time()
        self.n_actions = 6
        self.n_observations = 3
        self.position_ee_max = 10.0
        self.position_ee_min = -10.0
        self.position_joints_max = 3.14
        self.position_joints_min = -3.14

        self.init_pos = {"Joint_1": 0.0,
                "Joint_2": 0.0,
                "Joint_3": 0.0,
                "Joint_4": 0.0,
                "Joint_5": 0.0,
                "Joint_Servo_Arm_Gear": 0.0
                }
        
        self.setup_ee_pos = {"x": 0.598,
                            "y": 0.005,
                            "z": 0.9}


        self.position_delta = 0.1
        self.step_punishment = -1
        self.closer_reward = 10
        self.impossible_movement_punishement = -100
        self.reached_goal_reward = 100

        self.max_distance = 3.0
        self.max_speed = 1.0
        self.ee_z_max = 1.0
        # Normal z pos of cube minus its height/2
        self.ee_z_min = 0.3

        self.steps_in_current_episode =0


    def _set_init_pose(self):
        """
        Sets the Robot in its init pose
        The Simulation will be unpaused for this purpose.
        """
        self.gazebo.unpauseSim()
        if not self.set_trajectory_joints(self.init_pos):
            assert False, "Initialisation is failed...."

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        :return:
        """
        rospy.logdebug("Init Env Variables...")
        rospy.logdebug("Init Env Variables...END")

    def _set_action(self, action):
        self.new_pos = {"Joint_1": float(action[0]),
                "Joint_2": float(action[1]),
                "Joint_3": float(action[2]),
                "Joint_4": float(action[3]),
                "Joint_5": float(action[4]),
                }
        
        self.new_gripangle = float(action[5]),
        print(self.new_gripangle)
        self.joint_result = self.set_trajectory_joints(self.new_pos)
        self.gripper_result = self.set_gripper(self.new_gripangle)
        self.movement_result= self.joint_result and self.gripper_result
    def _get_obs(self):
        """
        It returns the Position of the TCP/EndEffector as observation.
        And the speed of cube
        Orientation for the moment is not considered
        """
        self.gazebo.unpauseSim()

        grip_pose = self.get_ee_pose()
        ee_array_pose = [grip_pose.position.x, grip_pose.position.y, grip_pose.position.z]

        # the pose of the cube/box on a table        
        object_data = self.obj_positions.get_states()

        # speed cube
        object_pos = object_data[3:]

        distance_from_cube = self.calc_dist(object_pos,ee_array_pose)


        cube_height = object_data[2]
        # speed = np.linalg.norm(object_velp)

        # We state as observations the distance form cube, the speed of cube and the z postion of the end effectors
        observations_obj = np.array([distance_from_cube,
                             cube_height, ee_array_pose[2]])

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

        distance = observations[0]
        cube_height = observations[1]

        # Did the movement fail in set action?
        done_fail = not(self.movement_result)

        done_sucess = cube_height >= 0.5

        print(">>>>>>>>>>>>>>>>done_fail="+str(done_fail)+",done_sucess="+str(done_sucess))
        # If it moved or the arm couldnt reach a position asced for it stops
        done = done_fail or done_sucess

        return done

    def _compute_reward(self, observations, done):
        """
        Reward moving the cube
        Punish movint to unreachable positions
        Calculate the reward: binary => 1 for success, 0 for failure
        """
        distance = observations[0]
        cube_height = observations[1]
        ee_z_pos = observations[2]

        # Did the movement fail in set action?
        done_fail = not(self.movement_result)

        done_sucess = cube_height >= self.max_speed

        if done_fail:
            # We punish that it trie sto move where moveit cant reach
            reward = self.impossible_movement_punishement
        else:
            if done_sucess:
                #It moved the cube
                reward = -1*self.impossible_movement_punishement
            else:
                # It didnt move the cube. We reward it by getting closser
                print("Reward for getting closser") 
                self.steps_in_current_episode+=1
                reward = -distance +cube_height*100

        return reward