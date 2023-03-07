"""
File with various subscriber classes allowing easier manipulation

@author Lukas Rustler
"""
import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import sensor_msgs.point_cloud2 as pc2
import PyKDL as kdl
from kdl_parser_py import urdf as kdl_parser


class CUSUM:
    """
    Cumulative sum detection from torques
    """
    def __init__(self, threshold, drift, joints_idxs):
        self.threshold = threshold
        self.drift = np.tile(drift, (1, 7))
        self.data_last = None
        self.g_m_last = np.zeros((1, 7))
        self.g_p_last = np.zeros((1, 7))
        self.zeros_ = np.zeros((1, 7))
        self.impact_pose = None
        self.sub = None
        self.joint_idxs = joints_idxs
        self.trajectory_event_publisher = None
        self.drift_compute = np.zeros((1, 7))
        self.step = 0
        self.s_last = 0

    def __call__(self, data):
        """
        Computes cumulative sum from joint effort data
        @return:
        @rtype:
        """

        if self.data_last is None:
            self.data_last = data
            return 0
        s = np.array(data.effort)[self.joint_idxs] - np.array(self.data_last.effort)[self.joint_idxs]

        if self.step < 50:
            self.drift_compute += np.abs(s)
        elif self.step == 50:
            self.drift = self.drift_compute/50
            self.drift *= 1.25
        g_p = np.maximum(self.g_p_last + s - self.drift, self.zeros_)
        g_m = np.maximum(self.g_m_last - s - self.drift, self.zeros_)

        gpt = np.count_nonzero(g_p > self.threshold)
        gmt = np.count_nonzero(g_m > self.threshold)
        if self.step >= 85 and (gpt + gmt >= 1):
            e = String()
            e.data = "stop"
            self.trajectory_event_publisher.publish(e)
            self.sub.unregister()
            self.impact_pose = np.array(data.position)
            return 0

        if self.step >= 50:
            self.g_p_last = g_p
            self.g_m_last = g_m
        self.step += 1
        self.data_last = data

        return 0


class THRESHOLD:
    """
    Pure threshold detection from joint torques
    """
    def __init__(self, threshold, joints_idxs):
        self.threshold = threshold
        self.impact_pose = None
        self.sub = None
        self.joint_idxs = joints_idxs
        self.step = 0

    def __call__(self, data):
        """
        Check for torques threshold from external torques given by robot/mujoco
        @return:
        @rtype:
        """

        self.step += 1
        if self.step > 50 and np.any(np.abs(data.effort)[self.joint_idxs] > self.threshold):
            e = String()
            e.data = "stop"
            self.trajectory_event_publisher.publish(e)
            self.sub.unregister()
            self.impact_pose = np.array(data.position)

        return 0


class KDL_PUBLISHER:
    """
    Takes values from robot joint torques and from kdl and publish directly the external torques
    """

    def __init__(self, joint_idxs, end_effector):
        self.sub = None
        self.vel_last = None
        self.pos_last = None
        self.time_last = None
        self.pub = None
        self.joint_idxs = joint_idxs
        self.torques = kdl.JntArray(7)
        _, self.kdl_tree = kdl_parser.treeFromParam("robot_description")
        self.chain = self.kdl_tree.getChain("base_link", end_effector)
        self.id_solver = kdl.ChainIdSolver_RNE(self.chain, kdl.Vector(0, 0, -9.81))

    def kdl_to_np(self, ar):
        python_ar = np.zeros(ar.rows())
        for idx, _ in enumerate(ar):
            python_ar[idx] = _
        return python_ar

    def np_to_kdl(self, ar):
        kdl_ar = kdl.JntArray(len(ar))
        for idx, _ in enumerate(ar):
            kdl_ar[idx] = _
        return kdl_ar

    def __call__(self, data):
        """
        Check for torques threshold from external torques computed with openrave
        @return:
        @rtype:
        """

        if self.vel_last is None:
            self.pos_last = np.array(data.position)[self.joint_idxs]
            self.vel_last = np.array(data.velocity)[self.joint_idxs]
            self.time_last = data.header.stamp.to_sec()
            return 0

        pos = np.array(data.position)[self.joint_idxs]

        time_curr = data.header.stamp.to_sec()
        vel = np.array(data.velocity)[self.joint_idxs]
        acc = (vel - self.vel_last) / float((time_curr - self.time_last) * 1e3)

        self.id_solver.CartToJnt(self.np_to_kdl(pos), self.np_to_kdl(vel), self.np_to_kdl(acc),
                                 [kdl.Wrench() for _ in range(self.chain.getNrOfSegments())], self.torques)

        ext_effort = self.kdl_to_np(self.torques) - np.array(data.effort)[self.joint_idxs]

        effort = np.zeros(len(data.position))
        effort[self.joint_idxs] = ext_effort

        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.position = np.array(data.position)
        msg.velocity = np.array(data.velocity)
        msg.effort = effort
        msg.name = data.name
        self.pub.publish(msg)

        self.pos_last = pos
        self.vel_last = vel
        self.time_last = time_curr
        return 0


class POINTCLOUD:
    """
    Handle to obtain pointclouds
    """
    def __init__(self, num_messages):
        self.sub = None
        self.points = np.array([])
        self.is_bigendian = None
        self.received = 0
        self.end = False
        self.num_messages = num_messages

    def __call__(self, msg):
        """
        Stack point until required number of messages is acquired
        @return:
        @rtype:
        """

        if self.received >= self.num_messages:
            self.points = np.array(self.points)
            self.end = True
            self.sub.unregister()
            return 0
        if self.is_bigendian is None:
            self.is_bigendian = msg.is_bigendian
        if len(msg.fields) > 3:
            msg.fields[-1].datatype = 6

        if len(self.points) == 0:
            self.points = list(pc2.read_points(msg))
        else:
            for _ in pc2.read_points(msg):
                self.points.append(_)
        self.received += 1
        return 0


class GRIPPER_FEEDBACK:
    """
    Handler for feedback from the gripper. Stops the gripper when current limit is exceeded
    """
    def __init__(self, threshold):
        self.threshold = threshold
        self.sub = None
        self.detect = False
        self.send_speed = None
        self.num_samples = 0

    def __call__(self, msg):
        if self.detect:
            motor = msg.interconnect.oneof_tool_feedback.gripper_feedback[0].motor[0]
            self.num_samples += 1
            if self.num_samples >= 10 and motor.current_motor > self.threshold:
                self.send_speed(0.0)
                self.detect = False
                self.num_samples = 0
                #self.sub.unregister()

