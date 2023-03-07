#!/usr/bin/env python3

import rospy
from mujoco_interface_msgs.srv import GetAllObjectPoses, GetAllObjectPosesRequest
import tf


def send_tf():
    # pull state from mujoco
    rate = rospy.Rate(10)
    get_all_poses = rospy.ServiceProxy('/kinova_mujoco/getAllObjectPoses', GetAllObjectPoses)
    br = tf.TransformBroadcaster()

    while not rospy.is_shutdown():
        res = get_all_poses.call(GetAllObjectPosesRequest())
        for name, pose in zip(res.names, res.poses):
            pose = pose.pose
            trans = [pose.position.x, pose.position.y, pose.position.z]
            rot = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            br.sendTransform(trans, rot, rospy.Time.now(), name, 'base_link')
        rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node('mujoco_tf_broadcaster')
        send_tf()
    except rospy.ROSInterruptException:
        pass