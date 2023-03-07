#!/usr/bin/env python3
"""
Help function to visualize new point cloud

@author Lukas Rustler
"""
import os.path

import rospy
from sensor_msgs.msg import PointCloud2
import std_msgs
import sensor_msgs.point_cloud2 as pcl2
import numpy as np


def new_pc_pub():
    """
    Visualize new point cloud
    @return:
    @rtype:
    """
    pointcloud_publisher = rospy.Publisher("/new_pc", PointCloud2, queue_size=20)
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        paths = rospy.get_param("/shape_completion/new_pc_path", None)
        if paths is None:
            rospy.sleep(1)
            continue

        points_ = None
        for path_id, path in enumerate(paths):
            if os.path.isfile(path):
                if points_ is None:
                    points_ = np.load(path)
                else:
                    points_ = np.vstack((points_, np.load(path)))

        if points_ is not None:
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'base_link'
            new_pc = pcl2.create_cloud_xyz32(header, points_[:, :3])
            pointcloud_publisher.publish(new_pc)
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node("new_pc_node")
    try:
        new_pc_pub()
    except rospy.ROSInterruptException:
        pass