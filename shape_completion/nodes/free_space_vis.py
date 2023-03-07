#!/usr/bin/env python3
"""
Help function to visualize new point cloud

@author Lukas Rustler
"""
import rospy
from sensor_msgs.msg import PointCloud2
import std_msgs
import sensor_msgs.point_cloud2 as pcl2
import numpy as np
from sensor_msgs.msg import PointField


def create_cloud_xyzrgb(header, points):
    """
    Create a L{sensor_msgs.msg.PointCloud2} message with 3 float32 fields (x, y, z) and 1 UINT32 field (rgba).

    :param header: The point cloud header.
    :type  header: L{std_msgs.msg.Header}
    :param points: The point cloud points.
    :type  points: iterable
    :return: The point cloud.
    :rtype:  L{sensor_msgs.msg.PointCloud2}
    """
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('rgba', 16, PointField.UINT32, 1)]

    return pcl2.create_cloud(header, fields, points)


def free_space_pub():
    """
    Visualize new point cloud
    @return:
    @rtype:
    """
    pointcloud_publisher = rospy.Publisher("/free_space", PointCloud2, queue_size=20)
    R = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).T
    rate = rospy.Rate(1)

    # "Blue", "Green", "Red", "Orange", "Yellow"
    colors_rgba = [4278190335, 4278255360, 4294901760, 4294944000, 4294967040]
    while not rospy.is_shutdown():

        objects_file_names = rospy.get_param("/actvh/objects_file_names", None)
        if objects_file_names is None:
            rate.sleep()
            continue
        points = None
        objs = 0
        for obj_id, obj_fname in enumerate(objects_file_names):
            points_ = rospy.get_param("/IGR/free_space/"+obj_fname, [])
            if len(points_) == 0:
                continue
            points_ = np.reshape(points_, (-1, 3))
            colors_ = np.tile(colors_rgba[objs], (points_.shape[0], ))
            if points is None:
                points = points_
                colors = colors_
            else:
                points = np.vstack((points, points_))
                colors = np.hstack((colors, colors_))
            objs += 1

        if points is None:
            rate.sleep()
            continue
        points = np.matmul(R.T, np.hstack((points, np.ones((points.shape[0], 1)))).T)[:3, :].T

        iterable = list(zip(points[:, 0], points[:, 1], points[:, 2], colors))

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'base_link'
        new_pc = create_cloud_xyzrgb(header, iterable)
        pointcloud_publisher.publish(new_pc)
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node("free_space_node")
    try:
        free_space_pub()
    except rospy.ROSInterruptException:
        pass