#!/usr/bin/env python3
"""
Help function to visualize new point cloud

@author Lukas Rustler
"""
import rospy
from sensor_msgs.msg import PointCloud2


class Publisher:

    def __init__(self):
        self.subscriber = rospy.Subscriber("/rgbd_segmentation/point_cloud_internal", PointCloud2, self)
        self.rate = rospy.Rate(rospy.get_param("/rgbd_segmentation/rate", 1))
        self.publisher = rospy.Publisher("/rgbd_segmentation/point_cloud", PointCloud2, queue_size=20)
        self.pcl = None
        rospy.loginfo("Mesh generation point cloud publisher run")

    def run(self):
        """
        Visualize new point cloud
        :return:
        :rtype:
        """

        while not rospy.is_shutdown():
            if self.pcl is None:
                self.rate.sleep()
                continue

            self.publisher.publish(self.pcl)
            self.rate.sleep()

        return 0

    def __call__(self, msg):
        self.pcl = msg


if __name__ == "__main__":
    rospy.init_node("point_cloud_publisher")
    publisher = Publisher()
    publisher.run()
