#!/usr/bin/env python3
"""
Help function to visualize mesh in RVIZ

@author Lukas Rustler
"""
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


def show_reconstruction():
    topic = 'reconstruction_mesh'
    publisher = rospy.Publisher(topic, MarkerArray, queue_size=20)
    rate = rospy.Rate(1)

    id = 0
    marray = MarkerArray()
    while not rospy.is_shutdown():
        mesh_path = rospy.get_param("/shape_completion/rec_mesh_path", False)
        if not mesh_path:
            rospy.sleep(1)
            continue

        marray.markers.clear()
        marker = Marker()
        marker.id = 0
        marker.ns = 'mesh_marker'
        marker.action = Marker.DELETEALL
        marray.markers.append(marker)
        publisher.publish(marray)

        marray.markers.clear()

        for m_id, m in enumerate(mesh_path):
            position = [0, 0, 0]
            marker = Marker()
            marker.id = m_id
            id += 1
            marker.ns = 'mesh_marker'
            marker.header.frame_id = "base_link"

            marker.action = marker.ADD
            marker.scale.x = 1
            marker.scale.y = 1
            marker.scale.z = 1
            marker.color.a = 0.8
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            marker.type = marker.MESH_RESOURCE
            marker.mesh_resource = m
            marker.pose.position = Point(*position)
            marker.pose.orientation.w = 1
            marray.markers.append(marker)

        publisher.publish(marray)
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node("reconstruction_shower")
    try:
        show_reconstruction()
    except rospy.ROSInterruptException:
        pass
