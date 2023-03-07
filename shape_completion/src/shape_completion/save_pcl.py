#!/usr/bin/env python3
"""
Util to read point cloud and save it

@author Lukas Rustler
"""
import copy
import rospy
from sensor_msgs.msg import PointCloud2
import numpy as np
import os
import tf
import tf.transformations as ts
from shape_completion.subscriber_classes import POINTCLOUD
from shape_completion.srv import smooth_pcl
import open3d as o3d
from yolo.srv import call_yolo
from shape_completion.srv import mujoco_bbox
from rgbd_segmentation.srv import segment


def save_pcl(out_path, camera="table", simulation=False, full_pc=False, pose_estimation=False,
             classes=None, bboxes=None, objects_names=[]):
    """
    Subscribes to topic with segmented point cloud, transforms it into /base_link and saves it into file
    @param out_path: Path to the folder where to save data
    @type out_path: string
    @param camera: name of the camera - table
    @type camera: string
    @param simulation: whether the simulation is used
    @type simulation: bool
    @param full_pc: whether to save the full PC, without segmentation
    @type full_pc: bool
    @param pose_estimation: whether the function is called from pose estimation
    @type pose_estimation: bool
    @param classes: classes for object detection functions
    @type classes: list
    @param bboxes: bonding boxed for detection functions
    @type bboxes: list
    @param objects_names: name of the known objects
    @type objects_names: list
    @return: objects_file_names, new classes, new bounding boxes
    @rtype: list, list, numpy array
    """

    rospy.loginfo("Waiting for segmented point cloud message")

    if not simulation:
        yolo = rospy.ServiceProxy('call_yolo', call_yolo)
    else:
        yolo = rospy.ServiceProxy('classify_mujoco', mujoco_bbox)
    seg = rospy.ServiceProxy('segment_scene', segment)
    smooth_pcl_service = rospy.ServiceProxy('smooth_pcl', smooth_pcl)

    if not full_pc:
        if camera == "table":
            topic = "/table_camera/rgb"
        elif camera == "kinova":
            topic = "/camera_kinova/rgb"

        if not simulation:
            yolo_response = yolo(topic, True)
        else:
            yolo_response = yolo(topic, classes, bboxes)

        classes = yolo_response.classes
        segmentation_response = seg(yolo_response.bboxes)
        points = []
        start_index = 0
        for pcl_id in range(len(segmentation_response.pointclouds_len)):
            pcl_len = segmentation_response.pointclouds_len[pcl_id]
            pcl_temp = np.reshape(segmentation_response.pointclouds[start_index:start_index+pcl_len], (-1, 3))
            colors_temp = np.reshape(segmentation_response.colors[start_index//3:start_index//3+pcl_len//3], (-1, 1))
            points.append(np.hstack((pcl_temp, colors_temp)))
            start_index += pcl_len
    else:
        # Create class handle for subscriber and set right parameters
        class_handle = POINTCLOUD(1)

        if camera == "table":
            class_handle.sub = rospy.Subscriber("/table_camera/pcl", PointCloud2, class_handle)
        elif camera == "kinova":
            class_handle.sub = rospy.Subscriber("/camera_kinova/depth_registered/points", PointCloud2, class_handle)

        while not class_handle.end:
            rospy.sleep(0.01)

        points = [class_handle.points]

    number_of_objects = 1 if full_pc else len(segmentation_response.pointclouds_len)
    objects_file_names = []
    if len(objects_names) == 0:
        objects_names = [obj for obj in rospy.get_param("/object_name", "out").split(",")]
    for obj_id in range(number_of_objects):
        if not full_pc:
            if camera == "table":
                # Run voxel grid filter, moving least squares filter and remove outliers
                try:
                    response = smooth_pcl_service(points[obj_id].ravel(), 0.002, 0.025, "table")
                except rospy.ServiceException as e:
                    print("Service call failed: %s" % e)
                points_temp = np.array(response.points).reshape((-1, 4))
            else:
                # Run voxel grid filter, moving least squares filter and remove outliers
                try:
                    response = smooth_pcl_service(points[obj_id].ravel(), 0.001, 0.025, "arm")
                except rospy.ServiceException as e:
                    print("Service call failed: %s" % e)
                points_temp = np.array(response.points).reshape((-1, 4))
        else:
            rgba = []
            mask = points[obj_id][:, 2] < 1.5
            points_temp = points[obj_id][mask]

        # Get color R,G,B,A colors values from one UINT32 number
        if not full_pc:
            rgba = []
            if not pose_estimation:
                for idx, p in enumerate(points_temp):
                    b = np.uint32(p[3]) >> np.uint32(0) & np.uint32(255)
                    g = np.uint32(p[3]) >> np.uint32(8) & np.uint32(255)
                    r = np.uint32(p[3]) >> np.uint32(16) & np.uint32(255)
                    a = np.uint32(p[3]) >> np.uint32(24) & np.uint32(255)
                    p = r << np.uint32(16) | g << np.uint32(8) | b << np.uint(0)  # a << np.uint32(0) |

                    points_temp[idx, 3] = p
                    rgba.append([r, g, b, a])
                    # print(r, g, b, a)
        rgba = np.array(rgba)
        if full_pc:
            # Transform point cloud to base
            tf_listener = tf.TransformListener()
            while True:
                try:
                    if camera == "table":
                        (trans, rot) = tf_listener.lookupTransform('/base_link', '/virtual_camera', rospy.Time(0))
                    else:
                        (trans, rot) = tf_listener.lookupTransform('/base_link', '/camera_kinova_color_frame', rospy.Time(0))
                    transform = ts.concatenate_matrices(ts.translation_matrix(trans), ts.quaternion_matrix(rot))
                    break
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue
        points_homog = np.hstack((points_temp[:, :3], np.ones((points_temp.shape[0], 1))))

        # There is a fixed rotation between RVIZ and what we want to see
        R = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        if not full_pc:
            transform = R.T
        else:
            # transform = R.T
            transform = np.matmul(R.T, transform)
        points_homog = np.matmul(transform, points_homog.T)
        points_temp[:, :3] = points_homog[:3, :].T

        if not full_pc:
            file_name = objects_names[obj_id]
        else:
            file_name = "full"

        # Save .pcd file and .npz with coordinates and colors
        rospy.loginfo(f"Saving pointcloud {obj_id+1} to file")
        pcl = o3d.geometry.PointCloud()

        if not os.path.exists(out_path.replace("pcd", "npz")):
            os.makedirs(out_path.replace("pcd", "npz"))
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        pcl.points = o3d.utility.Vector3dVector(points_temp[:, :3])
        if rgba.size > 0:
            pcl.colors = o3d.utility.Vector3dVector(rgba[:, :3]/255)

        if not full_pc:
            out_path_temp = os.path.join(out_path, file_name+"__"+str(obj_id)+".pcd")
        else:
            out_path_temp = os.path.join(out_path, file_name + ".pcd")

        if pose_estimation:
            out_path_temp = out_path_temp.replace(".pcd", "_pose.pcd")

        o3d.io.write_point_cloud(out_path_temp, pcl)
        np.savez(out_path_temp.replace("pcd", "npz"), points=points_temp[:, :3], colors=rgba)
        objects_file_names.append(file_name+"__"+str(obj_id))
    return objects_file_names, classes, np.array(yolo_response.bboxes)


if __name__ == "__main__":
    rospy.init_node("pcl_saver", anonymous=True)
    save_pcl("~/test", camera="table", simulation=True, full_pc=True)
