#!/usr/bin/env python3
"""
Service file to segment objects from scene

@author Lukas Rustler
"""
import rospy
import std_msgs
import sensor_msgs.point_cloud2 as pcl2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import PointField
import tf.transformations as ts
import tf
from rgbd_segmentation.srv import segment, segmentResponse, segmentRequest
from shape_completion.srv import smooth_pcl
from sensor_msgs.msg import PointCloud2
import os
import cv2


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


class Generator:

    def __init__(self):
        """
        Init function for the class
        """
        self.rgb_topic = rospy.get_param("/rgbd_segmentation/rgb_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("/rgbd_segmentation/depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.camera_info_topic = rospy.get_param("/rgbd_segmentation/depth_camera_info_topic", "/camera/aligned_depth_to_color/camera_info")
        self.base_link = rospy.get_param("/rgbd_segmentation/base_link", "base_link")
        self.rate = rospy.Rate(rospy.get_param("/rgbd_segmentation/rate", 1))
        self.publish_point_cloud = rospy.get_param("/rgbd_segmentation/publish_point_cloud", True)
        self.point_cloud_publisher = rospy.Publisher("/rgbd_segmentation/point_cloud_internal", PointCloud2, queue_size=20)
        self.debug = rospy.get_param("/rgbd_segmentation/debug", False)
        self.floodfill_change = rospy.get_param("/rgbd_segmentation/floodfill_change", 0.002)
        self.units = rospy.get_param("/rgbd_segmentation/units", 1000)
        self.rgb_info_topic = rospy.get_param("/rgbd_segmentation/rgb_camera_info_topic", "/camera/color/camera_info")
        self.do_alignment = rospy.get_param("/rgbd_segmentation/align_depth", False)
        self.smooth = rospy.ServiceProxy('smooth_pcl', smooth_pcl)
        self.bridge = CvBridge()
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.camera_R = None
        self.RVIZ_R = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    def __call__(self, bboxes):
        """
        Main callback
        :param bboxes: message with bboxes
        :type bboxes: generate_mesh
        :return: 0 for success
        :rtype: int
        """

        rospy.loginfo("Got request to segment")
        if isinstance(bboxes, segmentRequest):
            bboxes = bboxes.bboxes

        # get camera info
        cam_info = rospy.wait_for_message(self.camera_info_topic, CameraInfo)

        # get depth
        if not self.do_alignment:
            msg = rospy.wait_for_message(self.depth_topic, Image)
            depth = self.bridge.imgmsg_to_cv2(msg)/self.units
        else:
            depth = self.align_depth()
        depth = depth.astype(np.float32)

        if self.debug:
            cv2.imshow("Depth", depth)
            while True:
                cv2.waitKey(100)
                if cv2.getWindowProperty('Depth', cv2.WND_PROP_VISIBLE) < 1:
                    break
            cv2.destroyWindow("Depth")

        # get rgb
        msg = rospy.wait_for_message(self.rgb_topic, Image)
        rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # get camera info
        fx, fy, cx, cy = cam_info.K[0], cam_info.K[4], cam_info.K[2], cam_info.K[5]

        # get transformation to base frame
        translation, rotation = self.get_transformation(cam_info.header.frame_id, self.base_link)
        self.camera_R = ts.quaternion_matrix(rotation)
        self.camera_R[:3, 3] = translation

        points_pc = []
        colors = []

        pointcloud_len = []
        colors_len = []
        # for each bounding box
        for bbox_id in range(len(bboxes) // 4):
            bbox = bboxes[bbox_id * 4:(bbox_id + 1) * 4]

            #add mask from bounding box
            mask = np.zeros((depth.shape[0] + 2, depth.shape[1] + 2), np.uint8)
            mask[bbox[1], bbox[0]:bbox[2]] = 1
            mask[bbox[3], bbox[0]:bbox[2]] = 1
            mask[bbox[1]:bbox[3], bbox[0]] = 1
            mask[bbox[1]:bbox[3], bbox[2]] = 1

            start_pixel = ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2)

            num_tries = 25

            for ff_try in range(num_tries):
                # floodfill object with similar colors
                _, floodfilled, _, _ = cv2.floodFill(depth.copy()+0.1, mask, start_pixel, 0, self.floodfill_change, self.floodfill_change)
                num_pixels = np.count_nonzero(floodfilled == 0)
                if num_pixels < ((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))*0.1:
                    r = np.random.randint(-10, 11, (2, ))
                    start_pixel = (start_pixel[0]+r[0], start_pixel[1]+r[0])
                else:
                    break

            if num_pixels < ((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))*0.1:
                # if too few point found with the right color, use all point in the bbox
                rospy.loginfo("Floodfilling didnt work, using fall back method")
                depth = depth.T
                xx, yy = np.arange(bbox[0], bbox[2]), np.arange(bbox[1], bbox[3])
                u, v = np.meshgrid(xx, yy)
                pixels = np.hstack((u.flatten().reshape(-1, 1), v.flatten().reshape(-1, 1)))

                depth_box = depth[tuple(pixels.T)]
                if self.debug:
                    floodfilled = depth.copy()+0.1
                    floodfilled[tuple(pixels.T)] = 0
                    floodfilled = floodfilled.T
                depth = depth.T
            else:
                pixels = np.nonzero(floodfilled == 0)
                pixels = np.hstack((pixels[1].reshape(-1, 1), pixels[0].reshape(-1, 1)))
                depth_box = depth.T[tuple(pixels.T)]

            if self.debug:
                # cv2 window with segmented depth
                floodfilled[floodfilled != 0] = 255
                cv2.imshow("Segmentation", floodfilled)
                while True:
                    cv2.waitKey(100)
                    if cv2.getWindowProperty('Segmentation', cv2.WND_PROP_VISIBLE) < 1:
                        break
                cv2.destroyWindow("Segmentation")

            # call RGB-D to pcl function
            x_local, y_local, z_local = self.rgbd_to_pcl(pixels, depth_box, [fx, fy, cx, cy], cam_info.distortion_model, cam_info.D)

            # Transform back to base_link to filter out points base on coordinates
            points_in_base = (self.camera_R @ np.vstack((x_local, y_local, z_local, np.ones(len(x_local)))))[:3, :].T

            # get rid of points with "distance from base"-based segmentation
            mask = np.ones((points_in_base.shape[0],), dtype=bool)
            x_seg = rospy.get_param("/rgbd_segmentation/segmentation_x", -1)
            y_seg = rospy.get_param("/rgbd_segmentation/segmentation_y", -1)
            z_seg = rospy.get_param("/rgbd_segmentation/segmentation_z", -1)

            if x_seg != -1:
                if "[" in x_seg:
                    x_seg = eval(x_seg)
                    mask_temp = np.logical_and(points_in_base[:, 0] > x_seg[0], points_in_base[:, 0] < x_seg[1])
                    mask = np.logical_and(mask, mask_temp)
            if y_seg != -1:
                if "[" in y_seg:
                    y_seg = eval(y_seg)
                    mask_temp = np.logical_and(points_in_base[:, 1] > y_seg[0], points_in_base[:, 1] < y_seg[1])
                    mask = np.logical_and(mask, mask_temp)
            if z_seg != -1:
                if "[" in z_seg:
                    z_seg = eval(z_seg)
                    mask_temp = np.logical_and(points_in_base[:, 2] > z_seg[0], points_in_base[:, 2] < z_seg[1])
                    mask = np.logical_and(mask, mask_temp)

            # Get colors in RGB format, switch from BGR to RGB and transform to one number
            colors_local_rgb = rgb[tuple(pixels[:, [1, 0]].T)].astype(np.uint32)
            colors_local = (np.uint32(255) << np.uint32(24) | colors_local_rgb[:, 2] << np.uint32(16) |
                            colors_local_rgb[:, 1] << np.uint32(8) | colors_local_rgb[:, 0] << np.uint32(0))

            points_in_base = points_in_base[mask, :]
            colors_local = colors_local[mask]

            if bbox_id == 0:
                points_pc, colors = points_in_base, colors_local
            else:
                points_pc = np.vstack((points_pc, points_in_base))
                colors = np.hstack((colors, colors_local))

            pointcloud_len.append(points_in_base.size)
            colors_len.append(colors_local.size)

        if self.publish_point_cloud and len(points_pc) > 0 and len(colors) > 0:
            # Generate one point cloud which will be used as base with the publisher
            self.generate_point_cloud(list(zip(points_pc[:, 0], points_pc[:, 1], points_pc[:, 2], colors)))

        out = segmentResponse()
        out.pointclouds = points_pc.ravel()
        out.pointclouds_len = pointcloud_len
        out.colors = colors
        out.colors_len = colors_len

        return out

    def generate_point_cloud(self, iterable):
        """
        Function to publish internal point cloud which can be used by point cloud publisher node
        :param iterable: list of x, y, z and colors
        :type iterable: list
        :return:
        :rtype:
        """
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.base_link
        pc = create_cloud_xyzrgb(header, iterable)
        self.point_cloud_publisher.publish(pc)

    @staticmethod
    def rgbd_to_pcl(pixels, depth, intrinsic, dist_model="", dist_coefs=[]):
        """
        Function to generate point cloud given RGB-D information and camera infromation.
        Based on https://github.com/IntelRealSense/librealsense/blob/2decb32456fc68396da2bc4de25028b1ee2d735f/include/librealsense2/rsutil.h#L67

        :param pixels: Nx2 array of pixels in the image, where to compute 3D point
        :type pixels: numpy.array
        :param depth: Nx1 array of depth for given pixels
        :type depth: numpy.array
        :param intrinsic: list of intrinsic parameters [fx, fy, cx, cy]
        :type intrinsic: list/numpy.array
        :param dist_model: optional; name of the distortion model
        :type dist_model: string
        :param dist_coefs: optional; list of distortion parameters
        :type dist_coefs: list
        :return: x, y, z coordinates of the output point cloud
        :rtype: numpy.array
        """

        # FLT_EPSILON = 1e-5
        fx, fy, cx, cy = intrinsic
        x = (pixels[:, 0] - cx) / fx
        y = (pixels[:, 1] - cy) / fy

        if dist_model == "mujoco":
            depth *= dist_coefs[0]

        return depth * x, depth * y, depth

    @staticmethod
    def get_transformation(what, where):
        """
        Help util to get transformation
        :param what: For what frame to obtain the transformation
        :type what: string
        :param where: In which frame to express
        :type where: string
        :return:
        :rtype:
        """
        tf_listener = tf.TransformListener()
        i = 0
        while True:
            rospy.sleep(0.01)
            try:
                translation, rotation = tf_listener.lookupTransform(where, what, rospy.Time(0))
                return np.array(translation), np.array(rotation)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                i += 1
                if i > 1e2:
                    return None, None
                continue

    def align_depth(self):

        msg = rospy.wait_for_message(self.depth_topic, Image)
        depth = self.bridge.imgmsg_to_cv2(msg) / self.units

        cam_info_rgb = rospy.wait_for_message(self.rgb_info_topic, CameraInfo)
        cam_info_depth = rospy.wait_for_message(self.camera_info_topic, CameraInfo)

        int_rgb = np.reshape(cam_info_rgb.K, (3, 3))
        int_depth = np.reshape(cam_info_depth.K, (3, 3))

        tr, rot = self.get_transformation(cam_info_depth.header.frame_id, cam_info_rgb.header.frame_id)
        extrinsics = ts.concatenate_matrices(ts.translation_matrix(tr), ts.quaternion_matrix(rot))

        return cv2.rgbd.registerDepth(int_depth, int_rgb, None, extrinsics, depth,
                                      (cam_info_rgb.width, cam_info_rgb.height), depthDilation=True)


if __name__ == "__main__":
    rospy.init_node("segmentation_node")

    generator = Generator()

    rospy.Service('segment_scene', segment, generator)
    rospy.loginfo("Mesh generator service successfully run")
    rospy.spin()
