#!/usr/bin/env python3
"""
Super simple util to find bboxes of objects of interest in the mujoco simulation

@author Lukas Rustler
"""

import rospy
import cv2
from cv_bridge import CvBridge
from shape_completion.srv import mujoco_bbox, mujoco_bboxResponse
from sensor_msgs.msg import Image
import numpy as np


class Classificator:
    def __init__(self):
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("/yolo", Image, queue_size=20)

    def __call__(self, request):
        """
        Function to be done when service is called
        @param request: request for segmentation
        @type request: shape_completion.srv.mujoco_bboxRequest
        @return: bounding boxes and corresponding classes
        @rtype: shape_completion.srv.mujoco_bboxResponse
        """
        # get the image
        img = self.load_topic(request.topic)
        original = img.copy()

        # If not class included, use all of them
        if len(request.classes) == 0:
            classes = ["Blue", "Green", "Red1", "Red2", "Orange", "Yellow"]
            bboxes_crop = [None for _ in range(len(classes))]
        else:  # else only the needed classes and transform bboxes
            classes = request.classes
            bboxes_crop = []
            for bbox_id in range(len(request.bboxes) // 4):
                bboxes_crop.append(request.bboxes[bbox_id * 4:(bbox_id + 1) * 4])

        # Thresholds for each color for color based segmentation
        thresholds = {"Blue": [[128, 255, 255], [90, 10, 10]],
                      "Green": [[89, 255, 255], [36, 10, 10]],
                      "Red1": [[180, 255, 255], [159, 10, 10]],
                      "Red2": [[9, 255, 255], [0, 10, 10]],
                      "Orange": [[24, 255, 255], [10, 10, 10]],
                      "Yellow": [[35, 255, 255], [25, 10, 10]]}

        bboxes = []
        colors = []

        # transform to HSV for easier thresholding
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        for color, bbox_crop in zip(classes, bboxes_crop):
            th = thresholds[color]
            # get the mask and mask the image
            mask = cv2.inRange(hsv, np.array(th[1]), np.array(th[0])).astype(bool)

            if bbox_crop is not None: # if only provided region needs to be searched
                mask2 = np.ones(mask.shape).astype(bool)

                xx, yy = np.arange(bbox_crop[0], bbox_crop[0]+bbox_crop[2]), np.arange(bbox_crop[1], bbox_crop[1]+bbox_crop[3])
                u, v = np.meshgrid(xx, yy)
                pixels = np.hstack((v.flatten().reshape(-1, 1), u.flatten().reshape(-1, 1)))

                mask2[tuple(pixels.T)] = 0
                mask[mask2] = 0

            img[np.logical_not(mask)] = [255, 255, 255]
            img[mask] = [0, 0, 0]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Find contours, obtain bounding box
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if w*h < 1000:
                    continue
                cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)
                if bbox_crop is not None:
                    cv2.rectangle(original, (bbox_crop[0], bbox_crop[1]), (bbox_crop[0] + bbox_crop[2], bbox_crop[1] + bbox_crop[3]), (255, 0, 0), 2)
                bboxes.append([x, y, x+w, y+h])
                colors.append(color)

        im_msg = self.bridge.cv2_to_imgmsg(original)
        self.pub.publish(im_msg)
        return mujoco_bboxResponse(np.ravel(bboxes), colors)

    def load_topic(self, topic):
        # Read image
        msg = rospy.wait_for_message(topic, Image)
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        return img


if __name__ == "__main__":
    rospy.init_node("classification_node")

    classificator = Classificator()

    rospy.Service('classify_mujoco', mujoco_bbox, classificator)
    rospy.spin()
