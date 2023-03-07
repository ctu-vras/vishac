#!/usr/bin/env python3
"""
Function to use yolo from ROS

@author Lukas Rustler
"""

from yolo.srv import call_yolo, call_yoloResponse
import rospy
import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from yolo.models.experimental import attempt_load
from yolo.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolo.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from yolo.utils.datasets import letterbox
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from utils.plots import plot_one_box
import timm
import json


class YOLO:
    """
    Class to handle yolo requests for the scene
    """
    def __init__(self, opt):
        self.pub = None
        self.bridge = CvBridge()
        self.opt = opt
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        weights = os.path.join(self.file_dir, self.opt.weights)
        trace = not self.opt.no_trace

        print(self.opt.second_classifier)
        # Initialize
        set_logging()
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        if self.opt.second_classifier:
            self.modelc = timm.create_model("resnet152d", pretrained=True)
            # self.modelc = timm.create_model("efficientnet_b2", pretrained=True)

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.opt.img_size, s=self.stride)  # check img_size

        if trace:
            self.model = TracedModel(self.model, self.device, self.opt.img_size)

        if self.half:
            self.model.half()  # to FP16

        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Get names and colors
        self.names = np.array(self.model.module.names if hasattr(self.model, 'module') else self.model.names)
        if self.opt.second_classifier:
            with open(os.path.join(self.file_dir, "imagenet.json"), "r") as f:
                self.names = np.array([_ for _ in json.load(f).values()])
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1

    def __call__(self, msg):
        """
        Get the topic name and return bboxes and uncertainties
        @return:
        @rtype:
        """
        img, im0s = self.load_topic(msg.topic)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (
                self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=self.opt.augment)[0]

        # Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                   agnostic=self.opt.agnostic_nms)

        # Resnet
        if self.opt.second_classifier:
            pred = apply_classifier(pred, self.modelc, img, im0s)

        # Process detections
        for det in pred:  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                if msg.merge_box:
                    while True:
                        nothing_merged = True
                        det_local = []
                        j_added = -1
                        for i, d in enumerate(reversed(det)):
                            if j_added != -1:
                                break
                            for j, d2 in enumerate(reversed(det)):
                                if j <= i:
                                    continue
                                *xyxy, conf, cls = d
                                *xyxy2, conf2, cls2 = d2
                                c1 = np.abs(xyxy[0] - xyxy2[0]) < 15 or np.abs(xyxy[2] - xyxy2[2]) < 15
                                c2 = np.abs(xyxy[1] - xyxy2[1]) < 15 or np.abs(xyxy[3] - xyxy2[3]) < 15

                                overlap = not (xyxy[2] < xyxy2[0] or xyxy[0] > xyxy2[2] or xyxy[1] > xyxy2[3] or xyxy[3] < xyxy2[1])
                                if (c1 or c2) and overlap:
                                    new_conf = conf if conf > conf2 else conf2
                                    new_cls = cls if conf > conf2 else cls2
                                    det_local.append([np.min([xyxy[0], xyxy2[0]]), np.min([xyxy[1], xyxy2[1]]), np.max([xyxy[2], xyxy2[2]]), np.max([xyxy[3], xyxy2[3]]), float(new_conf), float(new_cls)])
                                    nothing_merged = False
                                    j_added = j
                                if j_added != -1:
                                    break
                        if j_added != -1:
                            for idx, d in enumerate(reversed(det)):
                                if idx != j and idx != (i-1):
                                    if isinstance(d, torch.Tensor):
                                        det_local.append(d.tolist())
                                    else:
                                        det_local.append(d)
                            det = det_local
                        if nothing_merged:
                            break
                    det = np.array(det)
                else:
                    det = reversed(det)
            # Write results
            output = []
            for *xyxy, conf, cls in det:
                label = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0s, label=label, color=self.colors[int(cls)], line_thickness=1)
                if isinstance(conf, torch.Tensor):
                    temp = [cls.numpy(), conf.numpy()] + xyxy  # xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]]
                else:
                    temp = [cls, conf] + xyxy  # xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]]
                output.append(temp)
            output = np.array(output)
            im_msg = self.bridge.cv2_to_imgmsg(im0s)
            self.pub.publish(im_msg)

            return call_yoloResponse(self.names[output[:, 0].flatten().astype(int)].tolist(),
                                     output[:, 1].flatten().tolist(), list(map(int, output[:, 2:].flatten())))

    def load_topic(self, topic):
        # Read image
        msg = rospy.wait_for_message(topic, Image)
        img0 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # img0 = cv2.resize(img0, (450, 240))
        # Padded resize
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img, img0


if __name__ == "__main__":
    rospy.init_node('call_yolo_node')

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--second-classifier', action='store_true')
    parser.add_argument('--merge-box', action='store_true')
    opt = parser.parse_args()
    with torch.no_grad():
        yolo = YOLO(opt)
        yolo.pub = rospy.Publisher("/yolo", Image, queue_size=20)
        rospy.Service('call_yolo', call_yolo, yolo)

        rospy.spin()

