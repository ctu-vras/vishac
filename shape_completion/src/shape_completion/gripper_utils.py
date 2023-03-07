#!/usr/bin/env python3
"""
Utils for Robotiq 85-2F gripper commands
@author Lukas Rustler
"""
import rospy
from kortex_driver.srv import SendGripperCommand
from kortex_driver.msg import GripperCommand
from kortex_driver.msg import GripperMode
from kortex_driver.msg import Finger
from kortex_driver.msg import Gripper
from kortex_driver.msg import BaseCyclic_Feedback
from shape_completion.subscriber_classes import GRIPPER_FEEDBACK


def send_speed(speed):
    """
    Send command to gripper with given speed
    @param speed: speed to be send to the gripper 0-1
    @type speed: float
    @return:
    @rtype:
    """
    # Prepare finger message
    finger = Finger()
    finger.finger_identifier = 0
    finger.value = speed

    # Prepare gripper message
    gripper = Gripper()
    gripper.finger.append(finger)

    # Prepare command message
    gc = GripperCommand()
    gc.gripper = gripper
    gc.mode = GripperMode.GRIPPER_SPEED  # We want to command with speed

    try:
        robot_name = rospy.get_param("/robot_name", "")
        gc_topic = '/base/send_gripper_command' if robot_name == "" else ("/"+robot_name+"/base/send_gripper_command")
        gripper_service = rospy.ServiceProxy(gc_topic, SendGripperCommand)
        res = gripper_service(gc)

    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


if __name__ == "__main__":
    """Basic test setting"""
    rospy.init_node("gripper_utils", anonymous=True)
    gripper_subscriber = GRIPPER_FEEDBACK(0.03)
    gripper_subscriber.send_speed = send_speed
    robot_name = rospy.get_param("/robot_name", "")
    bs_topic = "/base_feedback" if robot_name == "" else ("/"+robot_name+"/base_feedback")
    gripper_subscriber.sub = rospy.Subscriber(bs_topic, BaseCyclic_Feedback, gripper_subscriber)
    gripper_subscriber.detect = True
    send_speed(-0.05)
    while gripper_subscriber.detect:
        continue

