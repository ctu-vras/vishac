#!/usr/bin/env python3
"""
The main logic of the pipeline

@author Lukas Rustler
"""
import datetime
import glob
import time
from shape_completion.save_pcl import save_pcl
from shape_completion.utils import rotate_and_scale, compute_impact_position,\
    choose_voxel, make_movement, change_point_finger, add_to_PC, detect_collision, \
    is_feasible, transform_free_space, \
    move_files, create_mesh_bbox, remove_mesh_bbox, compute_pose, get_impact_pose, check_fall
from kinova_mujoco.python_interface import MoveGroupPythonInterface
from shape_completion.subscriber_classes import KDL_PUBLISHER
from kinova_mujoco import kinematics_interface
from subprocess import PIPE, Popen, call
import os
import rospy
import numpy as np
import argparse
from sensor_msgs.msg import JointState
from IGR.preprocess import preprocess
import json
import tf.transformations as ts
import open3d as o3d
from igr.msg import igr_request
import tf
from sensor_msgs.msg import CameraInfo
from mujoco_interface_msgs.srv import GetAllObjectPoses, GetAllObjectPosesRequest


def log_subprocess_output(proc):
    """
    Change prints to rospy.loginfo() messages
    @param proc: subsprocess.Popen object
    @type proc: subsprocess.Popen object
    @return:
    @rtype:
    """
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        if line.rstrip():
            rospy.loginfo(line.rstrip())


def prepare_parser():
    arg_parser = argparse.ArgumentParser(
        description="Main script for shape completion experiments"
    )
    arg_parser.add_argument(
        "--number_of_reconstruction",
        "-r",
        dest="recs_num",
        required=False,
        default=5,
        help="How many reconstruction to done, not including the first one"
    )

    arg_parser.add_argument(
        "--detection_type",
        "-d",
        dest="detection_type",
        required=False,
        default="cusum",
        help="What collision detection algo to use: cusum"
    )

    arg_parser.add_argument(
        "--publish",
        "-p",
        dest="publish_new_joints",
        action="store_true",
        required=False,
        default=False,
        help="if to publish new joint_states msg with external torques"
    )

    arg_parser.add_argument(
        "--net",
        "-n",
        dest="net",
        required=False,
        default="IGR",
        help="Which net to use: IGR, random"
    )

    arg_parser.add_argument(
        "--max_time",
        "-t",
        dest="max_time",
        required=False,
        default=180,
        help="How long to run the pipeline (in secs)"
    )

    args = arg_parser.parse_args()
    return int(args.recs_num)+1, args.detection_type, args.publish_new_joints, args.net, int(args.max_time)


def main(recs_num, detection_type, publish_new_joints, net, max_time):
    # absolute path to this file
    file_dir = os.path.dirname(os.path.abspath(__file__))

    rospy.init_node("shape_completion_main")
    t = rospy.Time.now().to_time()
    # Get params
    robot_name = rospy.get_param("/robot_name", "")
    real_setup = rospy.get_param("/real_setup", False)
    save_cameras = rospy.get_param("/save_cameras", False)
    joint_states_topic = "/joint_states" if robot_name == "" else ("/" + robot_name + "/joint_states")
    base_feedback_topic = "/base_feedback" if robot_name == "" else ("/" + robot_name + "/base_feedback")
    tf_listener = tf.TransformListener()
    msg = rospy.wait_for_message(joint_states_topic, JointState)
    names = msg.name

    camera_info = rospy.wait_for_message(rospy.get_param("/rgbd_segmentation/rgb_camera_info_topic"), CameraInfo)
    w_max, h_max = camera_info.width, camera_info.height

    R = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).T
    times_per_touch = []
    times_per_rec = []
    touch_time_start = time.time()

    # Change IGR config to work properly with different paths
    with open(os.path.join(file_dir, "../../IGR/configs/default_setup.conf"), "r") as f:
        conf = f.read()
    conf = conf.replace("PATH_PLACEHOLDER", os.path.join(file_dir, "../data/"))
    with open(os.path.join(file_dir, "../../IGR/configs/shape_completion_setup.conf"), "w") as f:
        f.write(conf)

    rospy.set_param("/IGR/config_ready", True)
    igr_request_publisher = rospy.Publisher("/IGR/request_topic", igr_request, queue_size=20)

    # Find  at which indexes are the joint of the robot (It can change depending on names of the other joints)
    joints_idxs = []
    for idx, name in enumerate(names):
        if "joint_" in name:
            joints_idxs.append(idx)

    cur_datetime = rospy.get_param("/actvh/timestamp", -1)
    while cur_datetime == -1:
        cur_datetime = rospy.get_param("/actvh/timestamp", -1)

    cur_datetime = str(cur_datetime)
    print(cur_datetime)
    # run rosbag to save information about joint_states
    bag_path = os.path.join(file_dir, "../data/rosbags", cur_datetime)
    rospy.loginfo("Starting rosbag recording into: " + str(bag_path))

    if not os.path.exists(os.path.dirname(bag_path)):
        os.makedirs(os.path.dirname(bag_path))
    topics_to_save = ["kinova_mujoco/joint_states", joint_states_topic, "joint_states_custom",
                      "trajectory_execution_event", "rgb_th", "depth_th"]
    if save_cameras:
        topics_to_save += ["camera_0_pc", "camera_1_pc", "camera_pc"]
    command = "rosbag record -O " + bag_path + " " + " ".join(topics_to_save) + " __name:=my_bag"

    Popen(command, stdout=PIPE, shell=True)

    # Move group python interface init for arm
    rospy.loginfo("Creating MoveGroup python interface instance for arm")
    MoveGroupArm = MoveGroupPythonInterface("arm")

    if MoveGroupArm.group.get_end_effector_link() == "end_effector_link":
        MoveGroupArm.group.set_end_effector_link("tool_frame")

    rospy.loginfo("Planning with: " + MoveGroupArm.group.get_end_effector_link())

    # Move group python interface init for gripper
    rospy.loginfo("Creating MoveGroup python interface instance for gripper")
    MoveGroupGripper = MoveGroupPythonInterface("gripper")

    # Publish external torques computed with KDL
    if publish_new_joints:
        ext_torque_publisher = KDL_PUBLISHER(joints_idxs, MoveGroupArm.group.get_end_effector_link())
        ext_torque_publisher.pub = rospy.Publisher("/joint_states_custom", JointState, queue_size=100)
        ext_torque_publisher.sub = rospy.Subscriber(joint_states_topic, JointState, ext_torque_publisher)
        joint_states_topic = "/joint_states_custom"

    # Prepare gripper subscriber
    if real_setup:
        from shape_completion.subscriber_classes import GRIPPER_FEEDBACK
        from shape_completion.gripper_utils import send_speed
        from kortex_driver.msg import BaseCyclic_Feedback

        gripper_subscriber = GRIPPER_FEEDBACK(0.05)
        gripper_subscriber.send_speed = send_speed
        gripper_subscriber.sub = rospy.Subscriber(base_feedback_topic, BaseCyclic_Feedback, gripper_subscriber)

    if not real_setup:
        # Close the gripper
        rospy.loginfo("Restarting the simulation if necessary")
        while True:
            gripper_cur_joints = MoveGroupGripper.group.get_current_joint_values()
            # The simulation sometimes glitched and run robot with open gripper -> restart simulation if it happens
            gripper_joints = [0.80, -0.79, 0.82, 0.83, 0.80, -0.80]

            if np.linalg.norm(np.array(gripper_cur_joints) - np.array(gripper_joints)) > 0.1:
                proc = Popen("rosservice call /kinova_mujoco/reset '{}'", stdout=PIPE, shell=True)
                proc.wait()
                rospy.sleep(3)
            else:
                break

    if not real_setup:
        get_all_poses = rospy.ServiceProxy('/kinova_mujoco/getAllObjectPoses', GetAllObjectPoses)

    # Inverse kinematics interface
    rospy.loginfo("Creating Inverse Kinematics instance")
    inverse_kinematics = kinematics_interface.InverseKinematics()

    # Forward kinematics interface
    rospy.loginfo("Creating Forward Kinematics instance")
    forward_kinematics = kinematics_interface.ForwardKinematics()

    mesh_path = os.path.join(file_dir, "../data/meshes", cur_datetime)
    rospy.set_param("/IGR/save_directory", mesh_path)
    # .pcd file path
    pcd_path = os.path.join(file_dir, "../data/pcd", cur_datetime)
    # .npy path
    npy_path = os.path.join(pcd_path.split("pcd")[0], "npy", cur_datetime)
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)

    # save point cloud to .pcd file to right location
    # bboxes (x,y,w,h)
    objects_file_names, classes, bboxes = save_pcl(pcd_path, simulation=not real_setup)
    rospy.set_param("/actvh/objects_file_names", objects_file_names)
    free_spaces = {}
    for obj_fname in objects_file_names:
        free_spaces[obj_fname] = []
        rospy.set_param("/IGR/free_space/"+obj_fname, [])

    bbox_enlarge = 0.25
    for bbox_id in range(len(bboxes) // 4):
        bbox = bboxes[bbox_id * 4:(bbox_id + 1) * 4]
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        bboxes[bbox_id * 4] = max(0, bbox[0] - int(bbox_enlarge * w))

        if (w + int(bbox_enlarge*2 * w) + bboxes[bbox_id * 4]) < w_max:
            bboxes[(bbox_id * 4) + 2] = w + int(bbox_enlarge*2 * w)
        else:
            bboxes[(bbox_id * 4) + 2] = w_max - bboxes[bbox_id * 4] - 1

        if h + int(bbox_enlarge * h) + bboxes[(bbox_id * 4) + 1] < h_max:
            bboxes[(bbox_id * 4) + 3] = h + int(bbox_enlarge * h)
        else:
            bboxes[(bbox_id * 4) + 3] = h_max - bboxes[(bbox_id * 4) + 1] - 1

    rec_id = 0
    selected = np.array([])
    pcd_paths = []
    centers = []
    touched_objects = np.array([])
    obj_to_touch_transform = [np.eye(4) for _ in range(len(objects_file_names))]
    start_pose = None
    joint_angles = None
    
    while rec_id < recs_num:
        if net in ["IGR", "random"]:
            if rec_id == 0:
                # prepare command and call preprocessing
                for obj_file_name in objects_file_names:
                    preprocess(os.path.join(pcd_path, obj_file_name+".pcd"), npy_path, "save")

                # Create json split file for IGR
                split = {"npy": {cur_datetime: objects_file_names}}
                with open(os.path.join(file_dir, "../../IGR/splits/conf.json"), "w") as split_file:
                    json.dump(split, split_file)

                # call all the shapes to be created
                for obj_file_name_id in range(len(objects_file_names)):
                    igr_request_publisher.publish(igr_request(True, True, obj_file_name_id,
                                                              objects_file_names[obj_file_name_id]))

            else:
                if start_pose is None or joint_angles is None:
                    rospy.loginfo(f"Calling IGR to get random shape")
                    new_object_id = np.random.randint(0, len(objects_file_names))
                    igr_request_publisher.publish(igr_request(False, True, new_object_id, objects_file_names[new_object_id]))
                else:
                    rospy.loginfo(f"Calling IGR to get shape")
                    igr_request_publisher.publish(igr_request(False, True, -1, ""))

            wait_for_done_start = time.time()
            wait_threshold = 60 if rec_id == 0 else 15
            while not rospy.is_shutdown():
                igr_completed_ids = rospy.get_param("/IGR/completed_ids", [])
                if len(igr_completed_ids) != 0:
                    if rec_id != 0:
                        break
                    else:
                        if sorted(igr_completed_ids) == sorted(range(0, len(objects_file_names))):
                            break
                 
                if (time.time() - wait_for_done_start) > wait_threshold:
                    break

                rospy.sleep(0.1)

            rospy.set_param("/IGR/completed_ids", [])
            times_per_rec.append(time.time()-touch_time_start)

        old_paths = []
        for _ in objects_file_names:
            temp_path = os.path.join(pcd_path, _) + ".pcd"
            if os.path.isfile(temp_path.replace(".pcd", ".npy")):
                old_paths.append(temp_path.replace(".pcd", ".npy"))
            else:
                old_paths.append(temp_path)

        obj_to_touch_transform, classes, bboxes = \
            compute_pose(pcd_path, [os.path.join(pcd_path, _) + "_pose.pcd" for _ in objects_file_names],
                         old_paths, finger_pose=None, R=R, simulation=not real_setup,
                         classes=classes, bboxes=bboxes, objects_names=[_.split("__")[0] for _ in objects_file_names],
                         old_transforms=obj_to_touch_transform)

        for bbox_id in range(len(bboxes) // 4):
            bbox = bboxes[bbox_id * 4:(bbox_id + 1) * 4]
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            bboxes[bbox_id * 4] = max(0, bbox[0] - int(bbox_enlarge * w))

            if (w + int(bbox_enlarge * 2 * w) + bboxes[bbox_id * 4]) < w_max:
                bboxes[(bbox_id * 4) + 2] = w + int(bbox_enlarge * 2 * w)
            else:
                bboxes[(bbox_id * 4) + 2] = w_max - bboxes[bbox_id * 4] - 1

            if h + int(bbox_enlarge * h) + bboxes[(bbox_id * 4) + 1] < h_max:
                bboxes[(bbox_id * 4) + 3] = h + int(bbox_enlarge * h)
            else:
                bboxes[(bbox_id * 4) + 3] = h_max - bboxes[(bbox_id * 4) + 1] - 1

        # Rotate and scale the object back
        rospy.loginfo("Rotating and scaling back the mean mesh")
        meshes_paths = []
        for obj_id in range(len(objects_file_names)):
            pcd_path_temp = os.path.join(pcd_path, objects_file_names[obj_id])+".pcd"
            points = np.asarray(o3d.io.read_point_cloud(pcd_path_temp).points)

            points = (R.T @ obj_to_touch_transform[obj_id] @ np.hstack((points[:, :3], np.ones((points.shape[0], 1)))).T)[:3, :].T
            np.save(pcd_path_temp.replace(".pcd", "_rotated.npy"), points)

            if rec_id == 0:
                pcd_paths.append(pcd_path_temp.replace(".pcd", "_rotated.npy"))
            meshes_paths.append("package://shape_completion/data/meshes/" + cur_datetime + "/rep" + str(rec_id) + "/" +
                                objects_file_names[obj_id] + "_rec.stl")

            mesh_path_cur = os.path.join(mesh_path, objects_file_names[obj_id])

            center = rotate_and_scale(mesh_path_cur, rec_id, obj_to_touch_transform=obj_to_touch_transform[obj_id])
            if rec_id == 0:
                centers.append(center)

        if rec_id == 0:
            rospy.set_param("/shape_completion/new_pc_path", pcd_paths)
        rospy.set_param("/shape_completion/rec_mesh_path", meshes_paths)

        if rospy.Time.now().to_time() - t > (max_time - 10):
            rec_id += 1
            break

        if rec_id != recs_num - 1:
            if real_setup:
                gripper_subscriber.send_speed(-0.5)

            touch_done = False
            free_space_temp = []
            while not touch_done:
                # Create bounding box around the point cloud to not bump into the object
                create_mesh_bbox(MoveGroupArm, [0, 0, 0], os.path.join(mesh_path, "rep"+str(rec_id)))

                rospy.loginfo("Choosing the impact voxel")

                # check whether the object was touched the three last times -> delete it from the pool
                if rec_id > 2 and len(objects_file_names) > 1 and touched_object_id is not None and \
                        np.all(touched_objects[-3:] == touched_object_id):
                    exclude = objects_file_names[touched_object_id]
                else:
                    exclude = ""

                if selected is None:
                    selected = np.array([])

                start_pose, impact_pose, direction, selected, result_angles, touched_object_id = \
                    choose_voxel(mesh_path, inverse_kinematics, 8.5, selected, net, tf_listener=tf_listener,
                                 exclude=exclude, objects_file_names=objects_file_names)

                touch_done = start_pose is None
                if start_pose is not None:
                    rospy.loginfo(f"Going to touch shape {objects_file_names[touched_object_id]}")

                    if rec_id == 0:
                        touched_objects = np.array([touched_object_id])
                    else:
                        touched_objects = np.append(touched_objects, touched_object_id)

                    cur_pose = MoveGroupArm.get_finger_pose()
                    cur_pose = np.array([cur_pose.position.x, cur_pose.position.y, cur_pose.position.z])
                    goal_pose = np.array([start_pose.position.x, start_pose.position.y, start_pose.position.z])

                    eikonals = glob.glob(os.path.join(mesh_path, "*eikonal_rotated.npy"))
                    for eikonal_path_id, eikonal_path in enumerate(eikonals):
                        if eikonal_path_id == 0:
                            eikonal_points = np.load(eikonal_path)
                        else:
                            eikonal_points = np.vstack((eikonal_points, np.load(eikonal_path)))

                    # Move to the start position
                    rospy.loginfo("Moving to start position at: " + str(rospy.Time.now().to_time() - t))
                    make_movement(start_pose, MoveGroupArm, wait=False)

                    rospy.sleep(1)
                    last_pose = np.zeros((5, 3))
                    last_pose_id = 0
                    move_time = time.time()
                    while True:
                        cur_pose = MoveGroupArm.get_finger_pose()
                        cur_pose = [cur_pose.position.x, cur_pose.position.y, cur_pose.position.z]

                        free_space_temp.append(cur_pose)
                        if np.sum(np.linalg.norm(last_pose - np.array(cur_pose), axis=1)) < 0.005 or time.time()-move_time > 30:
                            break
                        last_pose[last_pose_id % 5] = np.array(cur_pose)
                        last_pose_id += 1
                        rospy.sleep(0.35)

                    if not real_setup and check_fall(get_all_poses, MoveGroupGripper):
                        touch_done = True
                        rec_id -= 1
                        touched_objects = touched_objects[:-1]
                        start_pose = None
                        break
                    rospy.loginfo("Turning off the collision detection for object and finger")
                    remove_mesh_bbox(MoveGroupArm, objects_file_names[touched_object_id]+"_bbox")

                    # Move to impact position
                    rospy.loginfo("Moving to impact position at: " + str(rospy.Time.now().to_time() - t))
                    possible_move = make_movement(impact_pose, MoveGroupArm, cartesian=True, speed=0.15)

                    # if possible move -> go to collision and end touch loop
                    touch_done = possible_move

                    joint_angles = None
                    if possible_move:
                        if detection_type == "cusum":
                            joint_angles = detect_collision("cusum", 0.0125, MoveGroupArm, drift=0.1, joints_idxs=joints_idxs,
                                                            topic=joint_states_topic,
                                                            free_space=free_space_temp)
                        elif detection_type == "threshold":
                            joint_angles = detect_collision("threshold", 0.5, MoveGroupArm, joints_idxs=joints_idxs,
                                                            topic=joint_states_topic,
                                                            free_space=free_space_temp)

                        rospy.loginfo("Stopping the robot at: " + str(rospy.Time.now().to_time() - t))
                        MoveGroupArm.stop_robot()

                        if joint_angles is not None:
                            # Compute pose when collision detected, from forward kinematics and find position in finger frame
                            pose_finger = get_impact_pose(impact=False, MoveGroupArm=MoveGroupArm, tf_listener=tf_listener)

                            new_path = os.path.join(pcd_path, objects_file_names[touched_object_id] + ".pcd")
                            if os.path.isfile(new_path.replace(".pcd", ".npy")):
                                new_path = new_path.replace(".pcd", ".npy")

                            touch_to_obj_transform, classes_temp, bboxes_temp = \
                                compute_pose(pcd_path, [new_path],
                                             [os.path.join(pcd_path, objects_file_names[touched_object_id] + "_pose.pcd")],
                                             finger_pose=pose_finger, R=R,
                                             simulation=not real_setup, classes=[classes[touched_object_id]],
                                             bboxes=bboxes[touched_object_id * 4:(touched_object_id + 1) * 4],
                                             objects_names=[objects_file_names[touched_object_id].split("__")[0]],
                                             old_transforms=[ts.inverse_matrix(obj_to_touch_transform[touched_object_id])])

                            if not real_setup and check_fall(get_all_poses, MoveGroupGripper):
                                touch_done = True
                                rec_id -= 1
                                touched_objects = touched_objects[:-1]
                                start_pose = None
                                break
                            else:
                                touch_to_obj_transform = touch_to_obj_transform[0]
                                obj_to_touch_transform[touched_object_id] = ts.inverse_matrix(touch_to_obj_transform)

                            w, h = bboxes_temp[2] - bboxes_temp[0], bboxes_temp[3] - bboxes_temp[1]
                            bboxes[touched_object_id * 4] = max(0, bboxes_temp[0] - int(bbox_enlarge * w))

                            if (w + int(bbox_enlarge*2 * w) + bboxes[touched_object_id * 4]) < w_max:
                                bboxes[(touched_object_id * 4) + 2] = w + int(bbox_enlarge*2 * w)
                            else:
                                bboxes[(touched_object_id * 4) + 2] = w_max - bboxes[touched_object_id * 4] - 1

                            if (h + int(bbox_enlarge * h) + bboxes[(touched_object_id * 4) + 1]) < h_max:
                                bboxes[(touched_object_id * 4) + 3] = h + int(bbox_enlarge * h)
                            else:
                                bboxes[(touched_object_id * 4) + 3] = h_max - bboxes[(touched_object_id * 4) + 1] - 1

                            rospy.loginfo("Moving from collision")
                            make_movement(start_pose, MoveGroupArm, cartesian=True, collisions=True,
                                          wait=False, speed=1)  # get from collision

                            rospy.loginfo("Collision detected at: \n" + str(pose_finger))

                            impact_finger_pose = np.array([pose_finger.position.x, pose_finger.position.y,
                                                                       pose_finger.position.z])
                            free_space_temp = np.array(free_space_temp)
                            fr_mask = np.linalg.norm(free_space_temp-impact_finger_pose, axis=1) > 0.005
                            for obj_id, obj_fname in enumerate(objects_file_names):
                                free_space_new = transform_free_space(free_space_temp[fr_mask], R,
                                                                      R.T @ touch_to_obj_transform @ R,
                                                                      centers[obj_id])
                                if rec_id == 0:
                                    free_spaces[obj_fname] = free_space_new
                                else:
                                    free_spaces[obj_fname] = np.vstack((free_spaces[obj_fname], free_space_new))
                                rospy.set_param("/IGR/free_space/"+obj_fname, np.ravel(free_spaces[obj_fname]).tolist())

                            # Add new information into pointcloud
                            rospy.loginfo("Adding new points into the point cloud")
                            add_to_PC(pose_finger, os.path.join(pcd_path, objects_file_names[touched_object_id])+".pcd", 5 / 1e3,
                                      rec_id, "circle", touch_to_obj_transform, add_pose_estimation=True)

                            move_files(mesh_path, npy_path, rec_id)

                            # prepare command and call preprocessing
                            preprocess(os.path.join(pcd_path, objects_file_names[touched_object_id] + ".pcd"), npy_path, "load")
                            igr_request_publisher.publish(igr_request(True, False, touched_object_id, objects_file_names[touched_object_id]))
                            last_pose = np.zeros((3, 3))
                            last_pose_id = 0
                            move_time = time.time()
                            while True:
                                cur_pose = MoveGroupArm.get_finger_pose()
                                cur_pose = [cur_pose.position.x, cur_pose.position.y, cur_pose.position.z]
                                if np.sum(np.linalg.norm(last_pose - np.array(cur_pose), axis=1)) < 0.0025 or time.time() - move_time > 30:
                                    break
                                last_pose[last_pose_id % 3] = np.array(cur_pose)
                                last_pose_id += 1
                                rospy.sleep(0.25)

                        else:
                            rospy.loginfo("Moving from collision")
                            make_movement(start_pose, MoveGroupArm, cartesian=True, collisions=True,
                                          wait=False, speed=1)  # get from collision
                            move_files(mesh_path, npy_path, rec_id)
                    else:  # remove touch id as it was never actually touched
                        touched_objects = touched_objects[:-1]

                    rospy.loginfo("Turning on the collision detection for object and finger")
                    create_mesh_bbox(MoveGroupArm, [0, 0, 0], os.path.join(mesh_path, "rep"+str(rec_id)))
                else:
                    # No start position -> End touch
                    touch_done = True
        rec_id += 1
        times_per_touch.append(time.time()-touch_time_start)
    np.save(os.path.join(npy_path, "free_space.npy"), free_spaces)

    rospy.set_param("/actvh_running", False)
    while not rospy.get_param("/IGR/ended", False):
        rospy.sleep(0.1)
    times_per_rec.append(time.time()-touch_time_start)
    rec_id -= 1

    for obj_id in range(len(objects_file_names)):
        mesh_path_cur = os.path.join(mesh_path, objects_file_names[obj_id])
        center = rotate_and_scale(mesh_path_cur, rec_id, obj_to_touch_transform=obj_to_touch_transform[obj_id])

    move_files(mesh_path, npy_path, rec_id)

    files = glob.glob(os.path.join(pcd_path, "*"))
    for file in files:
        if not os.path.isdir(file):
            _, f_name = os.path.split(file)
            if not os.path.exists(os.path.join(os.path.dirname(file), "rep" + str(rec_id))):
                os.makedirs(os.path.join(os.path.dirname(file), "rep" + str(rec_id)))
            proc = call("cp " + file + " " + os.path.join(os.path.dirname(file), "rep" + str(rec_id), f_name),
                         shell=True)

    np.save(os.path.join(npy_path, "times.npy"), times_per_touch)
    np.save(os.path.join(npy_path, "times_per_rec.npy"), times_per_rec)

    # Close the bag recording
    rospy.loginfo("Killing bag recording")
    proc = Popen("rosnode kill /my_bag", shell=True)
    proc.wait()
    if publish_new_joints:
        ext_torque_publisher.sub.unregister()

    rospy.loginfo("Total time: " + str(rospy.Time.now().to_time() - t))


if __name__ == "__main__":
    recs_num, detection_type, publish_new_joints, net, max_time = prepare_parser()
    main(recs_num, detection_type, publish_new_joints, net, max_time)
