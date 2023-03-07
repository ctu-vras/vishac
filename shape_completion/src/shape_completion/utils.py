#!/usr/bin/env python3
"""
Utils for almost everything

@author Lukas Rustler
"""
import copy
import multiprocessing
import os
import numpy
import scipy.linalg
import trimesh
import numpy as np
import glob
from subprocess import Popen, PIPE, call
from geometry_msgs.msg import PoseStamped
import rospy
from scipy.spatial import cKDTree as kd
import tf.transformations as ts
from geometry_msgs.msg import Quaternion, Point, Pose
import tf
from sensor_msgs.msg import JointState
from shape_completion.subscriber_classes import CUSUM
import ctypes
from std_msgs.msg import String
import open3d as o3d
from shape_completion.save_pcl import save_pcl
from mujoco_interface_msgs.srv import GetAllObjectPoses, GetAllObjectPosesRequest


def rotate_and_scale(mesh_path, rep, obj_to_touch_transform=None):
    """
    Rotates and scale the mesh to fit URDF and GT
    @param mesh_path: path to mesh folder
    @type mesh_path: str
    @param rep: From which repetition are the data, to create proper folder
    @type rep: int
    @param obj_to_touch_transform: transformation from pose estimation
    @type obj_to_touch_transform: list
    @return:
    @rtype:
    """

    scale_path = mesh_path.replace("meshes", "npy") + "_scale.npy"
    # load scale from file (saved while preprocessing)
    scale = np.load(scale_path)
    center = np.load(mesh_path.replace("meshes", "npy") + "_center.npy")

    # Rotate, scale and translate back from IGR reconstruction
    translation = np.eye(4)
    translation[:3, 3] = center.T
    R = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    # Rotate and scale eikonal coords
    eikonal = np.load(mesh_path + "_eikonal.npy")
    eikonal_points = eikonal[:, 1:] * (1 / scale)
    eikonal_points = eikonal_points + center
    eikonal_points = np.hstack((eikonal_points, np.ones((eikonal_points.shape[0], 1))))
    # eikonal_points = np.matmul(R, eikonal_points.T)
    eikonal_points = (R @ obj_to_touch_transform @ eikonal_points.T)[:3, :].T
    eikonal[:, 1:] = eikonal_points
    np.save(os.path.join(mesh_path + "_eikonal_rotated.npy"), eikonal)

    # load the mean mesh
    mesh = trimesh.load(mesh_path+".ply")
    mesh.apply_scale(1 / scale)
    mesh.export(mesh_path + "_eval.ply")
    mesh.apply_transform(translation)
    mesh.apply_transform(R)

    # transform center into base link (NEEDED for simulation right now!!!)
    center = np.matmul(R, np.hstack((center, 1)).T)[:3]

    obj_to_touch_transform = R @ obj_to_touch_transform @ R.T
    mesh.apply_transform(obj_to_touch_transform)

    new_rep_directory = os.path.join(os.path.dirname(mesh_path), "rep"+str(rep))
    if not os.path.exists(new_rep_directory):
        os.makedirs(new_rep_directory)
    mesh.export(os.path.join(new_rep_directory, os.path.basename(os.path.normpath(mesh_path)) + "_rec.stl"))

    mesh.export(mesh_path + "_rec.stl")
    mesh.export(mesh_path + "_rec.ply")

    vertices = mesh.vertices
    bbox_center = np.mean(vertices, axis=0)

    mesh.vertices -= bbox_center

    # Scale by 40%/60%
    if rep <= 3:
        bbox_scale = 1.6
    else:
        bbox_scale = 1.4
    mesh.apply_scale(bbox_scale)

    mesh.vertices += bbox_center
    mesh.export(os.path.join(new_rep_directory, os.path.basename(os.path.normpath(mesh_path)) + "_bbox.ply"))
    mesh.vertices -= bbox_center
    mesh.apply_scale(1 / bbox_scale)

    # Rotate to world
    mesh.apply_transform(R.T)
    mesh.export(mesh_path + "_rotated.ply")

    return center


def transform_free_space(points, R, to_canonical_frame, center):
    """
    Function to transform free space points to canonical pose
    @param points: list of point to be transformed
    @type points: numpy array
    @param R: rotation matrix from RVIZ to real world
    @type R: numpy array
    @param to_canonical_frame: rotation to canonical frame
    @type to_canonical_frame: numpy array
    @param center: center of the object
    @type center: numpy array
    @return:
    @rtype:
    """
    points = to_canonical_frame @ np.hstack((points, np.ones((len(points), 1)))).T

    mask = np.linalg.norm(points[:3, :]-center.reshape((3, 1)), axis=0) < 0.2

    return (R @ points[:, mask])[:3, :].T


def is_feasible(pose, collision_avoidance, ik, grasp=False, timeout=0.1):
    """
    Computes inverse kinematics to check if the position is valid for the robot

    @param pose: At which pose to compute the IK
    @type pose: 1x3 list of floats or Pose
    @param collision_avoidance: Whether to use collision avoidance
    @type collision_avoidance: bool
    @param ik: inverse kinematics solver
    @type ik: InverseKinematics()
    @param grasp: whether the feasibility is computed for grasping
    @type grasp: bool
    @param timeout: timeout of the IK computation
    @type timeout: float
    @return: List of found joint angles
    @rtype: 1x7 list of floats

    """
    ps = PoseStamped()
    ps.header.stamp = rospy.Time.now()
    ps.header.frame_id = "base_link"
    if isinstance(pose, list):
        ps.pose.position.x = pose[0]
        ps.pose.position.y = pose[1]
        ps.pose.position.z = pose[2]
        ps.pose.orientation.x = pose[3]
        ps.pose.orientation.y = pose[4]
        ps.pose.orientation.z = pose[5]
        ps.pose.orientation.w = pose[6]
    elif isinstance(pose, Pose):
        ps.pose = pose
    if not grasp:
        result = ik.getIK("arm", "tool_frame", ps, collision_avoidance, timeout=timeout)
    else:
        result = ik.getIK("arm", "robotiq_arg2f_base_link", ps, collision_avoidance, timeout=timeout)
    return result


def choose_voxel(mesh_path, ik, distance=20, selected=None, net="IGR", tf_listener=None, exclude="",
                 objects_file_names=[]):
    """
    Choses which voxel to take, when there is more than one with the same weight
    @param mesh_path: path to the folder with meshes
    @type mesh_path: str
    @param ik: Instance of inverse kinematics solver
    @type ik: kinova_mujoco.kinematics_interface.InverseKinematics
    @param distance: how far is the starting point from the impact point
    @type distance: float
    @param selected: list with already explored positions
    @type selected: list
    @param net: name of the network to use
    @type net: string
    @param tf_listener: tf listener handle
    @type tf_listener: tf_listener
    @param exclude: objects to be excluded from poitn selection
    @type exclude: string
    @param objects_file_names: names of the objects
    @type objects_file_names: list
    @return:
    @rtype:
    """
    multi_proc = False
    if multi_proc:
        cpus = np.min([8, multiprocessing.cpu_count()])

    # load all shapes and sort them accordingly to names
    eikonals_ = glob.glob(os.path.join(mesh_path, "*eikonal_rotated.npy"))
    indexes = []
    order = [os.path.basename(os.path.normpath(_)).split("_eikonal")[0] for _ in eikonals_]
    for _ in objects_file_names:
        indexes.append(order.index(_))
    eikonals_ = np.array(eikonals_)[indexes]

    # get the point and indexes to given shape
    if exclude != "":
        eikonals = []
        real_indexes = []
        for eik_id, eik in enumerate(eikonals_):
            if exclude not in eik:
                eikonals.append(eik)
                real_indexes.append(eik_id)
    else:
        eikonals = eikonals_
        real_indexes = np.arange(0, len(eikonals))

    # append data and indexes
    for eikonal_path_id, eikonal_path in enumerate(eikonals):
        if eikonal_path_id == 0:
            eikonal = np.load(eikonal_path)
            indexes_per_shape = np.zeros((eikonal.shape[0], 1))
        else:
            eikonal_temp = np.load(eikonal_path)
            eikonal = np.vstack((eikonal, eikonal_temp))
            indexes_per_shape = np.vstack((indexes_per_shape, eikonal_path_id*np.ones((eikonal_temp.shape[0], 1))))
    
    if net == "IGR":
        cluster = eikonal[:, 1:]
        counts = np.abs(eikonal[:, 0])
    elif net == "random":
        cluster = eikonal[:, 1:]
        counts = np.ones((cluster.shape[0], ))
        
    rospy.loginfo("Preproccessing of voxels completed")
    if multi_proc:
        # While there is any untested centroid
        man = multiprocessing.Manager()
        output = man.list()
        lock = multiprocessing.Lock()
        jobs = []
        num_per_cpu = (len(counts) // cpus) + 1
        break_point = cpus - (num_per_cpu * cpus - len(counts))
        last = 0
    else:
        output = []
    (finger_to_tool, rotation) = get_transformation('tool_frame', 'finger_link', tf_listener)
    if isinstance(cluster, numpy.ndarray):
        cl_centers = cluster
    else:
        cl_centers = cluster.cluster_centers_

    # get ckd tree for faster computation
    tree = kd(cl_centers)
    unc_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cl_centers))
    unc_pc.estimate_normals()
    unc_pc.orient_normals_consistent_tangent_plane(10)
    normals = np.asarray(unc_pc.normals)

    if multi_proc:
        for core_id in range(cpus):
            if core_id < break_point:
                shift = num_per_cpu
            else:
                shift = num_per_cpu - 1

            jobs.append(multiprocessing.Process(target=check_feasibility,
                                                args=(counts[last:last + shift],
                                                      cl_centers[last:last + shift],
                                                      net,
                                                      distance, ik, copy.deepcopy(selected),
                                                      output, lock, [finger_to_tool, rotation],
                                                      tree, normals, np.arange(last, last + shift), tf_listener)))
            last = last + shift

            jobs[-1].start()
        for job in jobs:
            job.join()
    else:
        check_feasibility(counts, cl_centers, net, distance, ik, copy.deepcopy(selected), output, None,
                          [finger_to_tool, rotation], tree, normals, np.arange(0, len(counts)),
                          tf_listener)
    if len(output) > 0:
        output = sorted(output, key=lambda x: x[-1], reverse=True)
        rospy.set_param("/shape_completion/impact_point", str(list(output[0][7] - output[0][6])))
        rospy.set_param("/shape_completion/direction", str(list(output[0][2] / (100. / distance))))
        rospy.set_param("/shape_completion/angle_axis", str(list(output[0][5])))

        return output[0][:5] + [real_indexes[int(indexes_per_shape[output[0][-2]])]]
    else:
        rospy.logerr("No feasible solution found!")
        return None, None, None, None, None, None


def check_feasibility(counts, cl_centers, net, distance,
                      ik, selected, output, lock, finger_transform, tree, normals, indexes_global,
                      tf_listener):
    """
    Function to check feasibility of points
    @param counts: uncertainty of given points
    @type counts: list
    @param cl_centers: coordinates of the points
    @type cl_centers: list
    @param net: name of the network to use
    @type net: string
    @param distance: distance from the point where to go
    @type distance: float
    @param ik: inverse kinematics instance
    @type ik: ik
    @param selected: list of already selected positions
    @type selected: numpy array
    @param output: output
    @type output: list
    @param lock: multiprocess lock
    @type lock: multiprocessing.Lock()
    @param finger_transform: transformation to the finger
    @type finger_transform: list
    @param tree: ckd tree of th epoints
    @type tree: cKDTree
    @param normals: normals of each point
    @type normals: numpy array
    @param indexes_global: indexes to shapes
    @type indexes_global: list
    @param tf_listener: transformation lister
    @type tf_listener: tf_listener
    @return:
    @rtype:
    """

    # Loop while get any remaining point and get the one with highest uncertainty
    while sum(1 for _ in counts if _ == -1) != len(counts):
        biggest_cluster = np.argmax(counts)
        centroid = cl_centers[biggest_cluster]
        if len(selected) == 0 or np.all(np.linalg.norm(centroid - selected, axis=1) > 0.05):
            number_of_tries = 2
            for camera_rotation in range(number_of_tries):
                if np.any(counts[biggest_cluster] <= np.array([_[-1] for _ in output])):
                    return True
                if camera_rotation == 0:
                    move_to_pose, impact_pose, direction, angle_axis, impact_point, finger_to_tool = compute_impact_position(
                        centroid, distance, net=net, tree=tree, normals=normals,
                        finger_transform=finger_transform,
                        tf_listener=tf_listener)
                else:
                    move_to_pose, impact_pose, direction, angle_axis, impact_point, finger_to_tool = compute_impact_position(
                        centroid, distance, [direction, angle_axis, impact_point, finger_to_tool, camera_rotation],
                        net=net, tree=tree, normals=normals,
                        finger_transform=finger_transform,
                        tf_listener=tf_listener)
                if move_to_pose is None:
                    counts[biggest_cluster] = -1
                    continue
                result_angles = is_feasible(move_to_pose, True, ik)
                if result_angles:  # Try to compute inverse kinematics for the point
                    if selected.shape[0] == 0:
                        selected = np.append(selected, centroid)
                        selected = np.expand_dims(selected, 0)
                    else:
                        selected = np.vstack((selected, centroid))
                    if lock is not None:
                        lock.acquire()
                    output.append([move_to_pose, impact_pose, direction, selected, result_angles,
                                   angle_axis, finger_to_tool, impact_point, indexes_global[biggest_cluster],
                                   counts[biggest_cluster]])
                    if lock is not None:
                        lock.release()
                    return True
                elif not result_angles and camera_rotation == (number_of_tries - 1):
                    counts[biggest_cluster] = -1
                    if selected.shape[0] == 0:
                        selected = np.append(selected, centroid)
                        selected = np.expand_dims(selected, 0)
                    else:
                        selected = np.vstack((selected, centroid))
        else:
            counts[biggest_cluster] = -1
    return True


def compute_direction_vector(impact_point, tree, normals, neighbor_size=5):
    """
    Computes direction vector to impact
    @param impact_point: impact point in metres of voxels
    @type impact_point: 1x3 list of floats
    @param tree: tree with points to quickly find neighbors
    @type tree: cKDTree
    @param normals: normals of the points
    @type normals: numpy array
    @param neighbor_size: Number of neigbors from KNN to take or distance if voxels are used
    @type neighbor_size: int
    @return: Direction to impact,
    @rtype: 1x3 list of floats
    """

    # Use KN-Neighbors to find 'k' nearest triangles to the impact point and compute mean normal in the point
    [_, i] = tree.query(impact_point, k=neighbor_size)
    direction = np.mean(normals[i, :], axis=0)
    direction = direction / np.linalg.norm(direction)

    return direction


def compute_impact_position(centroid, distance=20, rotate_up=None, tree=None, normals=None,
                            net="IGR", finger_transform=None,
                            tf_listener=None):
    """
    Computes impact point on mesh, direction of movement and starting point of the movement
    @param centroid: Position of impact point in voxel coordinates
    @type centroid: list, 3x1
    @param distance: How far is the starting point from impact point
    @type distance: float
    @param rotate_up: list with information for rotating the last joint for 180 degs
    @type rotate_up: list
    @param tree: tree with points for quicker look up
    @type tree: cKDTree
    @param normals: list of normals of each point
    @type normals: numpy array
    @param net: name of the network to use
    @type net: string
    @param finger_transform: transformation to the finger
    @type finger_transform: list
    @param tf_listener: tranformation listner
    @type tf_listener: tf_listener
    @return: Starting position, impact position, quaternion computed from angle-axis
    @rtype: list of float 1x3, list of Pose 1xn, list of float 1x4
    """
    # Distance coefficient for direction vector
    distance_coef = 100. / distance
    if rotate_up is None:
        # compute real impact point
        impact_point = centroid

        direction = compute_direction_vector(impact_point, tree, normals, 10)
        # compute angle-axis representation
        vecA = [0, 0, 1]
        vecB = -direction
        rotAngle = np.arctan2(np.linalg.norm(np.cross(vecA, vecB)), np.dot(vecA, vecB))
        rotAxis = np.cross(vecA, vecB)
        rotAxis /= np.linalg.norm(rotAxis)
        angle_axis = ts.quaternion_about_axis(rotAngle, rotAxis)

        e = ts.euler_from_quaternion(angle_axis)
        for rotation_fixer in range(5):
            angle = e[1]
            rot = ts.quaternion_about_axis(angle, -direction)
            angle_axis = ts.quaternion_multiply(rot, angle_axis)
            e = ts.euler_from_quaternion(angle_axis)
        rot = ts.quaternion_about_axis(np.pi / 2, -direction)
        angle_axis = ts.quaternion_multiply(rot, angle_axis)
    else:  # Rotate the last link with 180degs
        if rotate_up[0] is None:
            return None, None, None, None, None, None
        impact_point = rotate_up[2]
        direction = rotate_up[0]
        angle_axis = rotate_up[1]
        camera_up = ts.quaternion_about_axis(-np.pi / 2, -direction)
        angle_axis = ts.quaternion_multiply(camera_up, angle_axis)

    # change frame of the impact point
    if rotate_up is None:
        impact_point, finger_to_tool = change_point_finger(impact_point, angle_axis, False, finger_transform,
                                                           tf_listener=tf_listener)
    else:
        finger_to_tool = rotate_up[3]

    if finger_to_tool is None:
        return None, None, None, None, None, None

    # Pose 10cm from impact pose following normal
    move_to_pose = Pose()
    move_to_pose.position = Point(*impact_point + direction / distance_coef)
    move_to_pose.orientation = Quaternion(*angle_axis)

    # Impact pose
    impact_pose = Pose()
    impact_pose.position = Point(*impact_point)
    impact_pose.orientation = Quaternion(*angle_axis)

    impact_waypoints = []
    vec = np.array([move_to_pose.position.x, move_to_pose.position.y, move_to_pose.position.z])
    stop_point = 0.5 if net == "IGR" else 0.3
    tmp = copy.deepcopy(move_to_pose)
    tmp.position = Point(*(vec - direction / (distance_coef * stop_point)))
    impact_waypoints.append(tmp)

    return move_to_pose, impact_waypoints, direction, angle_axis, impact_point, finger_to_tool


def change_point_finger(point, quat, inverse=False, finger_transform=None, tf_listener=None):
    """
    Change impact point into tool_frame. Or into finger_link if inverse = True
    @param point: Impact point position
    @type point: list, 3x1
    @param quat: Quaternion representing impact direction
    @type quat: list, 4x1
    @param inverse: If to go from tool to finger
    @type inverse: bool
    @param finger_transform: transformation to the finger
    @type finger_transform: list
    @param tf_listener:
    @type tf_listener:
    @return: Point in right frame, transformation between the two frames
    @rtype: 1x3 list of float, 1x3 list of floats
    """

    if inverse:
        (finger_to_tool, rotation) = get_transformation('finger_link', 'tool_frame', tf_listener)
    else:
        finger_to_tool, rotation = finger_transform

    if rotation is None:
        return None, None
    rotation_to_base = ts.quaternion_matrix(quat)

    finger_to_tool = np.matmul(rotation_to_base, np.hstack((finger_to_tool, 1)))[:3]
    point += finger_to_tool
    return point, finger_to_tool


def terminate_thread(thread):
    """Terminates a python thread from another thread.

    :param thread: a threading.Thread instance
    """
    if not thread.isAlive():
        return

    proc = Popen("rosservice call /kinova_mujoco/reset '{}'", stdout=PIPE, shell=True)
    proc.wait()
    rospy.sleep(3)
    exc = ctypes.py_object(SystemExit)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread.ident), exc)
    if res == 0:
        raise ValueError("nonexistent thread id")
    elif res > 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def make_movement(pose, MoveGroup, cartesian=False, joints=False, collisions=True, wait=False, threshold=0.8,
                  speed=0.25):
    """
    Runs proper movement based on arguments
    @param pose: Pose where to move
    @type pose: geometric_msgs.msg.Pose or list of Poses
    @param MoveGroup: MoveGroup python interface instance
    @type MoveGroup: kinova_mujoco.python_interface.MoveGroupPythonInterface
    @param cartesian: If to use linear movement
    @type cartesian: int/bool
    @param joints: whether to use joint angles movement
    @type joints: bool
    @param collisions: Whether to check for collisions
    @type collisions: bool
    @param wait: Whether to wait for completion
    @type wait: bool
    @param threshold: threshold fot he cartesian movement planner
    @type threshold: float
    @param speed: speed to which recompute the plan
    @type speed: float
    @return:
    @rtype:
    """
    if joints:
        MoveGroup.go_to_joint_position(pose)
    elif cartesian:
        plan, frac = MoveGroup.plan_cartesian_path(pose, collisions)
        if frac < threshold:
            return False
        plan = MoveGroup.group.retime_trajectory(MoveGroup.group.get_current_state(), plan,
                                                 velocity_scaling_factor=speed)

        MoveGroup.stop_robot()
        MoveGroup.execute_plan(plan, wait)
        return True
    else:
        MoveGroup.go_to_pose(pose, wait)


def detect_collision(detection_type, threshold, MoveGroup=None, drift=None, joints_idxs=None,
                     topic="/joint_states", free_space=[]):
    """
    Handler for collision detection of multiple types
    @param detection_type: What detection to use: cusum, threshold
    @type detection_type: str
    @param threshold: threshold for robot stopping, based on detection_type
    @type threshold: float
    @param MoveGroup: handle to movegroup commander
    @type MoveGroup: MoveGroup
    @param drift: Drift for cusum method
    @type drift: float
    @param joints_idxs: list with indexes of joint into joint angles message
    @type joints_idxs: list
    @param topic: topic from which to read the joint angles
    @type topic: string
    @param free_space: free space points
    @type free_space: list
    @return:
    @rtype:
    """

    if detection_type == "cusum":  # Cusum detection from internal torques
        class_handle = CUSUM(threshold, drift, joints_idxs)

    start_time = rospy.Time.now().to_time()
    robot_name = rospy.get_param("/robot_name", "")
    tr_exec_topic = '/trajectory_execution_event' if robot_name == "" else (
                "/" + robot_name + '/trajectory_execution_event')
    class_handle.trajectory_event_publisher = rospy.Publisher(tr_exec_topic, String, queue_size=100)
    class_handle.sub = rospy.Subscriber(topic, JointState, class_handle)

    while class_handle.impact_pose is None:  # While contact not occurred
        cur_pose = MoveGroup.get_finger_pose()
        free_space.append([cur_pose.position.x, cur_pose.position.y, cur_pose.position.z])
        if rospy.Time.now().to_time() - start_time > 30:
            class_handle.sub.unregister()
            break
        rospy.sleep(0.25)

    return class_handle.impact_pose


def add_to_PC(pose, pcd_path, r_, rep, shape, touch_to_obj_transform=None, add_pose_estimation=True):
    """
    Function to add new information to the point cloud
    @param pose: Pose of the robot
    @type pose: Pose
    @param pcd_path: path to the .pcd file
    @type pcd_path: string
    @param r_: radius of the circle/square to be added
    @type r_: float
    @param rep: To which rep the touch belongs
    @type rep: int
    @param shape: Which shape to add -- circle, square
    @type shape: string
    @param touch_to_obj_transform: transformation from touch frame to canonical frame
    @type touch_to_obj_transform: numpy array
    @param add_pose_estimation: whether to add new visual information
    @type add_pose_estimation: bool
    @return:
    @rtype:
    """

    R = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    # Load point cloud
    if not os.path.isfile(pcd_path.replace(".pcd", ".npy")):
        pc = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pc.points)
        colors = np.asarray(pc.colors)
    else:
        points = np.load(pcd_path.replace(".pcd", ".npy"))
        colors = np.load(pcd_path.replace(".pcd", "_colors.npy"))

    normals = np.load(pcd_path.replace(".pcd", "_normals.npy"))

    num_new_points = int(np.min([points.shape[0]*0.1, 250])/25)

    # Rotate the points
    points = np.matmul(R, np.hstack((points[:, :3], np.ones((points.shape[0], 1)))).T)[:3, :].T

    voxel_center = np.array([pose.position.x, pose.position.y, pose.position.z])
    touch_to_obj_transform_rotated = R @ touch_to_obj_transform @ R.T

    # Direction vector computation
    direction = (ts.quaternion_matrix([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
                 @ [0, 0, 1, 1])[:3]  # Rotate Z-axis vector to point in direction of gripper
    direction /= np.linalg.norm(direction)  # normalize

    voxel_center = np.matmul(touch_to_obj_transform_rotated, np.hstack((voxel_center, 1)).T)[:3].reshape(
        voxel_center.shape)
    direction = np.matmul(touch_to_obj_transform_rotated, np.hstack((direction, 1)).T)[:3].reshape(direction.shape)

    # perpendicular plane is null of the direction vector
    null_space = scipy.linalg.null_space(direction.reshape(1, 3))
    u = null_space[:, 0]
    v = null_space[:, 1]

    noise_ratio = 0.5
    if shape == "circle":
        # circle = Origin + cos(theta)*u + sin(theta)*v, where theta goes from 0 to 2pi
        th = np.linspace(0, 2 * np.pi, num_new_points).reshape(num_new_points, 1)
        for r in np.linspace(0, r_, 25):
            u_ = u * r
            v_ = v * r
            if r == 0:
                touch = voxel_center + np.multiply(np.cos(th), u_) + np.multiply(np.sin(th), v_) + np.random.uniform(
                    -noise_ratio / 1e3, noise_ratio / 1e3, (th.shape[0], voxel_center.shape[0]))
            else:
                touch = np.vstack((touch, voxel_center + np.multiply(np.cos(th), u_) + np.multiply(np.sin(th),
                                                                                                   v_) + np.random.uniform(
                    -noise_ratio / 1e3, noise_ratio / 1e3, (th.shape[0], voxel_center.shape[0]))))

    elif shape == "square":
        num_points = 30
        r = r_
        xx, yy = np.meshgrid(np.linspace(-r / 2, r / 2, num_points), np.linspace(-r / 2, r / 2, num_points))
        touch = xx.reshape(-1, 1) * u.reshape(1, 3) + yy.reshape(-1, 1) * v.reshape(1,
                                                                                    3) + voxel_center + np.random.uniform(
            -noise_ratio / 1e3, noise_ratio / 1e3, (num_points ** 2, voxel_center.shape[0]))

    # compute new normals
    normals_ = (-1) * direction + np.random.uniform(-noise_ratio / 2, noise_ratio / 2,
                                                    (touch.shape[0], voxel_center.shape[0]))

    if os.path.isfile(pcd_path.replace(".pcd", ".npy")):
        touches = np.load(pcd_path.replace(".pcd", "_touches.npy"))
        touches_normals = np.load(pcd_path.replace(".pcd", "_touches_normals.npy"))
        touches = np.matmul(R, np.hstack((touches[:, :3], np.ones((touches.shape[0], 1)))).T)[:3, :].T
        touches_normals = np.matmul(R, np.hstack((touches_normals[:, :3], np.ones((touches_normals.shape[0], 1)))).T)[:3, :].T

        touches = np.vstack((touches, touch))
        touches_normals = np.vstack((touches_normals, normals_))
    else:
        touches = touch
        touches_normals = normals_

    if not os.path.exists(os.path.join(os.path.dirname(pcd_path), "rep" + str(rep))):
        os.makedirs(os.path.join(os.path.dirname(pcd_path), "rep" + str(rep)))

    files = glob.glob(os.path.join(os.path.dirname(pcd_path), "*"))
    for file in files:
        if not os.path.isdir(file):
            cmd = "cp " + file + " " + os.path.join(os.path.dirname(pcd_path), "rep" + str(rep), file.split("/")[-1])
            proc = Popen(cmd, stdout=PIPE, shell=True)
            proc.wait()

    if add_pose_estimation:
        pose_estimation_pc = o3d.io.read_point_cloud(pcd_path.replace(".pcd", "_pose.pcd"))
        pe_p = np.asarray(pose_estimation_pc.points)
        pe_n = np.asarray(pose_estimation_pc.normals)
        pose_estimation_points = (R @ touch_to_obj_transform @ np.hstack((pe_p, np.ones((pe_p.shape[0], 1)))).T)[:3, :].T
        pose_estimation_normals = (R @ touch_to_obj_transform @ np.hstack((pe_n, np.ones((pe_n.shape[0], 1)))).T)[:3, :].T

        normals = np.matmul(R, np.hstack((normals[:, :3], np.ones((normals.shape[0], 1)))).T)[:3, :].T

        normals = np.vstack((normals, pose_estimation_normals))
        points = np.vstack((points, pose_estimation_points))
        colors = np.vstack((colors, np.asarray(pose_estimation_pc.colors)))

        new_pc = o3d.geometry.PointCloud()
        new_pc.points = o3d.utility.Vector3dVector(points)
        new_pc.normals = o3d.utility.Vector3dVector(normals)
        new_pc.colors = o3d.utility.Vector3dVector(colors)

        new_pc = new_pc.voxel_down_sample(0.0025)
        if not new_pc.has_normals():
            new_pc.estimate_normals()
            new_pc.orient_normals_consistent_tangent_plane(k=10)
        normals = np.asarray(new_pc.normals)
        points = np.asarray(new_pc.points)
        colors = np.asarray(new_pc.colors)

    normals = np.matmul(R.T, np.hstack((normals[:, :3], np.ones((normals.shape[0], 1)))).T)[:3, :].T
    touches_normals = np.matmul(R.T, np.hstack((touches_normals[:, :3], np.ones((touches_normals.shape[0], 1)))).T)[:3, :].T

    np.save(pcd_path.replace(".pcd", "_normals.npy"), normals)
    np.save(pcd_path.replace(".pcd", ".npy"), np.matmul(R.T, np.hstack((points[:, :3], np.ones((points.shape[0], 1)))).T)[:3, :].T)
    np.save(pcd_path.replace(".pcd", "_colors.npy"), colors)
    np.save(pcd_path.replace(".pcd", "_touches.npy"), np.matmul(R.T, np.hstack((touches[:, :3], np.ones((touches.shape[0], 1)))).T)[:3, :].T)
    np.save(pcd_path.replace(".pcd", "_touches_normals"), touches_normals)

    points = np.vstack((points, touches))
    normals = np.vstack((normals, touches_normals))
    colors = np.vstack((colors, np.tile([1, 0, 0], (touches.shape[0], 1))))

    points_rotated = (R @ ts.inverse_matrix(touch_to_obj_transform) @ R.T
                      @ np.hstack((points[:, :3], np.ones((points.shape[0], 1)))).T)[:3, :].T
    np.save(pcd_path.replace(".pcd", "_rotated.npy"), points_rotated)

    points = np.matmul(R.T, np.hstack((points[:, :3], np.ones((points.shape[0], 1)))).T)[:3, :].T

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)
    pc.normals = o3d.utility.Vector3dVector(normals)

    o3d.io.write_point_cloud(pcd_path, pc)


def move_files(mesh_path, npy_path, rec_id):
    """
    Move old files into corresponding folder
    @param mesh_path: path with meshes
    @type mesh_path: string
    @param npy_path: path with npy files
    @type npy_path: strig
    @param rec_id: reconstruction number
    @type rec_id: int
    @return:
    @rtype:
    """

    rospy.loginfo("Moving files into rep" + str(rec_id) + " folder")
    for path in [mesh_path, npy_path]:
        files = glob.glob(os.path.join(path, "*"))
        if not os.path.exists(os.path.join(path, "rep" + str(rec_id))):
            os.makedirs(os.path.join(path, "rep" + str(rec_id)))
        for file in files:
            if not os.path.isdir(file) and "free_space" not in file:
                cmd = "cp " + file + " " + os.path.join(path, "rep" + str(rec_id), file.split("/")[-1])
                proc = call(cmd, stdout=PIPE, shell=True)


def get_transformation(what, where, tf_listener):
    """
    Help util to get transformation
    @param what: For what frame to obtain the transformation
    @type what: string
    @param where: In which frame to express
    @type where: string
    @param tf_listener: Transformation lister
    @type tf_listener: tf_listener
    @return:
    @rtype:
    """
    i = 0
    while True:
        try:
            translation, rotation = tf_listener.lookupTransform(where, what, rospy.Time(0))
            return np.array(translation), np.array(rotation)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            i += 1
            if i > 1e2:
                return None, None
            rospy.sleep(0.01)
            continue


def create_mesh_bbox(MoveGroupArm, center, path):
    """
    Create bounding box for the obejcts
    @param MoveGroupArm: MoveGroup handle
    @type MoveGroupArm: MoveGroup
    @param center: center of the object
    @type center: list
    @param path: path to the meshes
    @type path: string
    @return:
    @rtype:
    """

    # Prepare pose message
    ps = Pose()
    ps.position.x = center[0]
    ps.position.y = center[1]
    ps.position.z = center[2]
    ps.orientation.w = 1.0

    # get all files
    paths = glob.glob(os.path.join(path, "*bbox.ply"))

    # publish them
    used = []
    MoveGroupArm.planning_scene.clear()
    for path_id, path in enumerate(paths):
        obj_name = os.path.basename(os.path.normpath(path)).split(".ply")[0]
        if obj_name not in used:
            MoveGroupArm.planning_scene.addMesh(obj_name, ps, path, use_service=True)
            MoveGroupArm.planning_scene.setColor(obj_name, 1, 1, 1, 0.1)
            used.append(obj_name)

    MoveGroupArm.planning_scene.sendColors()


def remove_mesh_bbox(MoveGroupArm, to_remove):
    """
    Remove bounding box for the given object for exploration
    @param MoveGroupArm: MoveGroup handle
    @type MoveGroupArm: MoveGroup
    @param to_remove: name of the object to be removed
    @type to_remove: string
    @return:
    @rtype:
    """
    MoveGroupArm.planning_scene.removeCollisionObject(to_remove, use_service=True)
    MoveGroupArm.planning_scene.waitForSync()


def compute_pose(pcd_path, paths_new, paths_old, finger_pose=None, R=None, simulation=False,
                 classes=[], bboxes=[], objects_names=[], old_transforms=[]):
    """
    Function to compute pose from point clouds
    @param pcd_path: path where to save point clouds
    @type pcd_path: string
    @param paths_new: path to the new point clouds
    @type paths_new: list of strings
    @param paths_old: path to the old point clouds
    @type paths_old: list of string
    @param finger_pose: pose of the finger
    @type finger_pose: list
    @param R: transformation between RVIZ and real world
    @type R: numpy array
    @param simulation: whether we are in simulation
    @type simulation: bool
    @param classes: classes for detection functions
    @type classes: list
    @param bboxes: bounding boxes for detection functions
    @type bboxes: list
    @param objects_names: names of the objects
    @type objects_names: list
    @param old_transforms: old transformations
    @type old_transforms: list of numpy arrays
    @return:
    @rtype:
    """

    # Save point cloud of current scene
    _, classes, bboxes = save_pcl(pcd_path, pose_estimation=True, simulation=simulation,
                                  classes=classes, bboxes=bboxes, objects_names=objects_names)

    transformations = []
    for path_old, path_new, old_transform in zip(paths_old, paths_new, old_transforms):
        # Load point cloud
        if ".npy" in path_old:
            old = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.load(path_old)))
        else:
            old = o3d.io.read_point_cloud(path_old)

        if ".npy" in path_new:
            new = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.load(path_new)))
        else:
            new = o3d.io.read_point_cloud(path_new)

        if finger_pose is not None:
            # Direction vector computation
            normal = np.matmul(ts.quaternion_matrix([finger_pose.orientation.x, finger_pose.orientation.y,
                                                     finger_pose.orientation.z, finger_pose.orientation.w]), [0, 0, 1, 1])[:3]
            normal /= np.linalg.norm(normal)  # normalize

            # perpendicular plane is null of the direction vector
            normal = (R @ np.hstack((normal, 1)).T)[:3]
            finger_pose_ = (R @ np.hstack(([finger_pose.position.x, finger_pose.position.y, finger_pose.position.z], 1)).T)[
                          :3]

            plane_signs = np.sign(normal @ (np.asarray(old.points)-finger_pose_).T)
            old = old.select_by_index(np.where(plane_signs == 1)[0])

            if np.asarray(old.points).size:
                old.estimate_normals()
                old.orient_normals_consistent_tangent_plane(k=10)
                o3d.io.write_point_cloud(path_old, old)

        old.transform(old_transform)

        # Do ICP
        res = o3d.pipelines.registration.registration_icp(old, new, 0.05, np.eye(4),
                                                          o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                          o3d.pipelines.registration.ICPConvergenceCriteria(
                                                              max_iteration=100, relative_rmse=1e-10,
                                                              relative_fitness=1e-10))

        transformations.append(res.transformation @ old_transform)
    return transformations, classes, bboxes


def get_impact_pose(impact=False, MoveGroupArm=None, forward_kinematics=None, joint_angles=[],
                    joints_idxs=[], tf_listener=None):
    """
    Compute pose of the finger
    @param impact: whether to compute after impact
    @type impact: bool
    @param MoveGroupArm: handle to MoveGroup of the arm
    @type MoveGroupArm: MoveGroup
    @param forward_kinematics: forward kinematics interface
    @type forward_kinematics: FK
    @param joint_angles: joint angles during impact
    @type joint_angles: list
    @param joints_idxs: indexes of the right joints
    @type joints_idxs: list
    @param tf_listener: transformation listener
    @type tf_listener: tf_listener
    @return:
    @rtype:
    """
    if impact:
        pose = forward_kinematics.getFK("tool_frame", ["joint_" + str(i) for i in range(1, 8)],
                                        joint_angles[joints_idxs]).pose_stamped[0].pose
        pose_finger, _ = change_point_finger(
            [pose.position.x, pose.position.y, pose.position.z],
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
            True, tf_listener=tf_listener)
        pose_finger = Pose(Point(*pose_finger), pose.orientation)
    else:
        pose_finger = MoveGroupArm.get_finger_pose()

    return pose_finger


def check_fall(get_all_poses, MoveGroupGripper):
    """
    Check whether some of the objects fallen
    @param get_all_poses: handle to service
    @type get_all_poses: GetAllPoses
    @param MoveGroupGripper: handle to Gripper move group
    @type MoveGroupGripper: MoveGroup
    @return:
    @rtype:
    """
    # Check for fallen object and restart the simulation if necessary
    poses = get_all_poses.call(GetAllObjectPosesRequest())
    for name, pose in zip(poses.names, poses.poses):
        pose = pose.pose
        rot = np.rad2deg(ts.euler_from_quaternion(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]))
        if np.abs(rot[0]) > 30 or np.abs(rot[1]) > 30:
            proc = Popen("rosservice call /kinova_mujoco/reset '{}'", stdout=PIPE,
                         shell=True)
            proc.wait()
            rospy.sleep(5)
            while True:
                gripper_cur_joints = MoveGroupGripper.group.get_current_joint_values()
                # The simulation sometimes glitched and run robot with open gripper -> restart simulation if it happens
                gripper_joints = [0.80, -0.79, 0.82, 0.83, 0.80, -0.80]

                if np.linalg.norm(
                        np.array(gripper_cur_joints) - np.array(gripper_joints)) > 0.1:
                    proc = Popen("rosservice call /kinova_mujoco/reset '{}'", stdout=PIPE,
                                 shell=True)
                    proc.wait()
                    rospy.sleep(5)
                else:
                    break
            return True
    return False
