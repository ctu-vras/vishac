#!/usr/bin/env python3
"""
Utils for evaluation of the results

@author Lukas Rustler
"""
import sys

from shape_completion.distance_utils import jaccard_similarity, chamfer_distance, compute_binvox
import os
import numpy as np
import glob
import open3d as o3d
import pickle as pkl


def compute_similarity(mesh_path, gt_path, voxel_res=40, num_objects=1, visualization=False):
    """
    Help function to compute Jaccard similarity and Chamfer distance
    @param mesh_path: path to the folder with meshes for given experiment
    @type mesh_path: string
    @param gt_path: bath to the folder with GT meshes
    @type gt_path: string
    @param voxel_res: resolution of meshes to use
    @type voxel_res: int
    @param num_objects: number of objects in the experiment
    @type num_objects: int
    @param visualization: whether to display debug visualization
    @type visualization: bool
    @return: jaccard similarity, chamfer distance and order of objects
    @rtype: list, list, list
    """
    reps = sorted([_.split("rep")[1] for _ in glob.glob(os.path.join(mesh_path, "rep*"))], key=lambda x: int(x))
    chamfer = [[] for _ in range(num_objects)]
    jaccard = [[] for _ in range(num_objects)]
    object_order = []

    for rep in reps:
        # find all objects
        objects = glob.glob(os.path.join(mesh_path, "rep" + rep, "*_eikonal.npy"))

        # fix the objects order in the first iteration to be the same
        if len(object_order) == 0:
            object_order = [os.path.basename(os.path.normpath(obj)).split("_eikonal.npy")[0] for obj in objects]

        for obj in objects:
            if not os.path.isfile(obj.replace("_eikonal.npy", ".ply")):
                continue
            obj_name = os.path.basename(os.path.normpath(obj)).split("_eikonal.npy")[0]
            obj_id = object_order.index(obj_name)

            # load scale, GT mesh and create GT point cloud
            scale = np.load(obj.replace("meshes", "npy").replace("_eikonal", "_scale"))
            gt_mesh = o3d.io.read_triangle_mesh(os.path.join(gt_path, obj_name.split("__")[0]+".stl"))
            gt = gt_mesh.sample_points_uniformly(10000)

            # load reconstruction mesh and make reconstruction point cloud
            rec_mesh = o3d.io.read_triangle_mesh(obj.replace("_eikonal.npy", ".ply"))
            rec_mesh.scale(1 / scale, [0, 0, 0])
            rec = rec_mesh.sample_points_uniformly(10000)

            # estimate normals and get points
            rec.estimate_normals()
            gt.estimate_normals()
            rec_points = np.asarray(rec.points)
            gt_points = np.asarray(gt.points)

            # find all point in the first half of the objects -> more robust ICP can be applied then
            rec_half = rec.select_by_index(np.arange(0, rec_points.shape[0])[rec_points[:, 2] > rec.get_center()[2]])
            gt_half = gt.select_by_index(np.arange(0, gt_points.shape[0])[gt_points[:, 2] > gt.get_center()[2]])

            # compute ICP registration
            res = o3d.pipelines.registration.registration_icp(rec_half, gt_half, 1, np.eye(4),
                                                              o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                              o3d.pipelines.registration.ICPConvergenceCriteria(
                                                                  max_iteration=1000, relative_rmse=1e-10,
                                                                  relative_fitness=1e-10))

            # get the point back
            rec = rec.transform(res.transformation)
            rec_points = np.asarray(rec.points)
            gt_points = np.asarray(gt.points)

            # compute chamfer
            chamfer[obj_id].append(chamfer_distance(rec_points, gt_points)*1e3)

            # compute jaccard
            rec_mesh.transform(res.transformation)
            o3d.io.write_triangle_mesh(obj.replace("_eikonal.npy", "_binvox.ply"), rec_mesh)
            data = []
            for path_id, path in enumerate([os.path.join(gt_path, obj_name.split("__")[0]+".ply"), obj.replace("_eikonal.npy", "_binvox.ply")]):
                data.append(compute_binvox(path, voxel_res, gt=path_id == 0))

            data = np.stack(data)
            jaccard[obj_id].append(jaccard_similarity(data[0, :, :, :], data[1, :, :, :])*1e2)

            # debug visualization
            if visualization:
                o3d.visualization.draw_geometries([gt.paint_uniform_color([1, 0, 0]),
                                                   rec.paint_uniform_color([0, 0, 1])], point_show_normal=True)

    return jaccard, chamfer, object_order


def evaluate_log(log_path, gt_path):
    """
    Help function to evaluate every experiment in log
    @param log_path: path to the log
    @type log_path: string
    @param gt_path: path to GT meshes
    @type gt_path: string
    @return: None
    @rtype:
    """

    # Load log and split by ";"
    with open(log_path, "r") as f:
        log = [_.split(";") for _ in f.read().splitlines()[1:]]

    jaccard = {}
    chamfer = {}
    objects_names = {}

    for line in log:
        objects = line[1]
        print(f"Processing folder {line[0]}")
        if objects not in jaccard.keys():
            jaccard[objects] = []
            chamfer[objects] = []
            objects_names[objects] = []

        # call similarity function
        jac, cham, obj_names = compute_similarity(os.path.normpath(os.path.join(log_path, "../../meshes", line[0])),
                                                  gt_path, num_objects=len(objects.split(",")))

        # append to temporary arrays
        jaccard[objects].append(jac)
        chamfer[objects].append(cham)
        objects_names[objects].append(obj_names)

    # prepare folder for evaluation data
    log_name = os.path.basename(os.path.normpath(log_path)).split(".log")[0]
    if not os.path.exists(os.path.normpath(os.path.join(log_path, "../../evaluation"))):
        os.makedirs(os.path.normpath(os.path.join(log_path, "../../evaluation")))

    # save data to pickle file
    with open(os.path.normpath(os.path.join(log_path, "../../evaluation", log_name+".pkl")), "wb") as f:
        pkl.dump({"chamfer": chamfer, "jaccard": jaccard,
                  "objects_names": objects_names}, f)


if __name__ == "__main__":
    """Prepared calls for easier use of this files"""
    file_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) > 1:
        log = sys.argv[1]
    else:
        log = "simulated.log"

    log_path = os.path.join(file_dir, "../../data/logs", log)
    gt_path = os.path.join(file_dir, "../../../kinova_mujoco/GT_meshes")
    evaluate_log(log_path, gt_path)
