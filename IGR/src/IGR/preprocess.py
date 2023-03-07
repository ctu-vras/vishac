"""
Main function to preprocess file for reconstruction

@author Lukas Rustler
"""
import numpy as np
import os
import argparse
import glob
import open3d
from scipy.linalg import lstsq, null_space


def preprocess(input_file, out_path, normals_action):
    """
    Takes inpus point cloud, scales it, transformes it into (0,0,0) origin and computes normals
    @param input_file: path to .pcd file
    @type input_file: str
    @param out_path: Path where to save .npy files
    @type out_path: str
    @param translation: How much to translate
    @type translation: list
    @return:
    @rtype:
    """

    # get file name
    file_name = os.path.basename(os.path.normpath(input_file))

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    pc = open3d.io.read_point_cloud(input_file)
    if normals_action == "save":
        # load point cloud and gets its center
        center = pc.get_center()

        # translate it into (0,0,0)
        pc.translate([0, 0, 0], False)

        # find the scale factor (we want the object to fit into an ellipse with bigger radius of 1)
        ma = np.max(np.asarray(pc.points), 0)
        mi = np.min(np.asarray(pc.points), 0)
        diff = np.max(np.abs(ma - mi))
        scale = 2 / diff

        # scale the object and find normals
        pc.scale(scale, [0, 0, 0])
        pc.estimate_normals()
        pc.orient_normals_consistent_tangent_plane(k=10)  # turn all normals consistently (hopefully outside)

        np.save(os.path.join(os.path.dirname(input_file), file_name.split(".pcd")[0]) + '_normals.npy', np.array(pc.normals))

    else:
        rep_path = [_ for _ in sorted(glob.glob(os.path.join(out_path, "*")), key=os.path.getmtime) if "rep" in _]
        if len(rep_path) == 0:
            rep_path = out_path
        else:
            rep_path = rep_path[0]
        center = np.load(os.path.join(rep_path, file_name.split(".pcd")[0]) + '_center.npy')
        scale = np.load(os.path.join(rep_path, file_name.split(".pcd")[0]) + '_scale.npy')
        pc.translate(-center, True)
        pc.points = open3d.utility.Vector3dVector(np.asarray(pc.points)*scale)
        # open3d.visualization.draw_geometries([pc], point_show_normal=True)
    pc.normalize_normals()
    # open3d.visualization.draw_geometries([pc], point_show_normal=True)

    # save datapoints + normals, scale factor and center
    data = np.hstack([np.asarray(pc.points), np.array(pc.normals)])
    np.save(os.path.join(out_path, file_name.split(".pcd")[0]) + '.npy', data)
    np.save(os.path.join(out_path, file_name.split(".pcd")[0]) + '_scale.npy', scale)
    np.save(os.path.join(out_path, file_name.split(".pcd")[0]) + '_center.npy', center)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Preprocessing of point clouds for shape completion"
    )

    arg_parser.add_argument(
        "--out_path",
        "-o",
        dest="out_path",
        required=False,
        default="~/work/shape_detection/main_ws/src/shape_completion/data/npy"
    )

    arg_parser.add_argument(
        "--file",
        "-f",
        dest="file",
        required=True,
        help=".pcd file with point cloud"
    )

    arg_parser.add_argument(
        "--normals_action",
        "-na",
        dest="normals_action",
        required=False,
        default="save",
        help="Save normals"
    )

    args = arg_parser.parse_args()
    preprocess(args.file, args.out_path, args.normals_action)
