#!/usr/bin/env python3
"""
Main function to prepare the object in simulation

@author Lukas Rustler
"""

from kinova_mujoco.utils import create_object_scene, prepare_urdf, bool_to_str, str_to_bool
import os
import argparse


def parse_arguments():
    arg_parser = argparse.ArgumentParser(
        description="Script for object preparation for the simulation"
    )
    arg_parser.add_argument(
        "--objects_names",
        "-n",
        dest="objects_names",
        required=True,
        help="list of the names of the .stl of the object"
    )

    arg_parser.add_argument(
        "--origins",
        "-o",
        dest="origins",
        required=False,
        default=[[0.6, 0, 0.2]],
        help="Positions of the objects in space"
    )

    arg_parser.add_argument(
        "--printed_finger",
        "-f",
        dest="printed_finger",
        required=False,
        default='false',
        help="If to use printed finger"
    )

    arg_parser.add_argument(
        "--convex_decomp",
        "-c",
        dest="convex_decomp",
        required=False,
        default='false',
        help="If to use convex decomposition"
    )

    arg_parser.add_argument(
        "--mujoco",
        "-m",
        dest="mujoco",
        required=False,
        default='false',
        help="If to use object in mujoco representation"
    )

    args = arg_parser.parse_args()

    return [obj for obj in args.objects_names.split(",")], eval(args.origins), args.printed_finger, str_to_bool(args.convex_decomp), str_to_bool(args.mujoco)


if __name__ == "__main__":

    objects, origins, printed_finger, convex_decomp, mujoco = parse_arguments()
    _, link_names = create_object_scene(objects, mujoco=mujoco, convex_decomp=convex_decomp, origins=origins)

    # if objects are fixed, prepare default URDF
    if not mujoco:
        out_obj_urdf = "<robot>"
        obj_names = []
        with open(os.path.join(os.path.dirname(__file__), '../urdf/object_default.urdf'), "r") as f:
            def_config = f.read()
        for obj, origin in zip(objects, origins):
            obj_name = (obj + "_static") if (obj + "_static") not in obj_names else (obj + "_" + str(obj_names.count(obj)) + "_static")
            obj_visual_name = obj_name + "_visual"
            jnt_name = obj_name + "_visual_joint"
            wrd_jnt_name = obj_name + "_world_joint"
            new_conf = def_config.replace("ORIGIN", " ".join(map(str, origin))).\
                replace("OBJ_NAME_VISUAL", obj_visual_name).\
                replace("OBJ_NAME", obj_name).\
                replace("MESH_NAME", obj). \
                replace("WORLD_JOINT_NAME", wrd_jnt_name).\
                replace("JOINT_NAME", jnt_name)
            out_obj_urdf += "\n" + new_conf
            obj_names.append(obj_name)

        out_obj_urdf += "\n</robot>"
        with open(os.path.join(os.path.dirname(__file__), '../urdf/object.urdf'), "w") as f:
            f.write(out_obj_urdf)

    # prepare the whole urdf
    prepare_urdf(printed_finger, bool_to_str(convex_decomp), mujoco)

    # Disable collision with the objects in SRDF, so the robot is able to touch them
    links = ["end_effector_link", "spherical_wrist_2_link", "spherical_wrist_1_link", "shoulder_link",
             "robotiq_arg2f_base_link", "right_outer_knuckle", "right_outer_finger", "right_inner_knuckle",
             "right_inner_finger_pad", "right_inner_finger", "left_outer_knuckle", "left_outer_finger",
             "left_inner_knuckle", "left_inner_finger_pad", "left_inner_finger", "half_arm_2_link",
             "half_arm_1_link", "forearm_link", "bracelet_link", "base_link"]
    line = '<disable_collisions link1="$(arg prefix)LINK1" link2="LINK2" reason="I want to"/>'
    srdf = open(os.path.join(os.path.dirname(__file__), '../../kinova_description/gen3_robotiq_2f_85_move_it_config/config/7dof/gen3_robotiq_2f_85.srdf.xacro'), "r")
    srdf = srdf.read().splitlines()
    srdf.pop()
    if not mujoco:
        link_names = obj_names + [obj+"_visual" for obj in obj_names]
    for link_name in link_names:
        for robot_link in links:
            line_temp = line.replace("LINK1", robot_link).replace("LINK2", link_name)
            srdf.append(line_temp)
    srdf.append("</robot>")
    with open(os.path.join(os.path.dirname(__file__), '../../kinova_description/gen3_robotiq_2f_85_move_it_config/config/7dof/mujoco_gen3_robotiq_2f_85.srdf.xacro'), "w") as new_srdf:
        for line in srdf:
            new_srdf.write(line+"\n")
