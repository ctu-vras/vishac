#!/usr/bin/env python3
"""
Script for easier experiment running
@author Lukas Rustler
"""
from subprocess import Popen, call
import json
import datetime
import os
import time
import argparse
import sys
import signal
import rospy

def bool_to_str(bool_):
    return "true" if bool_ else "false"


def signal_handler(signal, frame):
    """
    Handler for killing the program
    @param signal: signal type
    @type signal: signal
    @param frame:
    @type frame:
    @return:
    @rtype:
    """
    print("Killing on user request")
    for _ in os.popen("pgrep -f main.py").read().strip().splitlines():
        call("kill -9 "+_, shell=True)
    sys.exit(0)


def prepare_parser():
    """
    Reads argument and return them in correct form
    @return:
    @rtype:
    """
    arg_parser = argparse.ArgumentParser(
        description="Experimentator"
    )
    arg_parser.add_argument(
        "--setup_file",
        "-s",
        dest="setup_file_name",
        required=True,
        help="Path to setup_file_name"
    )
    arg_parser.add_argument(
        "--Logs output folder",
        "-l",
        dest="logs_folder",
        required=False,
        default=None,
        help="Folder where to save experiments logs"
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
        "--interactive",
        "-i",
        dest="interactive",
        action="store_true",
        required=False,
        default=False,
        help="When interactive, RVIZ is started and user is prompted if everything runs fine"
    )

    arg_parser.add_argument(
        "--publish",
        "-p",
        dest="publish",
        action="store_true",
        required=False,
        default=False,
        help="if to publish new joint_states msg with external torques"
    )

    arg_parser.add_argument(
        "--real_setup",
        "-rs",
        dest="real_setup",
        action="store_true",
        required=False,
        default=False,
        help="If to run real setup"
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
        "--fixed_objects",
        "-fo",
        dest="fixed_objects",
        action="store_true",
        required=False,
        default=False,
        help="Whether to use fixed objects"
    )

    arg_parser.add_argument(
        "--no_terminal",
        "-nt",
        dest="terminal",
        action="store_true",
        required=False,
        default=False,
        help="Whether to use terminal version"
    )

    args = arg_parser.parse_args()
    return args.setup_file_name, args.logs_folder, args.detection_type, args.interactive, args.publish, args.real_setup,\
           args.net, bool_to_str(not args.fixed_objects), not args.terminal


if __name__ == "__main__":
    # Kill ros after CTRL+C signal
    signal.signal(signal.SIGINT, signal_handler)

    # Open setup file
    setup_file_name, logs_folder, detection_type, interactive, publish, real_setup, net, free_objects, terminal = prepare_parser()

    if not terminal:
        from main import main

    with open(setup_file_name, "r") as setup_file:
        exp_setup = json.load(setup_file)

    # If logs_folder not specified, create one besides the setups directory
    if logs_folder is None:
        logs_folder = os.path.join(os.path.dirname(os.path.dirname(setup_file_name)), 'logs')

    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    # Help variables
    folders = ["meshes", "pcd", "npy", "npz", "plots", "rosbags"]
    file_dir = os.path.dirname(os.path.abspath(__file__))
    start_time_ = datetime.datetime.now()
    start_time = str(start_time_).replace(".", "-").replace(" ", "-").replace(":", "-")
    filename = os.path.join(logs_folder, start_time+".log")
    # Write header into .log file
    log_file = open(filename, "w")
    log_file.write("timestamp;objects;(number of reconstructions,max_time,number of repetitions);repetition;time\n")
    log_file.close()

    FPS = "true"
    free_space = "true"
    num_of_points = "10"
    level = "0"

    for objects, origins, rec_num, max_time, repetitions in zip(exp_setup["objects"], exp_setup["origins"], exp_setup["reconstructions"], exp_setup["max_time"], exp_setup["repetitions"]):
        for repetition in range(repetitions):
            if not real_setup:
                # Run RVIZ, mujoco etc.
                cmd = "run_simulation "+",".join(objects)+" '"+str(origins)+"' 'false' 'true' "+free_objects+" "+FPS+" "+free_space+" "+num_of_points+" "+level
            else:
                cmd = "roslaunch shape_completion real.launch printed_finger:='false' object_name:='" + ",".join(objects) + "'"
            print(cmd)
            rviz = Popen(cmd, shell=True)#, stdout=PIPE)

            if not interactive:
                # Wait long enough to give everything time to boot
                time.sleep(15)
            else:
                # Wait for user input confirming that everything is okay
                while True:
                    answer = input("Everything fine?")
                    if answer.lower() in ["y", "yes"]:
                        break
                    else:
                        call("pkill -f ros", shell=True)
                        time.sleep(15)
                        rviz = Popen(cmd, shell=True)  # , stdout=PIPE)

            start_time_main_ = datetime.datetime.now()
            start_time_main = str(start_time_main_).replace(".", "-").replace(" ", "-").replace(":", "-")
            rospy.set_param("/actvh/timestamp", start_time_main)

            if terminal:
                # Run the experiment and wait for end
                cmd = "rosrun shape_completion main.py -r " + str(rec_num) +" -d " + detection_type +" -n "+net+\
                      " -t "+str(max_time)
                if publish:
                    cmd += " -p"
                main_loop = call(cmd, shell=True)
            else:
                main(rec_num+1, detection_type, publish, net, max_time)

            # Kill all ros related to have clean starting point next iteration
            call("pkill -f ros", shell=True)

            # Move all files into timestamped folders
            timestamp_ = datetime.datetime.now()
            timestamp = str(timestamp_).replace(".", "-").replace(" ", "-").replace(":", "-")
            time.sleep(15)

            # Save info into .log file
            log_file = open(filename, "a")
            new_exps = start_time_main+";"+",".join(objects)+";(" + str(rec_num) + "," + str(max_time) + "," + str(repetitions)+");" + \
                       str(repetition+1) + ";" + str((timestamp_ - start_time_main_).total_seconds()) + "\n"
            log_file.write(new_exps)
            log_file.close()

    # Evaluate
    cmd = "rosrun shape_completion evaluation_utils.py " + start_time + ".log"
    call(cmd, shell=True)
