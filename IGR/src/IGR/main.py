#!/usr/bin/env python3
"""
Main function for IGT usage in VISHAC 2023.

@author Lukas Rustler, using function from IGR by Gropp, 2020
"""

import argparse
import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)
import json
import IGR.general as utils
import torch
from pyhocon import ConfigFactory
import rospy
import IGR.plots as plt
from IGR.sample import Sampler
from IGR.network import gradient
import time
import numpy as np
from scipy.spatial.distance import cdist
from pymeshfix import _meshfix
from igr.msg import igr_request
from queue import SimpleQueue
import multiprocessing


def get_bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ["true", "t", "1"]:
        return True
    return False


class Shape:
    """
    Class to represent shape with all its parameters
    """
    def __init__(self, id, name, igr_handle):
        self.id = id
        self.igr_handle = igr_handle
        self.name = name
        latent_size = igr_handle.conf.get_int('train.latent_size')
        self.latent = torch.ones(latent_size).normal_(0, 1 / latent_size)
        if torch.cuda.is_available():
            self.latent = self.latent.cuda()
        self.latent.requires_grad = True
        self.optimizer = torch.optim.Adam([self.latent], lr=self.igr_handle.lr)
        self.points = []
        self.normals = []
        self.first_done = False
        self.max_iterations = -1
        self.current_iteration = 0
        self.total_iterations = 0
        self.iterations_since_save = 0
        self.distances = []
        self.idxs = []
        self.N = 0
        self.number_new_points = 0
        self.filename = os.path.join(self.igr_handle.save_directory, self.name)
        self.scale = np.load(self.filename.replace("meshes", "npy")+"_scale.npy")
        self.center = np.load(self.filename.replace("meshes", "npy")+"_center.npy")
        self.eikonal_path = self.filename+"_eikonal.npy"
        self.mesh = None
        self.eikonal = None


class IGR:
    """
    Class to represent IGR module, with all parameters and function
    """
    def __init__(self, points=1000, free_space=False, level=0, fps=False):
        self.points = points
        self.free_space = free_space
        self.level = level
        self.fps = fps
        self.resolution = 40
        self.shapes = []
        self.shape_ids = []
        self.requested_ids = SimpleQueue()
        self.current_shape = None
        self.free_space_points = np.array([])
        self.shapes_completed = 0
        self.dataset = None
        self.meshes_ready = []
        self.shape_requested = False
        self.lr = 5e-3
        self.lock = multiprocessing.Lock()
        self.sub = rospy.Subscriber("/IGR/request_topic", igr_request, self.get_request)
        self.network, self.conf, self.save_directory, self.split_file = \
            self.prepare("no_ycb", "3500",  "conf.json", "shape_completion_setup.conf", "exps", "0")
        self.run()

    def get_request(self, msg):
        """
        Function to parse request for shape
        @param msg: msg with request
        @type msg: igr.msg.igr_request
        @return:
        @rtype:
        """

        self.lock.acquire() # Lock is needed for safe threading
        rospy.logerr(f"{msg}, {self.current_shape}, {self.meshes_ready}")
        if msg.shape_id != -1:
            self.requested_ids.put(msg.shape_id)

        if self.dataset is None:  # If first shape is requested, create dataset
            with open(self.split_file, "r") as f:
                split = json.load(f)
            self.dataset = utils.get_class(self.conf.get_string('train.dataset'))(split=split,
                                                                                  dataset_path=self.conf.get_string('train.dataset_path'),
                                                                                  with_normals=True)
        # Load new data from file and assign them to the correct Shape
        if msg.new_data:
            pc, normals, index = self.dataset[msg.shape_id]
            if torch.cuda.is_available():
                pc = pc.cuda().squeeze()
                normals = normals.cuda().squeeze()
            else:
                pc = pc.squeeze()
                normals = normals.squeeze()

            if msg.shape_id not in self.shape_ids:
                self.shapes.append(Shape(msg.shape_id, msg.shape_name, self))
                self.meshes_ready.append(False)
                self.shape_ids.append(msg.shape_id)

            shape_id = self.shape_ids.index(msg.shape_id)

            self.shapes[shape_id].points = pc
            self.shapes[shape_id].normals = normals
            self.shapes[shape_id].max_iterations = 200 if not self.shapes[shape_id].first_done else 100
            self.shapes[shape_id].current_iteration = 0

            if self.fps and self.points != -1:
                points_cpu = pc.cpu().detach().numpy()
                self.shapes[shape_id].distances = cdist(points_cpu, points_cpu, 'euclidean')
                N = points_cpu.shape[0]
                self.shapes[shape_id].number_new_points += N - self.shapes[shape_id].N
                self.shapes[shape_id].N = N
                self.shapes[shape_id].idxs = np.zeros((min(self.points, self.shapes[shape_id].N),))

        # Set-up new shape
        if self.current_shape is None and not self.requested_ids.empty():
            shape_id = self.shape_ids.index(self.requested_ids.get())
            self.current_shape = self.shapes[shape_id]

            if self.free_space:
                free_space_points = np.array(rospy.get_param("/IGR/free_space/"+self.current_shape.name, [])).reshape((-1, 3))
                free_space_points -= self.shapes[shape_id].center
                free_space_points *= self.shapes[shape_id].scale
                self.free_space_points = torch.tensor(free_space_points).float()
                if torch.cuda.is_available():
                    self.free_space_points = self.free_space_points.cuda()

            rospy.loginfo(f"IGR: Got request for shape {self.current_shape.name}")

        if msg.request:
            self.shape_requested = True

        self.lock.release()

    @staticmethod
    def prepare(exp_name, epoch="latest", split="conf.json", conf="shape_completion_setup.conf", exps_dir="exps",
                gpu_num="0"):
        """
        Function to prepare the network to run
        @param exp_name: name of the trained network
        @type exp_name: string
        @param epoch: epoch of the network to use
        @type epoch: string
        @param split: name of the split file
        @type split: string
        @param conf: name of the config file
        @type conf: string
        @param exps_dir: name of the dir with experimnt
        @type exps_dir: string
        @param gpu_num: gpu number to use
        @type gpu_num: string/int
        @return:
        @rtype:
        """
        while not rospy.get_param("/IGR/config_ready", False):
            time.sleep(0.1)

        code_path = os.path.dirname(os.path.abspath(__file__))
        exps_path = os.path.join(code_path, "../..", exps_dir)

        if gpu_num != 'cpu':
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu_num)

        conf = ConfigFactory.parse_file(os.path.join(code_path, '../../configs', conf))
        experiment_directory = os.path.join(exps_path, exp_name)

        # take the last in the folder
        timestamps = os.listdir(experiment_directory)
        timestamp = sorted(timestamps)[-1]

        experiment_directory = os.path.join(experiment_directory, timestamp)

        saved_model_state = torch.load(
            os.path.join(experiment_directory, 'checkpoints', 'ModelParameters', epoch + ".pth"),
            map_location=("cpu" if (not torch.cuda.is_available() or gpu_num == "cpu") else "cuda:" + str(gpu_num)))
        saved_model_epoch = saved_model_state["epoch"]
        with_normals = conf.get_float('network.loss.normals_lambda') > 0
        network = utils.get_class(conf.get_string('train.network_class'))(
            d_in=conf.get_int('train.latent_size') + conf.get_int('train.d_in'), **conf.get_config('network.inputs'))

        network.load_state_dict({k.replace('module.', ''): v for k, v in saved_model_state["model_state_dict"].items()})

        if torch.cuda.is_available():
            network = network.cuda()

        split_file = os.path.join(code_path, '../..', 'splits', split)

        save_directory = rospy.get_param("/IGR/save_directory", None)
        while save_directory is None:
            save_directory = rospy.get_param("/IGR/save_directory", None)
            rospy.sleep(0.1)

        utils.mkdir_ifnotexists(os.path.join(save_directory))

        return network, conf, save_directory, split_file

    def get_shape(self, latent, scale):
        """
        Help function to get shape from latent vector
        @param latent: latent vector
        @type latent: torch.tensor()
        @param scale: scale of the objects
        @type scale: float
        @return:
        @rtype:
        """
        self.network.eval()
        mesh, eikonal = plt.plot_surface(
            decoder=self.network,
            latent=latent,
            scale=scale,
            resolution=self.resolution,
            unc_level=self.level,
            mesh_level=0)
        self.network.train()
        return mesh, eikonal

    def run(self):
        """
        Main function of the class
        @return:
        @rtype:
        """
        global_sigma = self.conf.get_float('network.sampler.properties.global_sigma')
        local_sigma = self.conf.get_float('network.sampler.properties.local_sigma')
        sampler = Sampler.get_sampler(self.conf.get_string('network.sampler.sampler_type'))(global_sigma, local_sigma)

        latent_lambda = self.conf.get_float('network.loss.latent_lambda')

        normals_lambda = self.conf.get_float('network.loss.normals_lambda')

        grad_lambda = self.conf.get_float('network.loss.lambda')

        while not rospy.is_shutdown():
            self.lock.acquire()

            # If shape is requested, create it
            if self.shape_requested and (np.any(self.meshes_ready) or
                                         (self.current_shape is not None and
                                          self.current_shape.first_done and
                                          self.current_shape.current_iteration > (self.current_shape.max_iterations//2)-1)):
                self.shape_requested = False

                if np.any(self.meshes_ready):
                    temp_shape_id = np.where(self.meshes_ready)[0][0]
                    temp_shape = self.shapes[temp_shape_id]
                    self.meshes_ready[temp_shape_id] = False
                else:
                    temp_shape = self.current_shape

                rospy.loginfo(f"IGR: Ending with {temp_shape.iterations_since_save + 1} iterations")
                temp_shape.iterations_since_save = temp_shape.current_iteration

                if temp_shape.mesh is None:
                    rospy.loginfo(f"IGR: Creating mesh for {temp_shape.name}")
                    temp_shape.mesh, temp_shape.eikonal = \
                        self.get_shape(temp_shape.latent.detach().clone().requires_grad_(True),
                                       temp_shape.scale)

                np.save(temp_shape.eikonal_path, temp_shape.eikonal)
                temp_shape.mesh.export(temp_shape.filename + '.ply', 'ply')
                _ = _meshfix.clean_from_file(temp_shape.filename + '.ply', temp_shape.filename + '.ply')

                temp_shape.mesh, temp_shape.eikonal = None, None

                completed_ids = rospy.get_param("/IGR/completed_ids", [])
                completed_ids.append(temp_shape.id)
                rospy.set_param("/IGR/completed_ids", completed_ids)

                rospy.loginfo(f"IGR: Completed reconstruction of {temp_shape.name}")

            if self.current_shape is not None:
                # FPS
                if self.fps and self.points != -1:

                    distance = np.ones((self.current_shape.N,)) * 1e10
                    for i in range(self.current_shape.idxs.shape[0]):
                        if i <= int(self.points * 0.3) and i < self.current_shape.number_new_points:
                            farthest = np.random.randint(self.current_shape.N - self.current_shape.number_new_points, self.current_shape.N)
                        self.current_shape.idxs[i] = farthest
                        dist = self.current_shape.distances[farthest]
                        mask = dist < distance
                        distance[mask] = dist[mask]
                        farthest = np.argmax(distance, -1)

                    points = self.current_shape.points[self.current_shape.idxs]
                    normals = self.current_shape.normals[self.current_shape.idxs]

                else:
                    if self.points != -1 and self.points < self.current_shape.points.shape[0]:
                        random_idx = torch.randint(0, self.current_shape.points.shape[0] - self.points, (1, 1))
                        points = self.current_shape.points[random_idx: (random_idx + self.points)]
                        normals = self.current_shape.normals[random_idx: (random_idx + self.points)]
                        # random_idx = torch.randint(0, self.current_shape.points.shape[0], (self.points, ))
                        # points = self.current_shape.points[random_idx]
                        # normals = self.current_shape.normals[random_idx]
                    else:
                        points = self.current_shape.points
                        normals = self.current_shape.normals

                sample = sampler.get_points(points.unsqueeze(0)).squeeze()

                latent_all = self.current_shape.latent.expand(points.shape[0], -1)
                surface_pnts = torch.cat([latent_all, points], dim=1)

                sample_latent_all = self.current_shape.latent.expand(sample.shape[0], -1)
                nonsurface_pnts = torch.cat([sample_latent_all, sample], dim=1)

                surface_pnts.requires_grad_()
                nonsurface_pnts.requires_grad_()

                surface_pred = self.network(surface_pnts)
                nonsurface_pred = self.network(nonsurface_pnts)

                if self.free_space and self.free_space_points.shape[0] > 0:
                    fs_latent_all = self.current_shape.latent.expand(self.free_space_points.shape[0], -1)
                    fs_pnts = torch.cat([fs_latent_all, self.free_space_points], dim=1)
                    fs_pnts.requires_grad_()
                    fs_pred = self.network(fs_pnts)
                    fs_loss = torch.sum(torch.abs(fs_pred[torch.le(torch.mul(fs_pred, (1 / self.current_shape.scale)), 0.001)]))

                surface_grad = gradient(surface_pnts, surface_pred)
                nonsurface_grad = gradient(nonsurface_pnts, nonsurface_pred)

                surface_loss = torch.abs(surface_pred).mean()
                grad_loss = torch.mean((nonsurface_grad.norm(2, dim=-1) - 1).pow(2))
                normals_loss = ((surface_grad - normals).abs()).norm(2, dim=1).mean()
                latent_loss = self.current_shape.latent.abs().mean()

                loss = surface_loss + latent_lambda * latent_loss + normals_lambda * normals_loss + grad_lambda * grad_loss

                if self.free_space and self.free_space_points.shape[0] > 0 and fs_loss > 0:
                    loss += fs_loss
                self.current_shape.optimizer.zero_grad()

                loss.backward()
                self.current_shape.optimizer.step()
                self.current_shape.total_iterations += 1
                self.current_shape.current_iteration += 1
                if self.current_shape.mesh is None:
                    self.current_shape.iterations_since_save += 1

                if (self.current_shape.current_iteration + 1) == self.current_shape.max_iterations:
                    if not self.current_shape.first_done:
                        self.current_shape.first_done = True
                        # To complete all shapes for the first time
                        self.shape_requested = True

                    rospy.loginfo(f"IGR: Creating mesh for {self.current_shape.name}")
                    self.current_shape.mesh, self.current_shape.eikonal = \
                        self.get_shape(self.current_shape.latent.detach().clone().requires_grad_(True),
                                       self.current_shape.scale)

                    self.meshes_ready[self.shape_ids.index(self.current_shape.id)] = True
                    self.current_shape.current_iteration = 0

                    if self.requested_ids.empty():
                        self.current_shape = None
                    else:
                        self.current_shape = self.shapes[self.shape_ids[self.requested_ids.get()]]
                        if self.free_space:
                            shape_id = self.shape_ids.index(self.current_shape.id)
                            free_space_points = np.array(
                                rospy.get_param("/IGR/free_space/" + self.current_shape.name, [])).reshape((-1, 3))
                            free_space_points -= self.shapes[shape_id].center
                            free_space_points *= self.shapes[shape_id].scale
                            self.free_space_points = torch.tensor(free_space_points).float()
                            if torch.cuda.is_available():
                                self.free_space_points = self.free_space_points.cuda()
                        rospy.loginfo(f"IGR: Got request for shape {self.current_shape.name}")

            self.lock.release()
            if not rospy.get_param("/actvh_running", True) and self.current_shape is None:
                break
        rospy.set_param("/IGR/ended", True)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--fps",
        "-f",
        dest="fps",
        required=False,
        default=False
    )
    arg_parser.add_argument(
        "--free_space",
        "-fs",
        dest="free_space",
        required=False,
        default=False
    )
    arg_parser.add_argument(
        "--points",
        "-p",
        dest="num_of_points",
        required=False,
        default=1000
    )
    arg_parser.add_argument(
        "--level",
        "-l",
        dest="level",
        required=False,
        default=0
    )

    args_to_parse = []
    for arg in sys.argv[1:]:
        if ("__" and ":=" and "_") not in arg:
            args_to_parse.append(arg)

    args = arg_parser.parse_args(args_to_parse)
    rospy.init_node("IGR_node")
    igr_handle = IGR(int(args.num_of_points), get_bool(args.free_space), float(args.level), get_bool(args.fps))
