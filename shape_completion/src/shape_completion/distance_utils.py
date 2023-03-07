#!/usr/bin/env python3
"""
Utils for distance and similarity computation

@author Lukas Rustler
"""
import os

import numpy as np
from scipy.spatial import cKDTree as kd
from subprocess import call, PIPE, Popen
import binvox_rw


def compute_binvox(path, resolution=40, gt=False):
    """
    Function to compute binvoxes and return data as numpy array
    @param path: path to the mesh file
    @type path: string
    @param resolution: resolution of binvoxes
    @type resolution: int
    @param gt: whether we compute binvoxes for ground truth
    @type gt: bool
    @return: binvox data
    @rtype: numpy array
    """
    path = os.path.normpath(path)
    if os.path.isfile(path.replace(".ply", ".binvox").replace(".stl", ".binvox")):
        cmd = "rm "+path.replace(".ply", ".binvox").replace(".stl", ".binvox")
        p = Popen(cmd, shell=True)
        p.wait()

    if not gt or (gt and not os.path.isfile(path.replace(".ply", ".npy").replace(".stl", ".npy"))):
        cmd = "binvox -d " + str(resolution) + " " + path
        proc = Popen(cmd, shell=True, stdout=PIPE)
        proc.wait()
        with open(path.replace(".ply", ".binvox").replace(".stl", ".binvox"), 'rb') as binvox_file:
            binvox = binvox_rw.read_as_3d_array(binvox_file)
            data = binvox.data
        cmd = "rm " + path.replace(".ply", ".binvox").replace(".stl", ".binvox")
        proc = Popen(cmd, shell=True, stdout=PIPE)
        proc.wait()
        np.save(path.replace(".ply", ".npy").replace(".stl", ".npy"), binvox.data)
    else:  # if GT and we already binvoxed that once -> no need for new computation and just read results
        data = np.load(path.replace(".ply", ".npy").replace(".stl", ".npy"))

    return data


def jaccard_similarity(s1, s2):
    """
    Computes Jaccard similarity (intersection over union) of two arrays.
    @param s1: Array for the first object
    @type s1: list / np.array()
    @param s2: Array for the second object
    @type s2: list / np.array()
    @return: Jaccard similarity, in range <0,1>
    @rtype: float
    """
    intersection = np.count_nonzero(np.logical_and(s1, s2))
    union = np.count_nonzero(np.logical_or(s1, s2))

    return float(intersection)/float(union) if union != 0 else 0


def chamfer_distance(s1, s2):
    """
    Computes chamfer distance between two sets of point.
    @param s1: Array for the first object
    @type s1: list / np.array()
    @param s2: Array for the second object
    @type s2: list / np.array()
    @return: Chamfer distance
    @rtype: float
    """
    s1 = np.array(s1)
    s2 = np.array(s2)

    s1_tree = kd(s1)
    s2_tree = kd(s2)

    d_s1, _ = s2_tree.query(s1, p=2)
    d_s2, _ = s1_tree.query(s2, p=2)

    return np.mean(d_s1) + np.mean(d_s2)

