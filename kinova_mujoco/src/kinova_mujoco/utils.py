#!/usr/bin/env python3
"""
Utils for object handling in the simulation

@author Jan Behrens, edited by Lukas Rustler
"""
import urdf_parser_py.urdf
import subprocess
import os
import numpy as np
import trimesh
from odio_urdf import Child, Parent, Joint, Link, Origin, Inertial, Mass, Inertia, Visual, Geometry, Material, \
    Collision, Contact, Mesh, Robot, Dynamics


def link(mesh, robot_name, center_gravity, mass, I, material, geom_origin, has_inertia=True, has_visual=True,
         has_collision=True):
    """
        Most of the links are the same except for the passed in info.
        This function just allows grouping the important numbers better.
    """
    # N = str(N)
    assert isinstance(mesh, str)
    if robot_name is None:
        ret = Link(name=mesh.split('.')[0])
    else:
        ret = Link(name=robot_name)
    if has_inertia:
        ret(Inertial(
            Origin(list(center_gravity)),
            Mass(value=mass),
            Inertia(I)))

    if has_visual:
        ret(Visual(
            Origin(geom_origin),
            Geometry(Mesh(filename=os.path.join('package://kinova_mujoco/meshes', mesh.replace(os.path.dirname(mesh),'').replace('/','')))),
            Material(material)))
    if has_collision:
        ret(Collision(
            Origin(geom_origin),
            Geometry(Mesh(filename=os.path.join('package://kinova_mujoco/meshes', mesh.replace(os.path.dirname(mesh),'').replace('/','')))),
            Material(material)),
            Contact())

    return ret


def mesh2urdf(mesh_path, obj_name, maxhulls=10000, init_orig=[],
              robot_urdf=None, mujoco=True, convex_decomp=False, link_names=[], color="Blue"):

    mesh = trimesh.load(mesh_path)  # type: trimesh

    assert isinstance(mesh, trimesh.Trimesh)
    rot_mat = [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    mesh = mesh.apply_transform(rot_mat)

    # save scaled mesh
    mesh_path_visual = os.path.join(os.path.dirname(mesh_path), '../meshes/', obj_name + '_visual.stl')
    mesh.export(file_obj=mesh_path_visual, file_type='stl')

    if not mujoco:
        base_obj_name = os.path.basename(os.path.normpath(mesh_path))
        base_mesh_path = os.path.join(os.path.dirname(mesh_path), '../meshes/', base_obj_name)
        mesh.export(file_obj=base_mesh_path, file_type='stl')

    # test if mesh has several components
    mesh_components = mesh.split(only_watertight=False)

    # make convex decomposition for simplified collision model
    if not convex_decomp:
        bodies = [mesh_components[0]]
    else:
        bodies = []
        for mesh_component in mesh_components:
            assert isinstance(mesh_component, trimesh.Trimesh)
            if mesh_component.bounding_box.volume < 1e-5 or mesh_component.scale < 0.0005 or len(
                    mesh_component.vertices) < 15:
                continue

            # mesh_component.show()
            decomp = mesh_component.convex_decomposition(maxhulls=maxhulls, pca=0, mode=0, resolution=1000000,
                                                         maxNumVerticesPerCH=1, gamma=0.0005, concavity=0)
            if isinstance(decomp, list):
                bodies += decomp
            elif isinstance(decomp, trimesh.Trimesh):
                bodies.append(decomp)

    # show the decomposition
    scene = trimesh.Scene()
    for body in bodies:
        scene.add_geometry(body)

    if robot_urdf is None:
        myobj = Robot()
        obj_start_index = len(myobj)
    else:
        myobj = robot_urdf
        obj_start_index = len(myobj)
    # add visual link for RVIZ
    # mesh_path = obj_name + '.stl'
    if convex_decomp:
        link_vis = link(str(mesh_path_visual), obj_name + '_visual', None, None, None, "Grey", [0, 0, 0],
                        has_collision=False, has_inertia=False, has_visual=True)
        myobj(link_vis)
        link_names.append(obj_name + '_visual')

    mesh_extents = mesh.extents
    mesh_volume = mesh_extents[0] * mesh_extents[1] * mesh_extents[2]
    for idx, body in enumerate(bodies):  # type: trimesh.Trimesh
        name = obj_name + '_' + '{:06d}'.format(idx)
        link_names.append(name)
        body_mesh_path = name + '.stl'
        # os.path.join(os.path.dirname(mesh_path), body_mesh_path)
        body.export(file_obj=os.path.join(os.path.dirname(mesh_path), '../meshes/', body_mesh_path), file_type='stl')

        body.merge_vertices()
        body.remove_degenerate_faces()
        body.remove_duplicate_faces()

        d, w, h = body.extents  # get dimensions

        # set mass of the whole object to 0.5kg -> mainly to act more naturally
        body_volume = d * w * h
        mass = 0.5 * (body_volume/mesh_volume)

        # compute inertia tensor as inertia tensor of cuboid with given dimensions (inertia of objects bbox)
        inertia = np.zeros((3, 3))
        inertia[0, 0] = (1/12)*mass*(h*h+d*d)
        inertia[1, 1] = (1 / 12) * mass * (w*w + h*h)
        inertia[2, 2] = (1 / 12) * mass * (w*w + d*d)

        ixx = inertia[0, 0]
        iyy = inertia[1, 1]
        izz = inertia[2, 2]
        ixy = inertia[0, 1]
        ixz = inertia[0, 2]
        iyz = inertia[1, 2]

        centroid = body.center_mass

        print(f"\n{name}\nmass: {mass}\ninertia: {ixx, iyy, izz, ixy, ixz, iyz}\ncenter of mass: {centroid}\ndimensions: {d, w, h}")
        if convex_decomp:
            link4 = link(body_mesh_path, None, centroid, mass, [ixx, ixy, ixz, iyy, iyz, izz], color, [0, 0, 0])
            myobj(link4)

    if convex_decomp:
        for idx, links in enumerate(zip(myobj[obj_start_index:-1], myobj[obj_start_index+1:])):
            l1, l2 = links
            joint = Joint(obj_name + "_joint_{:04d}".format(idx), Parent(str(l1.name)), Child(str(l2.name)), type="fixed")
            myobj(joint)

        if mujoco:
            joint = Joint("joint_world_" + obj_name, Parent('base_link'), Child(myobj[obj_start_index].name), type="floating")
            joint(Dynamics(damping=0.1, friction=.1))
        else:
            joint = Joint("joint_world_" + obj_name, Parent('base_link'), Child(myobj[obj_start_index].name), type="fixed")
        joint(Origin(init_orig))
        myobj(joint)

    return myobj, link_names


def create_object_scene(objects, origins=[], mujoco=True, convex_decomp=False):
    """
    Call helping funtions to create given objects
    @param objects: name of the objects to load
    @type objects: list of string
    @param origins: list of origins of the objects
    @type origins: list of lists of floats
    @param mujoco: whether to use floating joints, so mujoco simulates the objects
    @type mujoco: bool
    @param convex_decomp: whether to do convex decomposition of the objects (mujoco needs this)
    @type convex_decomp: bool
    @return:
    @rtype:
    """

    colors = ["Blue", "Green", "Red", "Yellow", "Orange"]
    objects_names = []
    scene_urdf = None
    link_names = []
    for idx, obj, origin in zip(np.arange(0, len(objects)), objects, origins):
        obj_name = obj if obj not in objects_names else obj+"_"+str(objects_names.count(obj))
        scene_urdf, link_names = mesh2urdf(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../GT_meshes',
                                                        obj) + '.stl',
                                           obj_name=obj_name,
                                           maxhulls=1e6, init_orig=origin, robot_urdf=scene_urdf, mujoco=mujoco,
                                           convex_decomp=convex_decomp, link_names=link_names, color=colors[idx])
        objects_names.append(obj)

    text_file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../urdf/shape_completion_scene.urdf"), "w")
    text_file.write(scene_urdf.__str__())
    text_file.close()

    return scene_urdf, link_names


def str_to_bool(string):
    if string.lower() in ["true", 1]:
        return True
    else:
        return False


def bool_to_str(bool_):
    return "true" if bool_ else "false"


def prepare_urdf(printed_finger, convex_decomp, mujoco):
    proc = subprocess.Popen(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../scripts/run_xacro.sh '+printed_finger+' '+convex_decomp+ ' false true  _mujoco'), shell=True)
    proc.wait()
    if not mujoco:
        proc = subprocess.Popen(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../scripts/run_xacro.sh '+printed_finger+' '+convex_decomp+ ' false false _classic'), shell=True)
        proc.wait()
    files = ["kinova_fel_shape_completion_mujoco"]
    if not mujoco:
        files.append("kinova_fel_shape_completion_classic")
    for f in files:
        robot = urdf_parser_py.urdf.URDF.from_xml_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../urdf/'+f+'.urdf'))
        assert isinstance(robot, urdf_parser_py.urdf.URDF)

        file_object = open(os.path.join(os.path.dirname(__file__), '../../urdf/'+f+'.urdf'), 'r+')
        lines = file_object.readlines()
        file_object.close()

        new_lines = []
        for line in lines:
            # take over comment lines unchanged
            if '<!--' in line:
                new_lines.append(line + "\n")
            elif '<mesh filename="package://' in line:
                # write new stl mesh location in robot mujoco package
                link_name = line.split('/')[-2]
                if 'scale' in link_name:
                    pass
                    # link_name.
                new_line = line.split('//')[0] + '//' + 'kinova_mujoco/meshes/' + link_name.replace('.dae', '.STL') + '/>'
                # line = line.replace('.dae', '.stl')
                new_lines.append(new_line + "\n")
            elif '<material name="">' in line:
                # mujoco wants everything to have a filled material tag
                new_line = line.replace('""', '"DUMMY_MATERIAL"')
                new_lines.append(new_line + "\n")
            elif '<mimic joint=' in line:
                # mujoco does not support mimic joints. we have to use a custom controller to make the mimic functionality.
                pass
            else:
                # take over normal lines
                new_lines.append(line + "\n")

        file_object = open(os.path.join(os.path.dirname(__file__), '../../urdf/'+f+'_NoDae.urdf'), 'w+')
        file_object.writelines(new_lines)
        file_object.close()

    if mujoco:
        os.system("cp " +
                  os.path.join(os.path.dirname(__file__), '../../urdf/kinova_fel_shape_completion_mujoco_NoDae.urdf') +
                  " " +
                  os.path.join(os.path.dirname(__file__), '../../urdf/kinova_fel_shape_completion_classic_NoDae.urdf'))
