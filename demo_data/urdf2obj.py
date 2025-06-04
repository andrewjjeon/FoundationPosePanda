#! /usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy

import shutil

# python urdf2obj.py panda.urdf --output_fn panda.obj --config_json panda.json --show
parser = argparse.ArgumentParser()
parser.add_argument("urdf", type=Path, help="path to URDF file")
parser.add_argument(
    "--output_fn", type=Path, default=None, help="filename to save output mesh")
parser.add_argument(
    "--config",
    type=float,
    nargs="+",
    default=None,
    help="configuration of the robot joints",
)
parser.add_argument(
    "--config_json",
    type=Path,
    default=None,
    help="path to json file containing configuration of the robot joints",
)
parser.add_argument(
    "--random_config",
    action="store_true",
    help="sample a random configuration of the robot joints",
)
parser.add_argument(
    "--show",
    action="store_true",
    help="show the robot in a viewer",
)
parser.add_argument(
    "--info",
    action="store_true",
    help="print information about the robot",
)
args = parser.parse_args()

# load URDF
robot = yourdfpy.URDF.load(str(args.urdf))
print(f"Loaded URDF from {args.urdf}")
print(robot.scene.geometry.keys())

# robot.scene.show()

for name, mesh in robot.scene.geometry.items():
    print(f"Mesh: {name}, Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")



# set configuration, can only set 1 configuration out of config, config_json, random_config
assert (
    sum([
        not (args.config_json is None),
        not (args.config is None), 
        args.random_config is True, 
    ]) in [0,1]
), "Cannot set more than one config argument"

# if config_json provided do this
if args.config_json:
    with open(args.config_json, "r") as fp:
        config_json_raw = json.load(fp)["robot_state"]
    robot.update_cfg(
        [
            config_json_raw[joint.name]
            for joint in robot.actuated_joints
        ]
    )
    print(f"Set robot configuration: {robot.cfg}")

# if config provided do this
elif args.config:
    assert (
        len(args.config) == robot.num_actuated_joints
    ), f"Length of config ({len(args.config)}) != " + \
        f"num of joints ({robot.num_actuated_joints})"
    robot.update_cfg(args.config)
    print(f"Set robot configuration: {robot.cfg}")

# if random_config provided do this
elif args.random_config:
    robot.update_cfg(
        [
            np.random.uniform(joint.limit.lower, joint.limit.upper)
            for joint in robot.actuated_joints
        ]
    )
    print(f"Set random robot configuration: {robot.cfg}")
else:
    print(f"No configuration specified; using zero configuration: {robot.cfg}")

# show robot, if requested
if args.show:
    robot.scene.show()

# robot.scene.export("test.glb")
#mesh_data = robot.scene.graph.get(node_name)  # mesh_data: (T, 'hand.obj')

# stephens code: mesh will be a scene of meshes with .obj and .mtl mapping
if args.output_fn:
    if not args.output_fn.parent.exists():
        args.output_fn.parent.mkdir(parents=True)
    resolver = trimesh.resolvers.FilePathResolver(args.output_fn.parent)
    with open(args.output_fn, "wb") as fp:
        robot.scene.show() #robot.scene looks great so export is the problem
        robot.scene.export(fp, "obj", include_texture=True, write_texture=True, resolver=resolver) #this is the problem line
    print(f"Exported to {args.output_fn}")

# perfect combined mesh but only uses 1 material
# if args.output_fn:
#     if not args.output_fn.parent.exists():
#         args.output_fn.parent.mkdir(parents=True)
#     # Check if the scene has submeshes and combine them
#     if isinstance(robot.scene, trimesh.Scene):
#         unified_mesh = robot.scene.to_mesh()
#     else:
#         unified_mesh = robot.scene
#     resolver = trimesh.resolvers.FilePathResolver(args.output_fn.parent)
#     with open(args.output_fn, "wb") as fp:
#         unified_mesh.export(fp, "obj", include_texture=True, write_texture=True, resolver=resolver)
#     print(f"Exported to {args.output_fn}")
