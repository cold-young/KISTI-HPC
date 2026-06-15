#!/usr/bin/env python3
"""
Render a gallery PNG of validated grasps (KISTAR hand + object) — one thumbnail
per object, laid out in a grid. Headless (matplotlib Agg), no display needed.

Usage (from DexGrasp_KIST/grasp_generation/):
    cd /home/chanyoung/isaac_ws/DexGrasp_KIST/grasp_generation
    conda activate dexgraspnet
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    python /home/chanyoung/cy_ws/KISTI-HPC/dexgraspnet/scripts/gallery.py \
        --dataset_path /home/chanyoung/cy_ws/KISTI-HPC/dexgraspnet/data/dataset \
        --mesh_path /home/chanyoung/isaac_ws/DexGrasp_KIST/data/meshdata_new \
        --out /home/chanyoung/cy_ws/KISTI-HPC/dexgraspnet/data/gallery.png
"""

import os
import sys
import argparse
import math

sys.path.append(os.path.realpath("."))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh as tm  # noqa: E402
import transforms3d  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

translation_names = ["WRJTx", "WRJTy", "WRJTz"]
rot_names = ["WRJRx", "WRJRy", "WRJRz"]
joint_names = [
    "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
    "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
    "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
    "robot0:LFJ4", "robot0:LFJ3", "robot0:LFJ2", "robot0:LFJ1",
]


def qpos_to_hand_pose(qpos, device):
    rot = np.array(transforms3d.euler.euler2mat(*[qpos[n] for n in rot_names]))
    rot6d = rot[:, :2].T.ravel().tolist()
    vec = [qpos[n] for n in translation_names] + rot6d + [qpos[n] for n in joint_names]
    return torch.tensor(vec, dtype=torch.float, device=device).unsqueeze(0)


def add_mesh(ax, verts, faces, color, alpha, shade=True, zsort="average"):
    tris = verts[faces]
    coll = Poly3DCollection(tris, alpha=alpha, linewidths=0)
    if shade:
        # simple lambert shading from a fixed light direction
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        ln[ln == 0] = 1
        n = n / ln
        light = np.array([0.3, 0.4, 0.85])
        light = light / np.linalg.norm(light)
        shade_val = 0.45 + 0.55 * np.clip(n @ light, 0, 1)
        base = np.array(matplotlib.colors.to_rgb(color))
        facecolors = np.clip(base[None, :] * shade_val[:, None], 0, 1)
        coll.set_facecolor(facecolors)
    else:
        coll.set_facecolor(color)
    coll.set_zsort(zsort)
    ax.add_collection3d(coll)


def set_equal_aspect(ax, verts_all):
    mn = verts_all.min(axis=0)
    mx = verts_all.max(axis=0)
    center = (mn + mx) / 2
    r = (mx - mn).max() / 2 * 1.1
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num", type=int, default=0, help="which grasp index per object")
    parser.add_argument("--max_obj_faces", type=int, default=20000,
                        help="decimate object meshes above this face count")
    args = parser.parse_args()

    from utils.kistar_model import HandModel
    device = "cpu"

    hand_model = HandModel(
        urdf_path="kistar/kistar.urdf",
        mesh_path="kistar",
        contact_points_path="kistar/contact_points.json",
        penetration_points_path="kistar/penetration_points.json",
        n_surface_points=0,
        device=device,
    )

    # Collect objects with >=1 valid grasp
    objs = []
    for f in sorted(os.listdir(args.dataset_path)):
        if not f.endswith(".npy"):
            continue
        d = np.load(os.path.join(args.dataset_path, f), allow_pickle=True)
        if len(d) > 0:
            objs.append((f[:-4], d))

    n = len(objs)
    if n == 0:
        print("No objects with valid grasps found.")
        sys.exit(1)

    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig = plt.figure(figsize=(ncols * 4, nrows * 4))
    print(f"Rendering {n} objects in {nrows}x{ncols} grid...")

    for idx, (code, data) in enumerate(objs):
        num = min(args.num, len(data) - 1)
        d = data[num]
        qpos = d["qpos"]
        scale = float(d["scale"])

        # Hand mesh
        hand_pose = qpos_to_hand_pose(qpos, device)
        hand_model.set_parameters(hand_pose)
        hand_mesh = hand_model.get_trimesh_data(0)

        # Object mesh (normalized) scaled
        obj_path = os.path.join(args.mesh_path, code, "coacd", "decomposed.obj")
        obj_mesh = tm.load(obj_path, force="mesh", process=False)
        obj_mesh.apply_scale(scale)
        if len(obj_mesh.faces) > args.max_obj_faces:
            obj_mesh = obj_mesh.simplify_quadric_decimation(args.max_obj_faces)

        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        # hand first (semi-transparent), object on top (opaque, bright)
        add_mesh(ax, np.asarray(hand_mesh.vertices), np.asarray(hand_mesh.faces),
                 "#5b9bd5", 0.4)
        add_mesh(ax, np.asarray(obj_mesh.vertices), np.asarray(obj_mesh.faces),
                 "#ff8c42", 1.0)

        all_v = np.vstack([np.asarray(obj_mesh.vertices),
                           np.asarray(hand_mesh.vertices)])
        set_equal_aspect(ax, all_v)
        ax.view_init(elev=20, azim=45)
        ax.set_title(f"{code}\n({len(data)} valid)", fontsize=10)
        ax.set_axis_off()
        print(f"  [{idx+1}/{n}] {code} (grasp {num}, scale {scale:.2f})")

    fig.suptitle("KISTAR DexGraspNet — validated grasps (hand=blue, object=orange)",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
