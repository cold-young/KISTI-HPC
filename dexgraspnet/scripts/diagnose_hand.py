#!/usr/bin/env python3
"""
Headless diagnostic: render the KISTAR hand mesh with its contact candidates
(red) and penetration keypoints (blue) overlaid, from several angles.

Per the DexGraspNet hand-customization recipe, contact candidates must lie on
the grasping surfaces (finger pads / inner palm) and penetration keypoints must
sit inside the finger bones. This lets us verify that visually.

Usage (from DexGrasp_KIST/grasp_generation/):
    cd /home/chanyoung/isaac_ws/DexGrasp_KIST/grasp_generation
    conda activate dexgraspnet
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    python /home/chanyoung/cy_ws/KISTI-HPC/dexgraspnet/scripts/diagnose_hand.py --out /path/out.png
"""
import os, sys, argparse
sys.path.append(os.path.realpath("."))
import numpy as np
import torch
import transforms3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--flat", action="store_true",
                    help="use flat (all-zero joints) pose instead of half-curl")
    args = ap.parse_args()

    from utils.kistar_model import HandModel
    device = "cpu"
    hm = HandModel(
        urdf_path="kistar/kistar.urdf",
        mesh_path="kistar",
        contact_points_path="kistar/contact_points.json",
        penetration_points_path="kistar/penetration_points.json",
        n_surface_points=0,
        device=device,
    )
    n = hm.n_dofs
    # identity rotation in 6d
    pose = torch.zeros(3 + 6 + n)
    pose[3] = 1.0
    pose[7] = 1.0  # rot6d ~ identity (col0=x, col1=y)
    if not args.flat:
        # a light curl so finger pads face inward (matches generation init flavor)
        joints = torch.zeros(n)
        pose[9:] = joints
    hm.set_parameters(pose.unsqueeze(0))

    hand_mesh = hm.get_trimesh_data(0)
    contacts = hm.get_contact_candidates()[0].detach().cpu().numpy()
    try:
        penet = hm.get_penetraion_keypoints()[0].detach().cpu().numpy()
    except Exception:
        penet = np.zeros((0, 3))

    print(f"n_dofs={n}  contacts={len(contacts)}  penetration_kp={len(penet)}")

    V = np.asarray(hand_mesh.vertices)
    F = np.asarray(hand_mesh.faces)

    views = [(20, 45), (90, -90), (0, 0), (-90, 90)]
    titles = ["persp", "palm-down (top)", "front", "palm-up (bottom)"]
    fig = plt.figure(figsize=(16, 4.5))
    for k, ((elev, azim), ttl) in enumerate(zip(views, titles)):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        coll = Poly3DCollection(V[F], alpha=0.18, facecolor="#9fb6cf",
                                edgecolor="none")
        ax.add_collection3d(coll)
        ax.scatter(contacts[:, 0], contacts[:, 1], contacts[:, 2],
                   c="red", s=14, label=f"contact ({len(contacts)})")
        if len(penet):
            ax.scatter(penet[:, 0], penet[:, 1], penet[:, 2],
                       c="blue", s=14, label=f"penet ({len(penet)})")
        mn, mx = V.min(0), V.max(0)
        c = (mn + mx) / 2
        r = (mx - mn).max() / 2 * 1.05
        ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(ttl, fontsize=11)
        ax.set_axis_off()
        if k == 0:
            ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("KISTAR hand — contact candidates (red) & penetration keypoints (blue)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
