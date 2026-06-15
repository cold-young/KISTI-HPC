#!/usr/bin/env python3
"""
Render a gallery of validated grasps using Isaac Gym's offscreen camera sensors,
then tile the captures into one PNG. No GUI window needed (uses graphics device).

Usage (from DexGrasp_KIST/grasp_generation/):
    cd /home/chanyoung/isaac_ws/DexGrasp_KIST/grasp_generation
    conda activate dexgraspnet
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    python /home/chanyoung/cy_ws/KISTI-HPC/dexgraspnet/scripts/gallery_isaacgym.py \
        --dataset_path /home/chanyoung/cy_ws/KISTI-HPC/dexgraspnet/data/dataset \
        --mesh_path /home/chanyoung/isaac_ws/DexGrasp_KIST/data/meshdata_new \
        --out /home/chanyoung/cy_ws/KISTI-HPC/dexgraspnet/data/gallery_isaac.png
"""

import os
import sys
import argparse
import math

sys.path.append(os.path.realpath("."))

from isaacgym import gymapi  # noqa: E402  (must import before torch)
import numpy as np  # noqa: E402
import transforms3d  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

translation_names = ["WRJTx", "WRJTy", "WRJTz"]
rot_names = ["WRJRx", "WRJRy", "WRJRz"]
qpos_joint_names = [
    "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
    "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
    "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
    "robot0:LFJ4", "robot0:LFJ3", "robot0:LFJ2", "robot0:LFJ1",
]
urdf_joint_names = [
    "index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3",
    "middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3",
    "ring_joint_0", "ring_joint_1", "ring_joint_2", "ring_joint_3",
    "thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--hand_root", default="kistar")
    parser.add_argument("--cam_dist", type=float, default=0.28)
    parser.add_argument("--palm_axis", type=str, default="0,0,1",
                        help="hand-local axis of palm normal (x,y,z)")
    parser.add_argument("--flip", action="store_true",
                        help="flip camera to the other side of the palm normal")
    parser.add_argument("--settle_steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=560)
    args = parser.parse_args()

    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt = 1 / 60
    sim_params.substeps = 2
    sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0.0)
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim_params.use_gpu_pipeline = False

    sim = gym.create_sim(args.gpu, args.gpu, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("[ERROR] Failed to create sim (graphics device unavailable?)")
        sys.exit(1)

    cam_props = gymapi.CameraProperties()
    cam_props.width = args.width
    cam_props.height = args.height

    hand_opts = gymapi.AssetOptions()
    hand_opts.disable_gravity = True
    hand_opts.fix_base_link = True
    hand_opts.collapse_fixed_joints = True
    hand_asset = gym.load_asset(sim, args.hand_root, "kistar.urdf", hand_opts)

    # Collect objects with >=1 valid grasp
    objs = []
    for f in sorted(os.listdir(args.dataset_path)):
        if not f.endswith(".npy"):
            continue
        d = np.load(os.path.join(args.dataset_path, f), allow_pickle=True)
        if len(d) > 0:
            objs.append((f[:-4], d))
    if not objs:
        print("No objects with valid grasps.")
        sys.exit(1)

    # Lights
    gym.set_light_parameters(sim, 0, gymapi.Vec3(2, 2, 2),
                             gymapi.Vec3(0.2, 0.2, 0.2), gymapi.Vec3(0.5, -1, -0.5))
    gym.set_light_parameters(sim, 1, gymapi.Vec3(1, 1, 1),
                             gymapi.Vec3(0.2, 0.2, 0.2), gymapi.Vec3(-0.5, 1, 0.5))

    cams = []
    spacing = 1.0
    obj_opts = gymapi.AssetOptions()
    obj_opts.override_com = True
    obj_opts.override_inertia = True
    obj_opts.density = 500

    for code, data in objs:
        num = min(args.num, len(data) - 1)
        d = data[num]
        qpos = d["qpos"]
        scale = float(d["scale"])

        rot = transforms3d.euler.euler2quat(*[qpos[n] for n in rot_names])  # w,x,y,z
        trans = [qpos[n] for n in translation_names]
        hand_qpos = [qpos[n] for n in qpos_joint_names]

        env = gym.create_env(sim, gymapi.Vec3(-spacing, -spacing, -spacing),
                             gymapi.Vec3(spacing, spacing, spacing), 8)

        # Hand
        pose = gymapi.Transform()
        pose.r = gymapi.Quat(rot[1], rot[2], rot[3], rot[0])
        pose.p = gymapi.Vec3(*trans)
        hand_h = gym.create_actor(env, hand_asset, pose, "hand", 0, -1)
        props = gym.get_actor_dof_properties(env, hand_h)
        props["driveMode"].fill(gymapi.DOF_MODE_POS)
        props["stiffness"].fill(200)
        props["damping"].fill(50.0)
        gym.set_actor_dof_properties(env, hand_h, props)
        dof = gym.get_actor_dof_states(env, hand_h, gymapi.STATE_ALL)
        for i, jn in enumerate(urdf_joint_names):
            ji = gym.find_actor_dof_index(env, hand_h, jn, gymapi.DOMAIN_ACTOR)
            dof["pos"][ji] = hand_qpos[i]
        gym.set_actor_dof_states(env, hand_h, dof, gymapi.STATE_ALL)
        gym.set_actor_dof_position_targets(env, hand_h, dof["pos"])

        # Object
        obj_asset = gym.load_asset(
            sim, os.path.join(args.mesh_path, code, "coacd"), "coacd.urdf", obj_opts)
        opose = gymapi.Transform()
        opose.p = gymapi.Vec3(0, 0, 0)
        opose.r = gymapi.Quat(0, 0, 0, 1)
        obj_h = gym.create_actor(env, obj_asset, opose, "obj", 0, 1)
        gym.set_actor_scale(env, obj_h, scale)
        # color the object bright so it stands out from the white hand
        obj_color = gymapi.Vec3(1.0, 0.55, 0.1)  # orange
        n_obj_bodies = gym.get_actor_rigid_body_count(env, obj_h)
        for bi in range(n_obj_bodies):
            gym.set_rigid_body_color(env, obj_h, bi,
                                     gymapi.MESH_VISUAL_AND_COLLISION, obj_color)

        # Camera: look at the grasp center along the palm normal (palm-facing view)
        cam = gym.create_camera_sensor(env, cam_props)
        qx, qy, qz, qw = pose.r.x, pose.r.y, pose.r.z, pose.r.w
        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
        ])
        palm_axis = np.array([float(a) for a in args.palm_axis.split(",")])
        normal = R @ palm_axis
        normal = normal / (np.linalg.norm(normal) + 1e-9)
        if args.flip:
            normal = -normal
        center = np.array([0.0, 0.0, 0.0])  # object is at origin = grasp focus
        d_ = args.cam_dist
        cam_pos = gymapi.Vec3(*(center + normal * d_))
        gym.set_camera_location(cam, env, cam_pos,
                                gymapi.Vec3(*center))
        cams.append((code, len(data), env, cam))

    # Settle a few steps so PD controller holds the pose, then render
    for _ in range(args.settle_steps):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)

    def autocrop(arr, pad=24):
        # crop to non-black content (square), then return PIL image
        gray = arr.sum(axis=2)
        ys, xs = np.where(gray > 12)
        if len(xs) == 0:
            return Image.fromarray(arr)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        half = max(x1 - x0, y1 - y0) // 2 + pad
        H, W = arr.shape[:2]
        x0 = max(0, cx - half); x1 = min(W, cx + half)
        y0 = max(0, cy - half); y1 = min(H, cy + half)
        return Image.fromarray(arr[y0:y1, x0:x1])

    # Capture
    tiles = []
    tile_px = 520
    for code, nvalid, env, cam in cams:
        img = gym.get_camera_image(sim, env, cam, gymapi.IMAGE_COLOR)
        img = img.reshape(args.height, args.width, 4)[:, :, :3].copy()
        im = autocrop(img).resize((tile_px, tile_px), Image.LANCZOS)
        tiles.append((code, nvalid, im))
        print(f"  captured {code} ({nvalid} valid)")

    gym.destroy_sim(sim)

    # Tile into a grid
    n = len(tiles)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    pad = 10
    label_h = 30
    tw = th = tile_px
    W = ncols * tw + (ncols + 1) * pad
    H = nrows * (th + label_h) + (nrows + 1) * pad
    canvas = Image.new("RGB", (W, H), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i, (code, nvalid, im) in enumerate(tiles):
        r, c = divmod(i, ncols)
        x = pad + c * (tw + pad)
        y = pad + r * (th + label_h + pad)
        canvas.paste(im, (x, y))
        draw.text((x + 4, y + th + 6), f"{code}  ({nvalid} valid)",
                  fill=(235, 235, 235), font=font)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    canvas.save(args.out)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
