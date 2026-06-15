#!/usr/bin/env python3
"""
Phase 2: KISTAR hand sanity check — quick grasp generation on 2-3 objects.

Based on quick_kistar_example.ipynb. Generates a small batch of grasps
and saves plotly HTML for visual verification.

Usage:
    cd DexGrasp_KIST/grasp_generation
    python /path/to/quick_kistar_smoke.py \
        --meshdata_path ../data/meshdata \
        --object_codes contactdb-apple ycb-banana \
        --batch_size 5 \
        --output_dir /path/to/smoke_results
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import transforms3d

sys.path.insert(0, os.path.realpath("."))


def run_smoke_test(args):
    from utils.kistar_model import HandModel
    from utils.object_model import ObjectModel
    from utils.initializations import initialize_convex_hull
    from utils.energy import cal_energy
    from utils.optimizer import Annealing
    from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load hand model
    hand_model = HandModel(
        urdf_path="kistar/kistar.urdf",
        mesh_path="kistar",
        contact_points_path="kistar/contact_points.json",
        penetration_points_path="kistar/penetration_points.json",
        n_surface_points=1000,
        device=device,
    )
    print(f"Hand model loaded: {hand_model.n_dofs} DOF")

    # Load object model
    object_model = ObjectModel(
        data_root_path=args.meshdata_path,
        batch_size_each=args.batch_size,
        num_samples=2000,
        device=device,
    )

    # Verify objects exist
    for code in args.object_codes:
        obj_path = Path(args.meshdata_path) / code / "coacd" / "decomposed.obj"
        if not obj_path.exists():
            print(f"[ERROR] Object not found: {obj_path}")
            sys.exit(1)
        print(f"Object OK: {code}")

    object_model.initialize(args.object_codes)
    print(f"Objects loaded: {len(args.object_codes)} x {args.batch_size} grasps")

    # Initialize and optimize
    class Args:
        n_contact = 4
        switch_possibility = 0.5
        mu = 0.98
        step_size = 0.005
        starting_temperature = 18
        annealing_period = 100
        temperature_decay = 0.99
        stepsize_period = 50
        w_dis = 105.0
        w_pen = 11.18
        w_spen = 44.71
        w_joints = 1.0
        jitter_strength = 0.1
        distance_lower = 0.01262
        distance_upper = 0.11839
        theta_lower = -0.5236
        theta_upper = 0.5236
        thres_fc = 0.3
        thres_dis = 0.005
        thres_pen = 0.005
        seed = 42

    opt_args = Args()
    weight_dict = dict(
        w_dis=opt_args.w_dis, w_pen=opt_args.w_pen,
        w_spen=opt_args.w_spen, w_joints=opt_args.w_joints,
    )
    initialize_convex_hull(hand_model, object_model, opt_args)

    n_iter = args.n_iter
    optimizer = Annealing(
        hand_model,
        switch_possibility=opt_args.switch_possibility,
        starting_temperature=opt_args.starting_temperature,
        temperature_decay=opt_args.temperature_decay,
        annealing_period=opt_args.annealing_period,
        step_size=opt_args.step_size,
        stepsize_period=opt_args.stepsize_period,
        mu=opt_args.mu,
        device=device,
    )

    # Initial energy computation
    optimizer.zero_grad()
    energy, E_fc, E_dis, E_pen, E_spen, E_joints = cal_energy(
        hand_model, object_model, verbose=True, **weight_dict,
    )
    energy.sum().backward(retain_graph=True)

    print(f"\nOptimizing {n_iter} iterations...")
    for step in range(1, n_iter + 1):
        s = optimizer.try_step()

        optimizer.zero_grad()
        new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints = cal_energy(
            hand_model, object_model, verbose=True, **weight_dict,
        )
        new_energy.sum().backward(retain_graph=True)

        with torch.no_grad():
            accept, t = optimizer.accept_step(energy, new_energy)
            energy[accept] = new_energy[accept]
            E_fc[accept] = new_E_fc[accept]
            E_dis[accept] = new_E_dis[accept]
            E_pen[accept] = new_E_pen[accept]
            E_spen[accept] = new_E_spen[accept]
            E_joints[accept] = new_E_joints[accept]

        if step % 500 == 0 or step == n_iter:
            print(f"  [{step:5d}/{n_iter}] E={energy.mean():.4f} "
                  f"fc={E_fc.mean():.4f} dis={E_dis.mean():.4f} "
                  f"pen={E_pen.mean():.4f} spen={E_spen.mean():.4f}")

    # Save visualization
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go

        for obj_idx, code in enumerate(args.object_codes):
            for grasp_idx in range(min(3, args.batch_size)):
                idx = obj_idx * args.batch_size + grasp_idx
                hand_mesh = hand_model.get_trimesh_data(idx)
                obj_mesh = object_model.object_mesh_list[obj_idx].copy()
                scale = object_model.object_scale_tensor.flatten()[idx].item()
                obj_mesh.apply_scale(scale)

                fig = go.Figure()

                # Hand mesh
                fig.add_trace(go.Mesh3d(
                    x=hand_mesh.vertices[:, 0],
                    y=hand_mesh.vertices[:, 1],
                    z=hand_mesh.vertices[:, 2],
                    i=hand_mesh.faces[:, 0],
                    j=hand_mesh.faces[:, 1],
                    k=hand_mesh.faces[:, 2],
                    color="lightblue", opacity=0.7, name="hand",
                ))

                # Object mesh
                fig.add_trace(go.Mesh3d(
                    x=obj_mesh.vertices[:, 0],
                    y=obj_mesh.vertices[:, 1],
                    z=obj_mesh.vertices[:, 2],
                    i=obj_mesh.faces[:, 0],
                    j=obj_mesh.faces[:, 1],
                    k=obj_mesh.faces[:, 2],
                    color="lightyellow", opacity=0.7, name="object",
                ))

                fig.update_layout(
                    title=f"{code} grasp #{grasp_idx} (E_pen={E_pen[idx]:.4f})",
                    scene=dict(aspectmode="data"),
                )

                html_path = output_dir / f"{code}_grasp{grasp_idx}.html"
                fig.write_html(str(html_path))
                print(f"  Saved: {html_path}")

    except ImportError:
        print("[WARN] plotly not available — skipping visualization")

    # Summary
    print(f"\n=== Smoke Test Results ===")
    print(f"Objects: {args.object_codes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Final energy: {energy.mean():.4f}")
    print(f"E_pen range: [{E_pen.min():.4f}, {E_pen.max():.4f}]")
    print(f"Output: {output_dir}/")

    if E_pen.mean() > 0.01:
        print("\n[WARN] Mean penetration energy is high. Check object scale/URDF compatibility.")
    else:
        print("\n[OK] Smoke test passed. Proceed to Phase 3.")


def main():
    parser = argparse.ArgumentParser(description="KISTAR quick sanity check")
    parser.add_argument("--meshdata_path", type=str, required=True)
    parser.add_argument("--object_codes", nargs="+",
                        default=["contactdb-apple", "ycb-apple"])
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="smoke_results")
    args = parser.parse_args()
    run_smoke_test(args)


if __name__ == "__main__":
    main()
