#!/usr/bin/env python3
"""
Phase 1: Convert DexGrasp_KIST/cad/{contactdb,ycb}/ objects to DexGraspNet meshdata format.

Input per object:
    cad/{source}/{name}/coacd_allinone.obj   (merged convex decomposition)
    cad/{source}/{name}/{name}.stl           (original visual mesh)

Output per object:
    data/meshdata/{source}-{name}/coacd/
        decomposed.obj             (copy of allinone)
        coacd_convex_piece_0.obj   (individual convex pieces)
        coacd_convex_piece_1.obj
        ...
        coacd.urdf                 (multi-link URDF for Isaac Gym)

Usage:
    python dexgraspnet/scripts/cad_to_meshdata.py \
        --cad_root /path/to/DexGrasp_KIST/cad \
        --output_root /path/to/data/meshdata \
        [--sources contactdb ycb] \
        [--object_codes_out object_codes.txt]
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

import numpy as np
import trimesh as tm
import lxml.etree as et


def split_multibody_obj(obj_path):
    """Load a multi-body OBJ and split into individual convex pieces."""
    scene_or_mesh = tm.load(str(obj_path), process=False)

    if isinstance(scene_or_mesh, tm.Scene):
        pieces = []
        for name in scene_or_mesh.geometry:
            pieces.append(scene_or_mesh.geometry[name])
        return pieces
    elif isinstance(scene_or_mesh, tm.Trimesh):
        splits = list(scene_or_mesh.split())
        if len(splits) > 1:
            return splits
        return [scene_or_mesh]
    else:
        raise ValueError(f"Unexpected type from trimesh.load: {type(scene_or_mesh)}")


def generate_urdf(pieces, object_name):
    """Generate a multi-link URDF string from convex pieces (matches DexGraspNet format)."""
    root = et.Element("robot", name="root")
    prev_link_name = None

    for i, piece in enumerate(pieces):
        piece_name = f"coacd_convex_piece_{i}"
        piece_filename = f"{piece_name}.obj"
        link_name = f"link_{piece_name}"

        try:
            I = piece.moment_inertia
        except Exception:
            I = np.eye(3) * 1e-6

        link = et.SubElement(root, "link", name=link_name)

        inertial = et.SubElement(link, "inertial")
        et.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        et.SubElement(
            inertial, "inertia",
            ixx=f"{I[0][0]:.2E}", ixy=f"{I[0][1]:.2E}", ixz=f"{I[0][2]:.2E}",
            iyy=f"{I[1][1]:.2E}", iyz=f"{I[1][2]:.2E}", izz=f"{I[2][2]:.2E}",
        )

        visual = et.SubElement(link, "visual")
        et.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
        geom_v = et.SubElement(visual, "geometry")
        et.SubElement(
            geom_v, "mesh", filename=piece_filename,
            scale="1.0000E+00 1.0000E+00 1.0000E+00",
        )
        material = et.SubElement(visual, "material", name="")
        et.SubElement(material, "color", rgba="7.50E-01 7.50E-01 7.50E-01 1")

        collision = et.SubElement(link, "collision")
        et.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
        geom_c = et.SubElement(collision, "geometry")
        et.SubElement(
            geom_c, "mesh", filename=piece_filename,
            scale="1.0000E+00 1.0000E+00 1.0000E+00",
        )

        if prev_link_name is not None:
            joint = et.SubElement(
                root, "joint",
                name=f"{link_name}_joint", type="fixed",
            )
            et.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
            et.SubElement(joint, "parent", link=prev_link_name)
            et.SubElement(joint, "child", link=link_name)

        prev_link_name = link_name

    tree = et.ElementTree(root)
    return et.tostring(root, pretty_print=True, xml_declaration=False).decode("utf-8")


def convert_one_object(cad_dir, output_dir, source, name):
    """Convert a single object from cad format to meshdata format.

    Returns:
        object_code: str (e.g. "contactdb-apple")
        n_pieces: int
    """
    object_code = f"{source}-{name}"
    cad_path = Path(cad_dir) / source / name
    out_path = Path(output_dir) / object_code / "coacd"

    allinone_obj = cad_path / "coacd_allinone.obj"
    if not allinone_obj.exists():
        print(f"  [SKIP] {object_code}: coacd_allinone.obj not found")
        return None, 0

    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load and normalize (max bounding box dim → 1.0, centered at origin)
    full_mesh = tm.load(str(allinone_obj), force="mesh", process=False)
    max_dim = full_mesh.bounding_box.extents.max()
    scale_factor = 1.0 / max_dim if max_dim > 0 else 1.0
    center = full_mesh.centroid

    def normalize_mesh(mesh):
        mesh.vertices = (mesh.vertices - center) * scale_factor
        return mesh

    normalize_mesh(full_mesh)
    full_mesh.export(str(out_path / "decomposed.obj"))

    # 2. Split into individual convex pieces (normalize each)
    pieces = split_multibody_obj(allinone_obj)

    for i, piece in enumerate(pieces):
        normalize_mesh(piece)
        piece_path = out_path / f"coacd_convex_piece_{i}.obj"
        piece.export(str(piece_path))

    # 3. Generate multi-link URDF
    urdf_str = generate_urdf(pieces, object_code)
    urdf_path = out_path / "coacd.urdf"
    with open(str(urdf_path), "w") as f:
        f.write(urdf_str)

    return object_code, len(pieces)


def main():
    parser = argparse.ArgumentParser(description="Convert cad/ objects to DexGraspNet meshdata format")
    parser.add_argument("--cad_root", type=str, required=True,
                        help="Path to DexGrasp_KIST/cad/")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Output path for meshdata/")
    parser.add_argument("--sources", nargs="+", default=["contactdb", "ycb"],
                        help="Object source directories to process")
    parser.add_argument("--object_codes_out", type=str, default=None,
                        help="Write object code list to this file")
    parser.add_argument("--object_names", nargs="*", default=None,
                        help="Process only these object names (default: all)")
    args = parser.parse_args()

    cad_root = Path(args.cad_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_codes = []
    total = 0
    errors = 0

    for source in args.sources:
        source_dir = cad_root / source
        if not source_dir.exists():
            print(f"[WARN] Source directory not found: {source_dir}")
            continue

        names = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])
        if args.object_names:
            names = [n for n in names if n in args.object_names]

        print(f"\n=== Processing {source}: {len(names)} objects ===")

        for name in names:
            total += 1
            try:
                code, n_pieces = convert_one_object(
                    str(cad_root), str(output_root), source, name
                )
                if code:
                    all_codes.append(code)
                    print(f"  [{total:3d}] {code}: {n_pieces} convex pieces")
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                print(f"  [ERR] {source}-{name}: {e}")

    print(f"\n=== Summary ===")
    print(f"Total: {total}, Success: {len(all_codes)}, Errors: {errors}")
    print(f"Output: {output_root}")

    if args.object_codes_out:
        codes_path = Path(args.object_codes_out)
        codes_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(codes_path), "w") as f:
            for code in sorted(all_codes):
                f.write(code + "\n")
        print(f"Object codes written to: {codes_path}")


if __name__ == "__main__":
    main()
