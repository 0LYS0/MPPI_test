"""
urdf_parser.py — Parse a URDF file into a Three.js-friendly kinematic tree.

Returns a dict with:
  links : {name → {visuals: [{mesh_url, origin_xyz, origin_rpy, scale}]}}
  joints: [{name, type, parent, child, origin_xyz, origin_rpy, axis}]
  root_link: str

Mesh URL resolution rules
─────────────────────────
  /meshes/visual/link1.obj     → {base_url}/meshes/visual/link1.obj
  package://meshes/visual/link6.obj → {base_url}/meshes/visual/link6.obj
  meshes/visual/link6.obj      → {base_url}/meshes/visual/link6.obj

where base_url is the web URL prefix that maps to the URDF package directory.
"""
import xml.etree.ElementTree as ET
import re
from pathlib import Path


def _parse_vec3(s: str) -> list:
    return [float(x) for x in s.strip().split()]


def _resolve_mesh_url(filename: str, base_url: str) -> str:
    """Normalise a URDF mesh filename to a web-accessible URL."""
    fn = filename.strip()
    # ROS package:// scheme
    if fn.startswith("package://"):
        fn = fn[len("package://"):]
    # Absolute path starting with /
    elif fn.startswith("/"):
        fn = fn[1:]
    base = base_url.rstrip("/")
    return f"{base}/{fn}"


def parse_urdf(urdf_path: str, mesh_base_url: str = "/assets/franka") -> dict:
    """
    Parse URDF file and return kinematic tree as a serialisable dict.

    Parameters
    ----------
    urdf_path     : path to .urdf file on disk
    mesh_base_url : web URL prefix for mesh files
                    e.g. "/assets/franka_panda/franka_panda"
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links  = {}
    joints = []

    # ── Links ────────────────────────────────────────────────
    for link in root.findall("link"):
        name    = link.get("name")
        visuals = []

        for vis in link.findall("visual"):
            orig = vis.find("origin")
            geo  = vis.find("geometry")
            mat  = vis.find("material")

            xyz = _parse_vec3(orig.get("xyz", "0 0 0")) if orig is not None else [0,0,0]
            rpy = _parse_vec3(orig.get("rpy", "0 0 0")) if orig is not None else [0,0,0]

            mesh_url = None
            scale    = [1.0, 1.0, 1.0]

            if geo is not None:
                mesh_elem = geo.find("mesh")
                if mesh_elem is not None:
                    fn = mesh_elem.get("filename", "")
                    if fn:
                        mesh_url = _resolve_mesh_url(fn, mesh_base_url)
                    sc_str = mesh_elem.get("scale", None)
                    if sc_str:
                        sc_vals = _parse_vec3(sc_str)
                        scale = sc_vals if len(sc_vals) == 3 else [sc_vals[0]]*3

            # Material colour (RGBA or name)
            color = None
            if mat is not None:
                col_el = mat.find("color")
                if col_el is not None:
                    rgba = _parse_vec3(col_el.get("rgba", "0.8 0.8 0.8 1"))
                    color = {"r": rgba[0], "g": rgba[1],
                             "b": rgba[2], "a": rgba[3] if len(rgba)>3 else 1.0}

            visuals.append({
                "mesh_url":   mesh_url,
                "origin_xyz": xyz,
                "origin_rpy": rpy,
                "scale":      scale,
                "color":      color,
            })

        links[name] = {"visuals": visuals}

    # ── Joints ───────────────────────────────────────────────
    parent_map = {}  # child_link → parent_link
    children_map = {}

    for joint in root.findall("joint"):
        jname  = joint.get("name")
        jtype  = joint.get("type", "fixed")
        parent = joint.find("parent")
        child  = joint.find("child")
        orig   = joint.find("origin")
        axis   = joint.find("axis")
        limit  = joint.find("limit")

        parent_name = parent.get("link", "") if parent is not None else ""
        child_name  = child.get("link", "")  if child  is not None else ""

        xyz = _parse_vec3(orig.get("xyz", "0 0 0")) if orig is not None else [0,0,0]
        rpy = _parse_vec3(orig.get("rpy", "0 0 0")) if orig is not None else [0,0,0]
        ax  = _parse_vec3(axis.get("xyz", "0 0 1")) if axis is not None else [0,0,1]

        q_min = q_max = vel_max = None
        if limit is not None:
            try: q_min   = float(limit.get("lower", "-1e9"))
            except: pass
            try: q_max   = float(limit.get("upper",  "1e9"))
            except: pass
            try: vel_max = float(limit.get("velocity", "10"))
            except: pass

        joints.append({
            "name":        jname,
            "type":        jtype,
            "parent_link": parent_name,
            "child_link":  child_name,
            "origin_xyz":  xyz,
            "origin_rpy":  rpy,
            "axis":        ax,
            "q_min":       q_min,
            "q_max":       q_max,
            "vel_max":     vel_max,
        })

        parent_map[child_name]  = parent_name
        if parent_name not in children_map:
            children_map[parent_name] = []
        children_map[parent_name].append(child_name)

    # Find root link (no parent)
    all_children = set(parent_map.keys())
    root_link    = next((n for n in links if n not in all_children), "")

    return {
        "robot_name": root.get("name", "robot"),
        "root_link":  root_link,
        "links":      links,
        "joints":     joints,
    }


if __name__ == "__main__":
    import json, sys
    path     = sys.argv[1] if len(sys.argv) > 1 else "assets/franka_panda/franka_panda/model.urdf"
    base_url = sys.argv[2] if len(sys.argv) > 2 else "/assets/franka_panda/franka_panda"
    data     = parse_urdf(path, base_url)
    print(json.dumps(data, indent=2))
