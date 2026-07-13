"""bake_moves — normalize generated move npz clips (movegen.py output) and
export per-move motion JSONs + a manifest, ready for the ALIGN-stage runtime
(retarget.js / certify.mjs) or any engine importer (BAKE.md §5).

Normalization: rotate/translate every clip so frame 0 has the pelvis at the
origin with heading 0 (mujoco +X). Guarantees all baked moves share one
canonical start, whatever world frame their stance keyframe came from.

Trimming: a messy recovery segment (a hop or an underground scuff after the
move's real ending) is cut at bake time — `--trim MOVE=A:B` (python half-open
slice, either end may be blank) or a move-level `"trim": [A, B]` in the spec
(B null = to the end; the CLI overrides the spec). Trims apply BEFORE
canonicalization so frame 0 of the trimmed clip is the canonical origin, and
the move's frame data shifts with the cut.

The manifest records each move's fps and the spec's `loop` flag alongside
frames/frame data/gates, so runtimes and prebake never re-derive which clips
loop.

Usage (from GR00T-WholeBodyControl/motionbricks/, no GPU needed — pure
mujoco FK):
  python bake_moves.py [--in-dir out/moves] [--out-dir baked]
                       [--spec moves_example.json] [--trim slide=0:30 ...]
Moves are exported in spec order when --spec is readable, else every npz in
--in-dir sorted by name.
"""
import argparse
import glob
import json
import os

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from make_motion_json import to_three_pos, to_three_quat, MAP

XML = "assets/skeletons/g1/scene_29dof.xml"


def load_spec_moves(spec_path):
    """{name: move-spec} in spec order, or None when the spec is unreadable."""
    if spec_path and os.path.exists(spec_path):
        with open(spec_path) as fp:
            return {m["name"]: m for m in json.load(fp)["moves"]}
    return None


def move_order(in_dir, spec_moves):
    if spec_moves is not None:
        return list(spec_moves)
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(in_dir, "*.npz")))


def parse_trim(pairs):
    """--trim MOVE=A:B -> {move: [A, B|None]} (half-open, blank end = open)."""
    out = {}
    for kv in pairs:
        name, eq, rng = kv.partition("=")
        s, colon, e = rng.partition(":")
        if not eq or not colon:
            raise SystemExit(f"--trim expects MOVE=A:B, got: {kv}")
        out[name] = [int(s) if s else 0, int(e) if e else None]
    return out


def shift_frame_data(fd, t0, frames):
    """Re-express frame data after dropping t0 head frames and clamping to
    the trimmed length. Keys beyond the timing triple (contact, strike_tip,
    …) pass through — contact shifted like the window it sits in."""
    if not fd:
        return fd
    a0 = max(0, min(frames, fd["active"][0] - t0))
    a1 = max(a0, min(frames, fd["active"][1] - t0))
    out = {**fd, "startup": max(0, min(frames, fd["startup"] - t0)),
           "active": [a0, a1], "recovery": frames - a1}
    if fd.get("contact") is not None:
        out["contact"] = max(a0, min(a1, fd["contact"] - t0))
    return out


def canonicalize(qpos):
    """Frame-0 pelvis -> origin, heading -> 0 (yaw about Z). qpos (F, 36) WXYZ."""
    q = qpos.copy()
    rot0 = R.from_quat(q[0, 3:7][[1, 2, 3, 0]])  # wxyz -> xyzw
    yaw0 = rot0.as_euler("zyx")[0]
    unyaw = R.from_euler("z", -yaw0)
    p0 = q[0, :3] * [1.0, 1.0, 0.0]
    q[:, :3] = unyaw.apply(q[:, :3] - p0)
    rots = R.from_quat(q[:, 3:7][:, [1, 2, 3, 0]])
    out = (unyaw * rots).as_quat()          # xyzw
    q[:, 3:7] = out[:, [3, 0, 1, 2]]        # -> wxyz
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="out/moves")
    ap.add_argument("--out-dir", default="baked")
    ap.add_argument("--spec", default="moves_example.json")
    ap.add_argument("--trim", action="append", default=[], metavar="MOVE=A:B",
                    help="keep frames [A:B) of MOVE (before canonicalizing); "
                         "repeatable; overrides the spec's per-move \"trim\"")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    spec_moves = load_spec_moves(a.spec)
    cli_trim = parse_trim(a.trim)

    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    nbody = m.nbody
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(nbody)]
    parents = [int(m.body_parentid[i]) for i in range(nbody)]

    mujoco.mj_resetData(m, d)
    d.qpos[:] = 0.0
    d.qpos[0:3] = [0.0, 0.0, 0.8]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)
    rest_xpos, rest_xquat = d.xpos.copy(), d.xquat.copy()

    manifest = []
    for name in move_order(a.in_dir, spec_moves):
        src = os.path.join(a.in_dir, f"{name}.npz")
        if not os.path.exists(src):
            print(f"[skip] {name} (no npz)")
            continue
        mv = (spec_moves or {}).get(name, {})
        trim = cli_trim.get(name, mv.get("trim"))
        z = np.load(src, allow_pickle=True)
        q = z["qpos"]
        t0 = 0
        if trim:
            t0 = int(trim[0] or 0)
            q = q[t0:(int(trim[1]) if trim[1] is not None else None)]
            if not len(q):
                raise SystemExit(f"[bake] trim {trim} leaves no frames for {name}")
        q = canonicalize(q)
        F = len(q)
        xpos = np.zeros((F, nbody, 3), dtype=np.float32)
        xquat = np.zeros((F, nbody, 4), dtype=np.float32)
        for i in range(F):
            d.qpos[:] = q[i]
            mujoco.mj_forward(m, d)
            xpos[i] = d.xpos
            xquat[i] = d.xquat

        p3, q3 = to_three_pos(xpos), to_three_quat(xquat)
        out = {
            "fps": int(z["fps"]), "numFrames": F, "mode": name,
            "names": names, "parents": parents,
            "pos": np.round(p3, 4).tolist(),
            "rest": np.round(to_three_pos(rest_xpos), 4).tolist(),
            "quat": np.round(q3, 5).tolist(),
            "restQuat": np.round(to_three_quat(rest_xquat), 5).tolist(),
            "mapSource": list(MAP.values()), "mapTarget": list(MAP.keys()),
        }
        with open(os.path.join(a.out_dir, f"{name}.json"), "w") as fp:
            json.dump(out, fp)

        meta_path = os.path.join(a.in_dir, f"{name}.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        fd = meta.get("frame_data")
        if trim:
            fd = shift_frame_data(fd, t0, F)
        entry = {"name": name, "file": f"{name}.json", "frames": F,
                 "fps": int(z["fps"]), "loop": bool(mv.get("loop", False)),
                 "frame_data": fd,
                 "gates": {k: round(v, 4) for k, v in
                           meta.get("gates", {}).items()
                           if isinstance(v, (int, float))}}
        if trim:
            entry["trim"] = [t0, trim[1]]
        manifest.append(entry)
        print(f"[bake] {name}: {F} frames"
              + (f" (trimmed {t0}:{'' if trim[1] is None else trim[1]})" if trim else "")
              + (" loop" if entry["loop"] else ""))

    with open(os.path.join(a.out_dir, "manifest.json"), "w") as fp:
        json.dump({"moves": manifest}, fp, indent=1)
    print(f"[bake] manifest: {len(manifest)} moves -> {a.out_dir}/manifest.json")


if __name__ == "__main__":
    main()
