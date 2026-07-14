"""kimoconstraints — move-spec constraint authoring, validation, and provenance
for Kimodo generation (the constraint half of kimogen.py).

Everything here runs BEFORE the diffusion model is loaded: schema validation
is pure numpy; resolution (FK of authored poses into world targets) needs only
the kimodo skeleton classes, not the model weights.

Authoring frame vs baked frame
------------------------------
Constraints are authored in Kimodo's NATIVE frame: Y-up, meters, heading 0
faces +Z, generation starts with the (smoothed) root at the XZ origin.
Accepted clips are later canonicalized (frame-0 root at origin, facing +X)
and possibly loop-trimmed; `transform_records` / `shift_records` re-express
the resolved constraint targets with the SAME rigid transform and trim the
motion got, so baked records line up with baked frames by construction.

Upstream schema (verified against the installed Kimodo package)
---------------------------------------------------------------
- types: root2d | fullbody | end-effector | left-hand | right-hand |
  left-foot | right-foot
- `end-effector.joint_names` entries are base-EE tokens (CamelCase):
  LeftHand, RightHand, LeftFoot, RightFoot. (`Hips` is upstream-legal but this
  wrapper rejects it: use root2d/fullbody for root intent.)
- fullbody / end-effector carry a complete SOMA pose per constrained frame
  (`local_joints_rot` [T,30|77,3] axis-angle + `root_positions` [T,3]); the
  world-space targets are the FK of that pose. An end-effector constraint
  additionally pins the smoothed root XZ, the root height, and the heading at
  its frames (upstream `update_constraints` writes all three), which is why
  conflict checks below compare root data across families.
- generic `end-effector` constraints CONDITION the prediction but get no
  MotionCorrection mask (only the four shorthand types and fullbody/root2d
  do); this wrapper therefore normalizes `end-effector` into the equivalent
  shorthand constraints so post-processing corrects them.
"""
import json
import os
from typing import NoReturn

import numpy as np

EE_TOKENS = ("LeftHand", "RightHand", "LeftFoot", "RightFoot")
EE_SHORTHAND = {"left-hand": "LeftHand", "right-hand": "RightHand",
                "left-foot": "LeftFoot", "right-foot": "RightFoot"}
SHORTHAND_OF = {v: k for k, v in EE_SHORTHAND.items()}
KNOWN_TYPES = ("root2d", "fullbody", "end-effector", *EE_SHORTHAND)
# upstream guidance: constrained frames per type should stay under 20
# (documented, not enforced upstream; root2d dense paths are exempt)
SPARSE_FRAME_LIMIT = 20
SOMA_JOINT_COUNTS = (30, 77)
MAX_ROOT_SPEED = 5.0          # m/s — cheap reachability gate on root targets
MAX_ROTVEC = 2 * np.pi + 1e-6  # axis-angle magnitude cap (catches degrees-as-radians)
ROOT_Y_RANGE = (0.05, 2.0)    # plausible SOMA hip heights, meters
XZ_AGREE_TOL = 1e-4           # required agreement of overlapping root XZ pins
POSE_AGREE_TOL = 1e-5         # required agreement of overlapping pose rows
HEADING_AGREE_DEG = 2.0       # required agreement of overlapping heading pins

# wrapper-only per-constraint fields (stripped before upstream save)
WRAPPER_FIELDS = {"required"}


class ConstraintSpecError(ValueError):
    """A move's constraint spec failed validation. Message carries
    move/constraint/frame context so authors can locate the defect."""


def _err(move, i, ctype, msg) -> NoReturn:
    where = f"{move}: constraint #{i}" + (f" ({ctype})" if ctype else "")
    raise ConstraintSpecError(f"{where}: {msg}")


def _as_float_array(move, i, ctype, value, name, shape_desc):
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        _err(move, i, ctype, f"{name} must be a numeric array {shape_desc}")
    if not np.isfinite(arr).all():
        _err(move, i, ctype, f"{name} contains non-finite values")
    return arr


def _validate_frames(move, i, ctype, frames, num_frames):
    if (not isinstance(frames, list) or not frames
            or any(isinstance(f, bool) or not isinstance(f, int) for f in frames)):
        _err(move, i, ctype, "frame_indices must be a non-empty list of integers")
    if any(f < 0 or f >= num_frames for f in frames):
        _err(move, i, ctype,
             f"frame_indices must lie in [0, {num_frames - 1}] "
             f"(move duration gives {num_frames} frames); got {frames}")
    if any(b <= a for a, b in zip(frames, frames[1:])):
        _err(move, i, ctype, f"frame_indices must be sorted and unique; got {frames}")


def _validate_pose_fields(move, i, ctype, c, T):
    rot = _as_float_array(move, i, ctype, c.get("local_joints_rot"),
                          "local_joints_rot", "[T,30|77,3] (axis-angle, radians)")
    if rot.ndim != 3 or rot.shape[0] != T or rot.shape[1] not in SOMA_JOINT_COUNTS \
            or rot.shape[2] != 3:
        _err(move, i, ctype,
             f"local_joints_rot must have shape [{T},30|77,3] (one complete "
             f"SOMA pose per constrained frame); got {list(rot.shape)}")
    mag = np.linalg.norm(rot, axis=-1)
    if (mag > MAX_ROTVEC).any():
        t, j = np.unravel_index(mag.argmax(), mag.shape)
        _err(move, i, ctype,
             f"local_joints_rot frame {c['frame_indices'][t]} joint {j} has "
             f"axis-angle magnitude {mag[t, j]:.2f} rad > 2π — rotations must "
             "be radians, not degrees")
    root = _as_float_array(move, i, ctype, c.get("root_positions"),
                           "root_positions", "[T,3] (meters)")
    if root.shape != (T, 3):
        _err(move, i, ctype, f"root_positions must have shape [{T},3]; got {list(root.shape)}")
    lo, hi = ROOT_Y_RANGE
    if (root[:, 1] < lo).any() or (root[:, 1] > hi).any():
        _err(move, i, ctype,
             f"root_positions Y must be a plausible hip height in [{lo}, {hi}] m; "
             f"got {root[:, 1].round(3).tolist()}")
    smooth = None
    if "smooth_root_2d" in c:
        smooth = _as_float_array(move, i, ctype, c["smooth_root_2d"],
                                 "smooth_root_2d", "[T,2] (ground XZ, meters)")
        if smooth.shape != (T, 2):
            _err(move, i, ctype,
                 f"smooth_root_2d must have shape [{T},2]; got {list(smooth.shape)}")
    return rot, root, smooth


def _validate_one(move, i, c, num_frames):
    """Validate a single raw constraint dict; returns its type."""
    if not isinstance(c, dict):
        _err(move, i, None, "every constraint must be a JSON object")
    if "required" in c and not isinstance(c["required"], bool):
        _err(move, i, c.get("type"), "'required' must be boolean")
    ctype = c.get("type")
    if ctype not in KNOWN_TYPES:
        _err(move, i, None,
             f"unknown constraint type {ctype!r}; supported: {', '.join(KNOWN_TYPES)}")
    _validate_frames(move, i, ctype, c.get("frame_indices"), num_frames)
    T = len(c["frame_indices"])

    if ctype == "root2d":
        allowed = {"type", "frame_indices", "smooth_root_2d", "global_root_heading"} | WRAPPER_FIELDS
        extra = set(c) - allowed
        if extra:
            _err(move, i, ctype, f"unsupported fields for root2d: {sorted(extra)}")
        xz = _as_float_array(move, i, ctype, c.get("smooth_root_2d"),
                             "smooth_root_2d", "[T,2] (ground XZ, meters)")
        if xz.shape != (T, 2):
            _err(move, i, ctype,
                 f"smooth_root_2d must have shape [{T},2]; got {list(xz.shape)}")
        if "global_root_heading" in c:
            h = _as_float_array(move, i, ctype, c["global_root_heading"],
                                "global_root_heading", "[T,2] ([cos t, sin t])")
            if h.shape != (T, 2):
                _err(move, i, ctype,
                     f"global_root_heading must have shape [{T},2]; got {list(h.shape)}")
            norms = np.linalg.norm(h, axis=-1)
            if (norms < 1e-6).any():
                _err(move, i, ctype,
                     f"global_root_heading contains a zero heading vector "
                     f"(frame {c['frame_indices'][int(norms.argmin())]})")
    else:
        allowed = {"type", "frame_indices", "local_joints_rot", "root_positions",
                   "smooth_root_2d"} | WRAPPER_FIELDS
        if ctype == "end-effector":
            allowed |= {"joint_names"}
        extra = set(c) - allowed
        if extra:
            _err(move, i, ctype, f"unsupported fields for {ctype}: {sorted(extra)}")
        _validate_pose_fields(move, i, ctype, c, T)
        if ctype == "end-effector":
            names = c.get("joint_names")
            if (not isinstance(names, list) or not names
                    or any(not isinstance(n, str) for n in names)):
                _err(move, i, ctype, "joint_names must be a non-empty list of strings")
            if len(set(names)) != len(names):
                _err(move, i, ctype, f"joint_names must be unique; got {names}")
            bad = [n for n in names if n not in EE_TOKENS]
            if bad:
                hint = ("'Hips' is not supported here — express root intent with "
                        "root2d or fullbody. " if "Hips" in bad else "")
                _err(move, i, ctype,
                     f"unsupported joint_names {bad}; {hint}supported end-effectors "
                     f"(exact casing): {', '.join(EE_TOKENS)}")
    return ctype


def normalize_constraints(move, constraints, num_frames, source):
    """Validate raw constraint dicts and normalize them to the exact upstream
    schema Kimodo's post-processing can correct:

    - `end-effector` entries are split into one shorthand constraint per
      joint name (generic end-effector constraints receive no
      MotionCorrection mask upstream; the shorthands do);
    - heading vectors are normalized to unit [cos, sin];
    - wrapper-only fields are moved to a parallel `meta` list.

    Returns (normalized, meta): equal-length lists; `normalized[k]` is a
    clean upstream constraint dict, `meta[k]` records its authoring source.
    """
    if not isinstance(constraints, list) or not constraints:
        raise ConstraintSpecError(f"{move}: constraints must be a non-empty array")
    normalized, meta = [], []
    for i, c in enumerate(constraints):
        ctype = _validate_one(move, i, c, num_frames)
        required = c.get("required", True)
        base_meta = {"source": source, "originalIndex": i,
                     "originalType": ctype, "required": required}
        clean = {k: v for k, v in c.items() if k not in WRAPPER_FIELDS}
        if ctype == "end-effector":
            for token in c["joint_names"]:
                entry = {k: v for k, v in clean.items() if k != "joint_names"}
                entry["type"] = SHORTHAND_OF[token]
                normalized.append(entry)
                meta.append(dict(base_meta))
        else:
            if ctype == "root2d" and "global_root_heading" in clean:
                h = np.asarray(clean["global_root_heading"], dtype=float)
                h = h / np.linalg.norm(h, axis=-1, keepdims=True)
                clean = dict(clean, global_root_heading=h.round(9).tolist())
            normalized.append(clean)
            meta.append(dict(base_meta))
    return normalized, meta


# ------------------------------------------------------------------ conflicts

def _effective_xz(c):
    """The root-XZ values a constraint pins at its frames."""
    if c["type"] == "root2d":
        return np.asarray(c["smooth_root_2d"], dtype=float)
    if "smooth_root_2d" in c:
        return np.asarray(c["smooth_root_2d"], dtype=float)
    return np.asarray(c["root_positions"], dtype=float)[:, [0, 2]]


def check_conflicts(move, constraints, meta, fps):
    """Reject duplicate or contradictory constraints (never silently let one
    overwrite another). Rules follow what the upstream conditioning actually
    writes per frame:

    - two constraints of the same (normalized) type must not share a frame;
    - fullbody already pins every joint: no EE constraint may share its frames;
    - EE constraints of different effectors may share a frame ONLY with
      identical root data (they all pin smooth root XZ, root Y and heading);
    - root2d and pose constraints sharing a frame must agree on root XZ;
    - consecutive pinned root positions must be reachable (<= 5 m/s).
    """
    frames_by_type = {}
    for k, c in enumerate(constraints):
        seen = frames_by_type.setdefault(c["type"], {})
        for f in c["frame_indices"]:
            if f in seen:
                raise ConstraintSpecError(
                    f"{move}: two {c['type']} constraints (#{meta[seen[f]]['originalIndex']}"
                    f" and #{meta[k]['originalIndex']}) both constrain frame {f}; "
                    "merge them or change one frame — one would silently overwrite the other")
            seen[f] = k

    fb_frames = frames_by_type.get("fullbody", {})
    for k, c in enumerate(constraints):
        if c["type"] in EE_SHORTHAND:
            clash = [f for f in c["frame_indices"] if f in fb_frames]
            if clash:
                raise ConstraintSpecError(
                    f"{move}: constraint #{meta[k]['originalIndex']} ({c['type']}) and a "
                    f"fullbody constraint both constrain frame(s) {clash}; a fullbody "
                    "keyframe already pins every end-effector — drop one of them")

    # EE x EE frame sharing: root rows must be identical
    ee = [(k, c) for k, c in enumerate(constraints) if c["type"] in EE_SHORTHAND]
    for a in range(len(ee)):
        ka, ca = ee[a]
        fa = {f: t for t, f in enumerate(ca["frame_indices"])}
        for b in range(a + 1, len(ee)):
            kb, cb = ee[b]
            for tb, f in enumerate(cb["frame_indices"]):
                if f not in fa:
                    continue
                ta = fa[f]
                ra = np.asarray(ca["root_positions"], dtype=float)[ta]
                rb = np.asarray(cb["root_positions"], dtype=float)[tb]
                pa = np.asarray(ca["local_joints_rot"], dtype=float)[ta, 0]
                pb = np.asarray(cb["local_joints_rot"], dtype=float)[tb, 0]
                xa, xb = _effective_xz(ca)[ta], _effective_xz(cb)[tb]
                if (np.abs(ra - rb).max() > POSE_AGREE_TOL
                        or np.abs(xa - xb).max() > XZ_AGREE_TOL
                        or np.abs(pa - pb).max() > POSE_AGREE_TOL):
                    raise ConstraintSpecError(
                        f"{move}: constraints #{meta[ka]['originalIndex']} ({ca['type']}) and "
                        f"#{meta[kb]['originalIndex']} ({cb['type']}) share frame {f} but "
                        "disagree on the root position/heading their poses imply — every "
                        "end-effector constraint also pins the root at its frames, so "
                        "co-framed EE constraints must be derived from the same pose")

    # root2d vs pose constraints: XZ agreement on shared frames
    for kr, cr in enumerate(constraints):
        if cr["type"] != "root2d":
            continue
        rframes = {f: t for t, f in enumerate(cr["frame_indices"])}
        rxz = _effective_xz(cr)
        for k, c in enumerate(constraints):
            if c["type"] == "root2d":
                continue
            cxz = _effective_xz(c)
            for t, f in enumerate(c["frame_indices"]):
                if f in rframes and np.abs(rxz[rframes[f]] - cxz[t]).max() > XZ_AGREE_TOL:
                    raise ConstraintSpecError(
                        f"{move}: root2d constraint #{meta[kr]['originalIndex']} pins frame {f} "
                        f"at XZ {rxz[rframes[f]].round(4).tolist()} but constraint "
                        f"#{meta[k]['originalIndex']} ({c['type']}) pins the root at "
                        f"{cxz[t].round(4).tolist()} on the same frame — resolve the contradiction")

    # combined root timeline must be physically reachable
    pins = []
    for k, c in enumerate(constraints):
        xz = _effective_xz(c)
        for t, f in enumerate(c["frame_indices"]):
            pins.append((f, xz[t], meta[k]["originalIndex"], c["type"]))
    pins.sort(key=lambda p: p[0])
    for (f0, x0, i0, t0), (f1, x1, i1, t1) in zip(pins, pins[1:]):
        if f1 == f0:
            continue
        speed = float(np.linalg.norm(x1 - x0)) / ((f1 - f0) / fps)
        if speed > MAX_ROOT_SPEED:
            raise ConstraintSpecError(
                f"{move}: root targets at frames {f0} (constraint #{i0}, {t0}) and {f1} "
                f"(constraint #{i1}, {t1}) require {speed:.1f} m/s root travel "
                f"(> {MAX_ROOT_SPEED} m/s) — unreachable; spread the frames or move the targets")

    # sparse-frame guidance limit (root2d exempt: dense paths are supported)
    for ctype, seen in frames_by_type.items():
        if ctype == "root2d":
            continue
        if len(seen) >= SPARSE_FRAME_LIMIT:
            raise ConstraintSpecError(
                f"{move}: {len(seen)} constrained frames of type {ctype} — Kimodo's "
                f"guidance limit is fewer than {SPARSE_FRAME_LIMIT} per constraint type")


# ------------------------------------------------------------------- bookends

def build_stance_bookend(stance, num_frames):
    """The shared-stance fullbody keyframe at the first and last frame."""
    rot = stance["local_joints_rot30"]
    rp = stance["root_pos"]
    return {
        "type": "fullbody",
        "frame_indices": [0, num_frames - 1],
        "local_joints_rot": [rot, rot],
        "root_positions": [[0.0, rp[1], 0.0], [0.0, rp[1], 0.0]],
    }


def merge_stance_bookend(move, constraints, meta, stance, num_frames, fps):
    """Append the stance bookend to explicitly authored constraints, then
    re-check conflicts so an authored constraint on frame 0 / T-1 is reported
    against the bookend rather than silently overwritten."""
    bookend = build_stance_bookend(stance, num_frames)
    merged = constraints + [bookend]
    merged_meta = meta + [{"source": "stance_bookend", "originalIndex": len(meta),
                           "originalType": "fullbody", "required": True}]
    try:
        check_conflicts(move, merged, merged_meta, fps)
    except ConstraintSpecError as e:
        raise ConstraintSpecError(
            f"{e} — note: stance_bookend adds a fullbody constraint at frames "
            f"[0, {num_frames - 1}]; moves with explicit constraints on those frames "
            "must drop stance_bookend or move the constraint") from None
    return merged, merged_meta


def load_constraints_file(move, path, spec_dir):
    """Load a Kimodo constraints JSON (same format the upstream demo/API
    saves). Returns the raw constraint list."""
    full = path if os.path.isabs(path) else os.path.join(spec_dir, path)
    if not os.path.exists(full):
        raise ConstraintSpecError(f"{move}: constraints_file not found: {full}")
    with open(full) as fp:
        try:
            data = json.load(fp)
        except json.JSONDecodeError as e:
            raise ConstraintSpecError(f"{move}: constraints_file is not valid JSON: {e}") from None
    if not isinstance(data, list):
        raise ConstraintSpecError(
            f"{move}: constraints_file must contain a JSON array of constraints")
    return data


# ------------------------------------------------------------------ resolving

def _mat_to_xyzw(m):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_matrix(m).as_quat()


def resolve_records(move, constraints, meta, skeleton, post_processing=True):
    """FK-resolve normalized constraints into per-frame world-target records
    (native authoring frame) using the SAME upstream code path that will
    condition the model (`load_constraints_lst`), so wrapper provenance can
    never drift from what Kimodo actually saw.

    Provenance: every record conditions the prediction; families with a
    MotionCorrection mask (fullbody, the four EE shorthands, root2d) are also
    corrected when post-processing is on.
    """
    from kimodo.constraints import load_constraints_lst
    objs = load_constraints_lst([dict(c) for c in constraints], skeleton)
    bone_index = skeleton.bone_index
    ee_names = ("LeftHand", "RightHand", "LeftFoot", "RightFoot", "Head")
    records = []
    for c, obj, m in zip(constraints, objs, meta):
        ctype = c["type"]
        provenance = "conditioned+corrected" if post_processing else "conditioned"
        frames = [int(f) for f in obj.frame_indices.tolist()]
        base = {"family": ("end-effector" if ctype in EE_SHORTHAND else ctype),
                "type": ctype, "source": m["source"], "required": m["required"],
                "originalType": m["originalType"], "provenance": provenance}
        if ctype == "root2d":
            xz = obj.smooth_root_2d.numpy()
            heading = np.asarray(c["global_root_heading"], dtype=float) \
                if "global_root_heading" in c else None
            for t, f in enumerate(frames):
                rec = dict(base, frame=f, rootXZ=xz[t].round(6).tolist(),
                           posConstrained=True, rotConstrained=heading is not None)
                if heading is not None:
                    # store as a facing DIRECTION vector (x,z) = (sin t, cos t)
                    rec["facing"] = [round(float(heading[t, 1]), 6),
                                     round(float(heading[t, 0]), 6)]
                records.append(rec)
            continue
        pos = obj.global_joints_positions.numpy()
        rots = obj.global_joints_rots.numpy()
        xz = obj.smooth_root_2d.numpy()
        heading = obj.global_root_heading.numpy()
        for t, f in enumerate(frames):
            rec = dict(base, frame=f,
                       rootXZ=xz[t].round(6).tolist(),
                       rootY=round(float(obj.root_y_pos[t]), 6),
                       facing=[round(float(heading[t, 1]), 6),
                               round(float(heading[t, 0]), 6)],
                       posConstrained=True, rotConstrained=True)
            if ctype in EE_SHORTHAND:
                role = EE_SHORTHAND[ctype]
                j = bone_index[role]
                rec["role"] = role
                rec["pos"] = pos[t, j].round(6).tolist()
                rec["quat"] = _mat_to_xyzw(rots[t, j]).round(6).tolist()
            else:  # fullbody
                rec["rootPos"] = pos[t, skeleton.root_idx].round(6).tolist()
                rec["localRot"] = np.asarray(
                    c["local_joints_rot"], dtype=float)[t].round(6).tolist()
                rec["ee"] = {}
                for name in ee_names:
                    if name not in bone_index:
                        continue
                    j = bone_index[name]
                    rec["ee"][name] = {
                        "pos": pos[t, j].round(6).tolist(),
                        "quat": _mat_to_xyzw(rots[t, j]).round(6).tolist(),
                    }
            records.append(rec)

    # second-tier conflict check that needs FK: heading agreement on frames
    # where both a root2d heading and a pose-implied heading are pinned
    by_frame = {}
    for rec in records:
        if "facing" in rec:
            by_frame.setdefault(rec["frame"], []).append(rec)
    for f, recs in by_frame.items():
        for a in range(len(recs)):
            for b in range(a + 1, len(recs)):
                va = np.array(recs[a]["facing"])
                vb = np.array(recs[b]["facing"])
                cosang = float(np.clip(np.dot(va, vb), -1.0, 1.0))
                if np.degrees(np.arccos(cosang)) > HEADING_AGREE_DEG:
                    raise ConstraintSpecError(
                        f"{move}: frame {f} is pinned to two different headings "
                        f"({recs[a]['type']} vs {recs[b]['type']}, "
                        f"{np.degrees(np.arccos(cosang)):.1f}° apart) — every pose "
                        "constraint pins the heading its hip line implies; align the "
                        "authored heading with the pose or drop one pin")
    return records


# ------------------------------------------- canonicalization / trim handling

def compose_canonical(R1, p01, R2=None, p02=None):
    """Compose kimogen's canonicalization steps into one rigid transform
    (x -> R @ x + t). Step 1: raw -> canonical; optional step 2: the
    re-canonicalization of a loop-trimmed clip."""
    R1, p01 = np.asarray(R1, dtype=float), np.asarray(p01, dtype=float)
    if R2 is None:
        return R1.copy(), -R1 @ p01
    R2, p02 = np.asarray(R2, dtype=float), np.asarray(p02, dtype=float)
    Rt = R2 @ R1
    return Rt, -(Rt @ p01 + R2 @ p02)


def transform_records(records, R, t):
    """Re-express resolved records under a rigid transform x -> R@x + t
    (R must be a yaw rotation with t.y == 0, which is what canonicalization
    produces: root heights and rotations about Y stay meaningful)."""
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float)
    if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() \
            or not np.isfinite(t).all():
        raise ValueError("canonical transform requires a finite 3x3 rotation and vec3 translation")
    if (not np.allclose(R.T @ R, np.eye(3), atol=1e-9)
            or not np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
            or not np.allclose(R[1], [0.0, 1.0, 0.0], atol=1e-9)
            or not np.allclose(R[:, 1], [0.0, 1.0, 0.0], atol=1e-9)
            or abs(float(t[1])) > 1e-9):
        raise ValueError("canonical transform must be a yaw rotation with no Y offset")
    from scipy.spatial.transform import Rotation as Rot
    qR = Rot.from_matrix(R)
    out = []
    for rec in records:
        r = dict(rec)
        if "rootXZ" in r:
            p = R @ np.array([r["rootXZ"][0], 0.0, r["rootXZ"][1]]) + t
            r["rootXZ"] = [round(float(p[0]), 6), round(float(p[2]), 6)]
        if "facing" in r:
            v = R @ np.array([r["facing"][0], 0.0, r["facing"][1]])
            r["facing"] = [round(float(v[0]), 6), round(float(v[2]), 6)]
        for key in ("pos", "rootPos"):
            if key in r:
                p = R @ np.asarray(r[key], dtype=float) + t
                r[key] = p.round(6).tolist()
        if "quat" in r:
            q = (qR * Rot.from_quat(r["quat"])).as_quat()
            r["quat"] = q.round(6).tolist()
        if "localRot" in r:
            lr = np.asarray(r["localRot"], dtype=float)
            root_rot = (qR * Rot.from_rotvec(lr[0])).as_rotvec()
            lr = lr.copy()
            lr[0] = root_rot
            r["localRot"] = lr.round(6).tolist()
        if "ee" in r:
            r["ee"] = {
                name: {
                    "pos": (R @ np.asarray(v["pos"], dtype=float) + t).round(6).tolist(),
                    "quat": (qR * Rot.from_quat(v["quat"])).as_quat().round(6).tolist(),
                } for name, v in r["ee"].items()}
        out.append(r)
    return out


def shift_records(move, records, i0, i1):
    """Apply a loop trim [i0, i1] (inclusive) to resolved records: shift
    retained frames, drop non-required records outside the interval, and
    REJECT the trim if it would remove a required constraint."""
    kept, dropped_required = [], []
    for rec in records:
        if i0 <= rec["frame"] <= i1:
            r = dict(rec)
            r["frame"] = rec["frame"] - i0
            kept.append(r)
        elif rec["required"]:
            dropped_required.append(rec)
    if dropped_required:
        desc = ", ".join(f"{r['type']}@{r['frame']}" for r in dropped_required[:6])
        raise ConstraintSpecError(
            f"{move}: loop trim [{i0}, {i1}] would remove required constraint(s) "
            f"{desc} — mark them \"required\": false, disable loop, or re-author "
            "the constraint inside the loop window")
    return kept


# ----------------------------------------------------------------- entrypoint

def prepare_move_constraints(mv, num_frames, fps, stance, spec_dir):
    """Everything kimogen needs before touching the model, for one move:
    validate + normalize + merge the stance bookend + conflict-check.

    Returns (constraints, meta): upstream-schema dicts ready for
    `load_constraints_lst` / `save_constraints_lst`, and their authoring
    provenance. Either list may be empty (pure-text move).
    """
    move = mv["name"]
    inline = mv.get("constraints")
    cfile = mv.get("constraints_file")
    if inline is not None and cfile is not None:
        raise ConstraintSpecError(
            f"{move}: 'constraints' (inline) and 'constraints_file' are mutually "
            "exclusive — supply one")
    constraints, meta = [], []
    if inline is not None:
        constraints, meta = normalize_constraints(move, inline, num_frames, "inline")
    elif cfile is not None:
        raw = load_constraints_file(move, cfile, spec_dir)
        constraints, meta = normalize_constraints(move, raw, num_frames, "file")
    if constraints:
        check_conflicts(move, constraints, meta, fps)
    if mv.get("stance_bookend"):
        if stance is None:
            raise ConstraintSpecError(
                f"{move}: stance_bookend requires the extracted stance pose")
        constraints, meta = merge_stance_bookend(
            move, constraints, meta, stance, num_frames, fps)
    return constraints, meta
