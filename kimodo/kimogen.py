"""kimogen — move-set generation on NVIDIA Kimodo (SOMA skeleton).

The Stage 2 generation step (BAKE.md §3-4): each move is authored as any
compatible combination of a text prompt, arbitrary full-body keyframes,
hand/foot end-effector targets, and sparse/dense root-path constraints
(inline `constraints` or a Kimodo `constraints_file`, schema:
kimoconstraints.py), optionally bookended with the shared-stance fullbody
keyframe at both ends (so clips chain/crossfade in-game). Constraint-only
moves (no text) are supported by the installed API: an empty prompt is
explicitly zeroed in the text conditioning, not replaced by wording.
Each move is generated as a best-of-N batch and pushed through numeric QA
gates, including per-constraint adherence. Winner NPZ + gate/frame-data JSON
+ resolved canonical constraint records per move.

Usage (venv: kimenv; start `TEXT_ENCODER_DEVICE=cpu kimodo_textencoder` first
or let it fall back to an in-process CPU encoder):

  python kimogen.py gen   --spec moveset_mk.json [--only jab,sweep] [--samples 8] [--seed 42]
  python kimogen.py stance                    # extract stance pose from out/moves/idle_stance.npz
  python kimogen.py report                    # gate table of everything generated so far

Axis conventions (Kimodo): Y-up, XZ ground, heading angle t about Y with
facing dir (sin t, cos t) — i.e. heading 0 faces +Z. Verified by
validate_axes.py on a generated walk. Constraints are authored in this
native frame; gates evaluate in a canonical frame (frame-0 root at origin,
frame-0 facing rotated to +X = the baked-clip/game convention), and accepted
constraint records are re-expressed in that same canonical frame
(kimoconstraints.transform_records) so QA and runtimes never repeat FK.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

try:
    from . import kimoconstraints as kc
except ImportError:                      # direct script execution
    import kimoconstraints as kc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
MOVES_OUT = os.path.join(OUT, "moves")
STANCE_PATH = os.path.join(OUT, "stance_pose.json")

# somaskel77 indices are resolved at runtime from the skeleton class
EE_NAMES = ["LeftHand", "RightHand", "LeftFoot", "RightFoot", "Head"]
CONTACT_JOINTS = ["LeftFoot", "LeftToeBase", "LeftToeEnd",
                  "RightFoot", "RightToeBase", "RightToeEnd"]
CONTACT_JOINTS_4 = ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]
STANCE_MAX_ERROR = 0.22
FOOT_SKATE_MAX = 0.12
JITTER_MAX = 0.015
SAFE_MOVE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


# ---------------------------------------------------------------- geometry --
def canonicalize(joints, root_pos, idx):
    """Rotate/translate so frame-0 root sits at XZ origin facing +X.

    Frame-0 facing is derived from the hip line: fwd = cross(up, rHip-lHip)
    (verified on Kimodo outputs by validate_axes.py — walk-forward clips
    travel along this vector). Returns (joints', root', R).
    """
    if len(joints) == 0 or len(root_pos) != len(joints):
        raise ValueError("motion must contain equally sized, non-empty joint and root arrays")
    right = joints[0, idx["RightLeg"]] - joints[0, idx["LeftLeg"]]
    fwd = np.cross(np.array([0.0, 1.0, 0.0]), right)
    fwd[1] = 0.0
    fwd_len = np.linalg.norm(fwd)
    if not np.isfinite(fwd_len) or fwd_len < 1e-6:
        raise ValueError("cannot derive heading: frame-0 hip line is degenerate")
    fwd /= fwd_len
    c, s = fwd[0], fwd[2]                     # rot_y mapping fwd -> +X
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    p0 = root_pos[0] * np.array([1.0, 0.0, 1.0])
    j = (joints - p0) @ R.T
    r = (root_pos - p0) @ R.T
    return j, r, R


# ------------------------------------------------------------------- gates --
def failed_gates(reason, nonfinite=False):
    """Consistent diagnostic record for a sample that cannot be measured."""
    return {"nonfinite": bool(nonfinite), "malformed": reason,
            "foot_skate_mean": 1e9, "contact_frac": 0.0,
            "contact_ok": False, "foot_skate_ok": False,
            "jitter_mean": 1e9, "jitter_ok": False,
            "travel_x": 0.0, "travel_z": 0.0, "travel_ok": False,
            "apex_ok": False, "stance_ok": False,
            "pass": False, "score": 1e30}


def gate_sample(j, r, fc, idx, mv, stance, fps):
    """All-numeric QA for one canonicalized sample. Returns dict of metrics."""
    g = {}
    if fc.ndim != 2 or fc.shape[0] != len(j) or fc.shape[1] not in (4, 6):
        return failed_gates(f"foot_contacts shape {fc.shape}")
    g["nonfinite"] = bool(not np.isfinite(j).all() or not np.isfinite(r).all()
                          or not np.isfinite(fc).all())
    if g["nonfinite"]:
        return failed_gates("motion contains non-finite values", nonfinite=True)

    # foot skate: horizontal speed of contact-labeled foot joints (m/s)
    contact_names = CONTACT_JOINTS if fc.shape[1] == 6 else CONTACT_JOINTS_4
    cj = [idx[n] for n in contact_names]
    v = np.linalg.norm(np.diff(j[:, cj][:, :, [0, 2]], axis=0), axis=-1) * fps
    c = (fc[1:] > 0.5) & (fc[:-1] > 0.5)
    g["foot_skate_mean"] = float(v[c].mean()) if c.any() else 0.0
    g["contact_frac"] = float((fc > 0.5).any(axis=-1).mean())
    g["contact_ok"] = g["contact_frac"] >= 0.05 and bool(c.any())
    g["foot_skate_ok"] = g["foot_skate_mean"] <= FOOT_SKATE_MAX

    # jitter: mean 2nd difference of joint positions (m/frame^2)
    g["jitter_mean"] = float(np.linalg.norm(np.diff(j, n=2, axis=0), axis=-1).mean())
    g["jitter_ok"] = g["jitter_mean"] <= JITTER_MAX

    # net travel along the fight axis (+X = toward opponent)
    g["travel_x"] = float(r[-1, 0] - r[0, 0])
    g["travel_z"] = float(abs(r[-1, 2] - r[0, 2]))

    tv = mv.get("travel")
    if tv == "in_place":
        g["travel_ok"] = abs(g["travel_x"]) < 0.45 and g["travel_z"] < 0.45
    elif tv == "fwd":
        g["travel_ok"] = g["travel_x"] > 0.3 and g["travel_z"] < 0.6
    elif tv == "back":
        g["travel_ok"] = g["travel_x"] < -0.15 and g["travel_z"] < 0.6
    else:
        g["travel_ok"] = True

    # apex checks (root rise/dip, strike height, floor contact)
    apex = mv.get("apex")
    if apex:
        y0 = r[0, 1]
        if apex["kind"] == "root_rise":
            g["apex_val"] = float(r[:, 1].max() - y0)
        elif apex["kind"] == "root_dip":
            g["apex_val"] = float(y0 - r[:, 1].min())
        elif apex["kind"] == "root_floor":
            g["apex_val"] = float(r[:, 1].min())
        elif apex["kind"] == "ankle_height":
            g["apex_val"] = max(float(j[:, idx["LeftFoot"], 1].max()),
                                float(j[:, idx["RightFoot"], 1].max()))
        g["apex_ok"] = ((apex.get("min") is None or g["apex_val"] >= apex["min"])
                        and (apex.get("max") is None or g["apex_val"] <= apex["max"]))
    else:
        g["apex_ok"] = True

    # stance match at both ends (root-relative EE positions vs stance ref)
    if stance is not None and mv.get("stance_bookend"):
        ee = [idx[n] for n in EE_NAMES]
        ref = np.asarray(stance["ee_root_rel"])              # [5,3]
        def enderr(f):
            cur = j[f, ee] - r[f] * np.array([1.0, 0.0, 1.0])
            return float(np.linalg.norm(cur - ref, axis=-1).mean())
        g["stance_err_start"] = enderr(0)
        g["stance_err_end"] = enderr(-1)
        g["stance_ok"] = (g["stance_err_start"] <= STANCE_MAX_ERROR
                          and g["stance_err_end"] <= STANCE_MAX_ERROR)
    else:
        g["stance_ok"] = True

    g["pass"] = (not g["nonfinite"]) and g["contact_ok"] and g["foot_skate_ok"] \
        and g["jitter_ok"] and g["travel_ok"] and g["apex_ok"] and g["stance_ok"]
    # score: lower better; only meaningful among passing samples
    g["score"] = (g["foot_skate_mean"] * 2.0 + g["jitter_mean"] * 30.0
                  + (g.get("stance_err_start", 0.0)
                     + g.get("stance_err_end", 0.0)) * 1.5)
    return g


def frame_data(j, r, idx, mv, fps):
    """startup/active/recovery from strike-limb tip speed (game frame data)."""
    if not mv.get("strike"):
        return None
    tips = ["LeftHand", "RightHand"] if mv["strike"] == "hand" else \
           ["LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"]
    sp = [np.linalg.norm(np.diff(j[:, idx[n]], axis=0), axis=-1) * fps for n in tips]
    sp = np.stack(sp)                                        # [tips, T-1]
    tip = int(sp.max(axis=1).argmax())
    s = sp[tip]
    pk = int(s.argmax())
    thr = max(0.6 * s[pk], 1.0)                              # m/s
    a = pk
    while a > 0 and s[a - 1] > thr * 0.5:
        a -= 1
    b = pk
    while b < len(s) - 1 and s[b + 1] > thr * 0.5:
        b += 1
    active = [int(a), int(min(b + 2, len(s)))]
    # contact = the frame the strike visually lands: max extension of the
    # striking tip from the root inside the active window. The active window
    # alone is not enough for impact timing — it opens while the limb is
    # still travelling (generated motion has real wind-up), so events synced
    # to active[0] fire visibly early. Games should register the hit no
    # earlier than ~contact-2 and one-shot effects (damage, sfx, hitstop)
    # exactly at contact.
    ext = np.linalg.norm(j[:, idx[tips[tip]]] - r, axis=-1)
    contact = active[0] + int(ext[active[0]:active[1] + 1].argmax())
    return {"startup": int(a), "active": active, "contact": int(contact),
            "recovery": int(len(s) + 1 - active[1]),
            "strike_tip": tips[tip], "peak_speed": round(float(s[pk]), 2),
            "height": mv.get("height")}


def best_loop(j, r, min_len, max_len, must_span=None):
    """Find (i0, i1) trimming to the best pose-space cycle for loop moves.

    must_span=(lo, hi): only consider windows with i0 <= lo and i1 >= hi —
    used to keep required constrained frames inside the retained loop. Raises
    ValueError when no window can satisfy the span.
    """
    T = len(j)
    root_rel = j - r[:, None, :] * np.array([1.0, 0.0, 1.0])
    if T < 2:
        raise ValueError("a loop needs at least two frames")
    min_len = max(1, min(int(min_len), T - 1))
    max_len = max(min_len, min(int(max_len), T - 1))
    lo, hi = must_span if must_span else (None, None)
    best, pair = float("inf"), None
    for i in range(0, T - min_len):
        if lo is not None and i > lo:
            break
        jmax = min(T - 1, i + max_len)
        kmin = i + min_len if hi is None else max(i + min_len, hi)
        for k in range(kmin, jmax + 1):
            d = float(np.linalg.norm(root_rel[i] - root_rel[k], axis=-1).mean())
            if d < best:
                best, pair = d, (i, k)
    if pair is None:
        raise ValueError(
            f"no loop window of {min_len}..{max_len} frames can retain required "
            f"constrained frames {lo}..{hi}; disable loop or re-author the constraints")
    return pair, best


# ------------------------------------------------------------- constraints --
# Hard adherence gates on the winning sample, in the native/canonical frame
# (rigid-invariant). EE and root2d numbers are the accuracy contract
# (task: authored target -> final SOMA output); fullbody is gated on its
# hand/foot/head end-effector set as well as the root. This makes arbitrary
# authored key poses fail loudly if MotionCorrection does not land the pose.
CONSTRAINT_GATES = {
    "ee_pos_max": 0.005,        # m, constrained hand/foot position
    "ee_rot_max_deg": 2.0,      # deg, constrained hand/foot rotation
    "root_xz_max": 0.02,        # m, constrained root waypoint/path samples
    "fullbody_ee_pos_max": 0.005,  # m, fullbody keyframe EE positions
    "fullbody_ee_rot_max_deg": 2.0,  # deg, fullbody keyframe EE rotations
    "heading_max_deg": 25.0,    # deg, authored heading pins
}


def _rot_angle_deg(Ra, Rb):
    """Geodesic angle between two rotation matrices, degrees."""
    tr = float(np.trace(Ra.T @ Rb))
    return float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))))


def _quat_to_mat(q):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_quat(q).as_matrix()


def measure_constraints(records, pj, rp, grm, heading, idx):
    """Per-record adherence of ONE sample (native frame, post-processing
    already applied): position/rotation error at every constrained frame.
    Returns (per_record, summary); raises if a metric cannot be computed.
    """
    per = []
    agg = {"ee_pos_max": 0.0, "ee_rot_max_deg": 0.0, "root_xz_max": 0.0,
           "fullbody_ee_pos_max": 0.0, "fullbody_ee_rot_max_deg": 0.0,
           "fullbody_root_max": 0.0,
           "heading_max_deg": 0.0}
    have = set()
    for rec in records:
        f = rec["frame"]
        if f >= len(pj):
            raise ValueError(f"constraint frame {f} outside generated motion ({len(pj)} frames)")
        row = {"type": rec["type"], "frame": f, "source": rec["source"]}
        if rec["family"] == "root2d":
            e = float(np.linalg.norm(rp[f, [0, 2]] - np.asarray(rec["rootXZ"])))
            row["root_xz_err"] = round(e, 4)
            agg["root_xz_max"] = max(agg["root_xz_max"], e)
            have.add("root2d")
        elif rec["family"] == "end-effector":
            j = idx[rec["role"]]
            e = float(np.linalg.norm(pj[f, j] - np.asarray(rec["pos"])))
            a = _rot_angle_deg(grm[f, j], _quat_to_mat(rec["quat"]))
            row.update(role=rec["role"], pos_err=round(e, 4), rot_err_deg=round(a, 2))
            agg["ee_pos_max"] = max(agg["ee_pos_max"], e)
            agg["ee_rot_max_deg"] = max(agg["ee_rot_max_deg"], a)
            have.add("end-effector")
        else:  # fullbody
            re_ = float(np.linalg.norm(pj[f, idx["Hips"]] - np.asarray(rec["rootPos"])))
            ee_errs = {n: float(np.linalg.norm(pj[f, idx[n]] - np.asarray(v["pos"])))
                       for n, v in rec["ee"].items()}
            ee_rot_errs = {n: _rot_angle_deg(grm[f, idx[n]], _quat_to_mat(v["quat"]))
                           for n, v in rec["ee"].items()}
            row.update(root_err=round(re_, 4),
                       ee_pos_err={n: round(e, 4) for n, e in ee_errs.items()},
                       ee_rot_err_deg={n: round(e, 2) for n, e in ee_rot_errs.items()})
            agg["fullbody_root_max"] = max(agg["fullbody_root_max"], re_)
            agg["fullbody_ee_pos_max"] = max(agg["fullbody_ee_pos_max"], *ee_errs.values())
            agg["fullbody_ee_rot_max_deg"] = max(
                agg["fullbody_ee_rot_max_deg"], *ee_rot_errs.values())
            have.add("fullbody")
        if rec.get("facing") is not None and heading is not None:
            hv = heading[f] / (np.linalg.norm(heading[f]) + 1e-12)   # [cos t, sin t]
            fv = np.asarray(rec["facing"])                            # [sin t, cos t]
            cosang = float(np.clip(hv[1] * fv[0] + hv[0] * fv[1], -1.0, 1.0))
            a = float(np.degrees(np.arccos(cosang)))
            row["heading_err_deg"] = round(a, 2)
            agg["heading_max_deg"] = max(agg["heading_max_deg"], a)
        per.append(row)

    ok = True
    if "end-effector" in have:
        ok &= (agg["ee_pos_max"] <= CONSTRAINT_GATES["ee_pos_max"]
               and agg["ee_rot_max_deg"] <= CONSTRAINT_GATES["ee_rot_max_deg"])
    if "root2d" in have:
        ok &= agg["root_xz_max"] <= CONSTRAINT_GATES["root_xz_max"]
    if "fullbody" in have:
        ok &= (agg["fullbody_ee_pos_max"] <= CONSTRAINT_GATES["fullbody_ee_pos_max"]
               and agg["fullbody_ee_rot_max_deg"]
               <= CONSTRAINT_GATES["fullbody_ee_rot_max_deg"])
    if records:
        ok &= agg["heading_max_deg"] <= CONSTRAINT_GATES["heading_max_deg"]
    summary = {k: round(v, 4) for k, v in agg.items()}
    summary["constraint_ok"] = bool(ok)
    return per, summary


# --------------------------------------------------------------- generation --
def load_kimodo(modelname):
    import torch
    from kimodo import load_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(modelname, device=device, default_family="Kimodo")
    return model


def gen_move(model, mv, stance, prepared, samples, seed, steps, root_margin=0.01):
    from kimodo.constraints import load_constraints_lst
    from kimodo.tools import seed_everything

    npz_path = os.path.join(MOVES_OUT, f"{mv['name']}.npz")
    report_path = os.path.join(MOVES_OUT, f"{mv['name']}.json")
    constraint_path = os.path.join(MOVES_OUT, f"{mv['name']}.constraints.json")
    resolved_path = os.path.join(MOVES_OUT, f"{mv['name']}.resolved_constraints.json")
    # A regeneration attempt owns these outputs. Clear prior accepted files up
    # front so an exception or interrupted run cannot silently leave stale
    # motion available to the bake step.
    for stale in (npz_path, report_path, constraint_path, resolved_path):
        if os.path.exists(stale):
            os.remove(stale)

    fps = model.fps
    nF = int(mv["duration"] * fps)
    constraints, records = prepared["constraints"], prepared["records"]
    constraint_lst = []
    cfg_kwargs = {}
    if constraints:
        # the exact upstream-schema constraint file conditioning this move —
        # reloadable by Kimodo's own demo tools
        with open(constraint_path, "w") as fp:
            json.dump(constraints, fp)
        constraint_lst = load_constraints_lst(constraint_path, model.skeleton)
        cfg_kwargs = {"cfg_type": "separated", "cfg_weight": [2.0, 2.0]}

    # absence of text is represented as the empty prompt: the installed API
    # explicitly zeroes empty-string text features (constraint-only mode),
    # so no motion-changing placeholder wording is ever substituted
    prompt = mv.get("prompt") or ""
    seed_everything(seed)
    # root_margin: MotionCorrection's allowed root slack. The upstream default
    # (0.04 m) leaves corrected roots exactly 4 cm off authored waypoints —
    # measured, and outside the 2 cm adherence gate. 0.01 keeps a little
    # optimizer freedom for foot cleanup while hitting the gate; 0.0 is exact.
    out = model([prompt], [nF],
                constraint_lst=constraint_lst,
                num_denoising_steps=steps,
                num_samples=samples,
                multi_prompt=True,
                post_processing=True,
                root_margin=root_margin,
                return_numpy=True,
                **cfg_kwargs)

    required_out = {"posed_joints", "root_positions", "foot_contacts",
                    "global_rot_mats", "local_rot_mats"}
    missing_out = required_out - set(out)
    if missing_out:
        raise RuntimeError("Kimodo output is missing " + ", ".join(sorted(missing_out)))
    posed = np.asarray(out["posed_joints"])
    roots = np.asarray(out["root_positions"])
    contacts = np.asarray(out["foot_contacts"])
    global_rots = np.asarray(out["global_rot_mats"])
    local_rots = np.asarray(out["local_rot_mats"])
    if (posed.ndim != 4 or posed.shape[0] < 1 or posed.shape[1] < 3
            or posed.shape[2:] != (77, 3)
            or roots.shape != (posed.shape[0], posed.shape[1], 3)
            or contacts.ndim != 3 or contacts.shape[:2] != posed.shape[:2]
            or contacts.shape[2] not in (4, 6)
            or global_rots.shape != (*posed.shape[:3], 3, 3)
            or local_rots.shape != (*posed.shape[:3], 3, 3)):
        raise RuntimeError("Kimodo returned inconsistent motion array shapes")

    skel77 = model.skeleton.somaskel77
    idx = {n: i for i, n in enumerate(skel77.bone_order_names)}
    root_i = skel77.root_idx

    headings = np.asarray(out["global_root_heading"]) if "global_root_heading" in out else None
    needs_heading = any(r.get("facing") is not None for r in records)
    if needs_heading and (headings is None or headings.shape[:2] != posed.shape[:2]):
        raise RuntimeError("Kimodo output lacks global_root_heading but the move "
                           "authors heading constraints — cannot measure adherence")

    results = []
    for s in range(posed.shape[0]):
        pj, rp, fc = posed[s], roots[s], contacts[s]
        try:
            if not np.isfinite(global_rots[s]).all() or not np.isfinite(local_rots[s]).all():
                raise ValueError("rotation channels contain non-finite values")
            jc, rc, R = canonicalize(pj, rp, idx)
            g = gate_sample(jc, rc, fc, idx, mv, stance, fps)
            # constraint adherence: measured on the RAW (native-frame) sample,
            # post-processing already applied — stage "authored -> final SOMA"
            if records:
                per_rec, summary = measure_constraints(
                    records, pj, rp, global_rots[s],
                    headings[s] if headings is not None else None, idx)
                g["constraints"] = summary
                g["constraint_ok"] = summary["constraint_ok"]
                g["pass"] = bool(g["pass"] and g["constraint_ok"])
                g["score"] += (summary["ee_pos_max"] * 20.0
                               + summary["ee_rot_max_deg"] * 0.02
                               + summary["root_xz_max"] * 10.0
                               + summary["fullbody_ee_pos_max"] * 20.0
                               + summary["fullbody_ee_rot_max_deg"] * 0.02)
                g["_per_record"] = per_rec
            else:
                g["constraint_ok"] = True
        except (ValueError, IndexError) as error:
            jc = rc = R = None
            g = failed_gates(f"canonicalization failed: {error}")
            g["constraint_ok"] = False
        results.append((g, s, jc, rc, R))
    if not results:
        raise RuntimeError(f"Kimodo returned no samples for {mv['name']}")

    def clean_gates(g):
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in g.items() if k != "_per_record"}

    passing = [t for t in results if t[0]["pass"]]
    if not passing:
        g, s, _, _, _ = min(results, key=lambda t: t[0]["score"])
        report = {"name": mv["name"], "picked_sample": int(s),
                  "num_samples": len(results), "num_passing": 0,
                  "accepted": False, "loop": bool(mv.get("loop")),
                  "prompt": mv.get("prompt") or None,
                  "gates": clean_gates(g),
                  "all_gates": [clean_gates(t[0]) for t in results],
                  "constraints": ({"records": g.get("_per_record"),
                                   "gates": CONSTRAINT_GATES} if records else None),
                  "frame_data": None,
                  "frames": int(posed[s].shape[0]), "fps": int(fps)}
        with open(report_path, "w") as fp:
            json.dump(report, fp, indent=1, allow_nan=False)
        return report
    g, s, jc, rc, R = min(passing, key=lambda t: t[0]["score"])

    # loop trim on the canonical winner; required constrained frames must
    # survive the trim (best_loop refuses windows that would drop them)
    trim = None
    if mv.get("loop"):
        required = [r["frame"] for r in records if r["required"]]
        must_span = (min(required), max(required)) if required else None
        try:
            (i0, i1), loop_err = best_loop(jc, rc, int(1.0 * fps), int(3.2 * fps),
                                           must_span=must_span)
        except ValueError as error:
            raise RuntimeError(f"{mv['name']}: {error}") from None
        trim = (i0, i1)
        g["loop_err"] = round(loop_err, 4)
        g["loop_trim"] = [i0, i1]

    sl = slice(trim[0], trim[1] + 1) if trim else slice(None)
    jc_clip, rc_clip, trim_R = canonicalize(jc[sl], rc[sl], idx)
    total_R = trim_R @ R
    grm = np.einsum("ij,tnjk->tnik", total_R, global_rots[s][sl])
    lrm = local_rots[s][sl].copy()
    lrm[:, root_i] = np.einsum("ij,tjk->tik", total_R, lrm[:, root_i])

    np.savez_compressed(
        npz_path,
        posed_joints=jc_clip.astype(np.float32),
        root_positions=rc_clip.astype(np.float32),
        global_rot_mats=grm.astype(np.float32),
        local_rot_mats=lrm.astype(np.float32),
        foot_contacts=contacts[s][sl].astype(np.float32),
        canonical_rotation=total_R.astype(np.float32),
        fps=fps)

    # resolved constraint provenance, re-expressed with the SAME rigid
    # transform + trim the accepted motion got: canonical frame indices and
    # world targets, directly usable by bake/QA/runtime without repeating FK
    if records:
        p01 = roots[s][0] * np.array([1.0, 0.0, 1.0])
        recs = kc.transform_records(records, *kc.compose_canonical(R, p01))
        if trim:
            recs = kc.shift_records(mv["name"], recs, trim[0], trim[1])
            p02 = rc[sl][0] * np.array([1.0, 0.0, 1.0])
            recs = kc.transform_records(recs, *kc.compose_canonical(trim_R, p02))
        resolved = {"name": mv["name"], "space": "canonical",
                    "frames": int(jc_clip.shape[0]), "fps": int(fps),
                    "records": recs}
        with open(resolved_path, "w") as fp:
            json.dump(resolved, fp, indent=1, allow_nan=False)

    fd = frame_data(jc_clip, rc_clip, idx, mv, fps)
    report = {"name": mv["name"], "picked_sample": int(s),
              "num_samples": len(results), "num_passing": len(passing),
              "accepted": True, "loop": bool(mv.get("loop")),
              "prompt": mv.get("prompt") or None,
              "gates": clean_gates(g),
              "all_gates": [clean_gates(t[0]) for t in results],
              "constraints": ({"records": g.get("_per_record"),
                               "gates": CONSTRAINT_GATES} if records else None),
              "frame_data": fd, "frames": int(jc_clip.shape[0]), "fps": int(fps)}
    with open(report_path, "w") as fp:
        json.dump(report, fp, indent=1, allow_nan=False)
    return report


# ------------------------------------------------------------------ stance --
def extract_stance():
    """Pick the most representative frame of idle_stance as THE stance pose."""
    import torch
    from kimodo.skeleton.definitions import SOMASkeleton30, SOMASkeleton77

    with np.load(os.path.join(MOVES_OUT, "idle_stance.npz")) as z:
        required = {"posed_joints", "root_positions", "local_rot_mats", "canonical_rotation"}
        missing = required - set(z.files)
        if missing:
            raise RuntimeError("idle_stance.npz is missing " + ", ".join(sorted(missing))
                               + "; regenerate it")
        j, r, lrm = z["posed_joints"], z["root_positions"], z["local_rot_mats"]
        canonical_rotation = z["canonical_rotation"]
    if (j.ndim != 3 or j.shape[1:] != (77, 3) or len(j) < 2
            or r.shape != (len(j), 3) or lrm.shape != (len(j), 77, 3, 3)
            or canonical_rotation.shape != (3, 3)
            or not np.isfinite(j).all() or not np.isfinite(r).all()
            or not np.isfinite(lrm).all() or not np.isfinite(canonical_rotation).all()):
        raise RuntimeError("idle_stance.npz has invalid motion shapes or values; regenerate it")
    sk30, sk77 = SOMASkeleton30(), SOMASkeleton77()
    idx = {n: i for i, n in enumerate(sk77.bone_order_names)}

    # medoid frame in root-relative pose space = the pose the idle keeps returning to
    rel = j - r[:, None, :] * np.array([1.0, 0.0, 1.0])
    D = np.linalg.norm(rel[:, None] - rel[None], axis=-1).mean(-1)
    f = int(D.mean(1).argmin())

    lrm30 = sk30.from_SOMASkeleton77(torch.from_numpy(lrm[f][None]))[0].numpy()
    # Constraints are authored in Kimodo's native frame. Undo the exact
    # canonicalization used for this generated (and possibly loop-trimmed)
    # clip; a hard-coded 90° undo loses the trim frame's heading correction.
    lrm30[sk30.root_idx] = canonical_rotation.T @ lrm30[sk30.root_idx]
    from scipy.spatial.transform import Rotation as Rot
    aa = Rot.from_matrix(lrm30).as_rotvec()

    ee = [idx[n] for n in EE_NAMES]
    stance = {
        "frame": f,
        "local_joints_rot30": np.round(aa, 5).tolist(),
        "root_pos": np.round(r[f], 4).tolist(),
        "ee_root_rel": np.round(j[f, ee] - r[f] * np.array([1.0, 0.0, 1.0]), 4).tolist(),
    }
    with open(STANCE_PATH, "w") as fp:
        json.dump(stance, fp)
    print(f"[stance] frame {f} of idle_stance -> {STANCE_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "stance", "report"])
    ap.add_argument("--spec", default=os.path.join(HERE, "moveset_mk.json"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--root-margin", type=float, default=0.01,
                    help="MotionCorrection root slack in meters (0 = exact root "
                         "constraint adherence; upstream default was 0.04)")
    ap.add_argument("--model", default="Kimodo-SOMA-RP-v1.1")
    a = ap.parse_args()
    os.makedirs(MOVES_OUT, exist_ok=True)

    if a.cmd == "stance":
        return extract_stance()

    if a.cmd == "report":
        for f in sorted(os.listdir(MOVES_OUT)):
            if f.endswith(".json") and not f.endswith("constraints.json"):
                with open(os.path.join(MOVES_OUT, f)) as fp:
                    rep = json.load(fp)
                g = rep["gates"]
                print(f"{rep['name']:<14} pass={g.get('pass')} "
                      f"skate={g.get('foot_skate_mean')} jitter={g.get('jitter_mean')} "
                      f"travel_x={g.get('travel_x')} apex={g.get('apex_val', '-')} "
                      f"stance_end={g.get('stance_err_end', '-')} "
                      f"({rep['num_passing']}/{rep['num_samples']} passing)")
        return

    with open(a.spec) as fp:
        spec = json.load(fp)
    if not isinstance(spec, dict):
        ap.error("spec must be a JSON object")
    fps_spec = spec.get("fps")
    if (fps_spec is not None
            and (isinstance(fps_spec, bool) or not isinstance(fps_spec, (int, float))
                 or not np.isfinite(fps_spec) or fps_spec <= 0
                 or abs(fps_spec - round(fps_spec)) > 1e-9)):
        ap.error("spec fps must be a positive integer")
    moves = spec.get("moves")
    if not isinstance(moves, list) or not moves:
        ap.error("spec must contain a non-empty 'moves' array")
    names = [mv.get("name") for mv in moves]
    if any(not isinstance(n, str) or not SAFE_MOVE_NAME.fullmatch(n) for n in names):
        ap.error("move names must contain only letters, numbers, '_' and '-'")
    if len(set(names)) != len(names):
        ap.error("move names must be unique")
    for mv in moves:
        if (isinstance(mv.get("duration"), bool)
                or not isinstance(mv.get("duration"), (int, float))
                or not 0 < mv["duration"] <= 10):
            ap.error(f"{mv['name']}: duration must be in (0, 10] seconds")
        prompt = mv.get("prompt")
        if prompt is not None and not isinstance(prompt, str):
            ap.error(f"{mv['name']}: prompt must be a string (or omitted/null for "
                     "constraint-only generation)")
        if isinstance(prompt, str) and not prompt.strip():
            ap.error(f"{mv['name']}: an empty/whitespace prompt is ambiguous; omit "
                     "prompt or set it to null for constraint-only generation")
        if "constraints" in mv and not isinstance(mv["constraints"], list):
            ap.error(f"{mv['name']}: constraints must be a JSON array")
        if "constraints_file" in mv and (not isinstance(mv["constraints_file"], str)
                                         or not mv["constraints_file"]):
            ap.error(f"{mv['name']}: constraints_file must be a non-empty path")
        has_text = isinstance(prompt, str) and bool(prompt.strip())
        has_constraints = (mv.get("constraints") is not None
                           or mv.get("constraints_file") is not None
                           or mv.get("stance_bookend"))
        if not has_text and not has_constraints:
            ap.error(f"{mv['name']}: a move needs a text prompt, constraints, or both "
                     "(constraint-only moves omit the prompt explicitly)")
    valid_travel = {None, "in_place", "fwd", "back"}
    valid_apex = {"root_rise": "min", "root_dip": "min",
                  "root_floor": "max", "ankle_height": "min"}

    def valid_bound(v):
        return (not isinstance(v, bool) and isinstance(v, (int, float))
                and np.isfinite(v) and v >= 0)
    for mv in moves:
        if mv.get("travel") not in valid_travel:
            ap.error(f"{mv['name']}: travel must be in_place, fwd, back, or null")
        if "loop" in mv and not isinstance(mv["loop"], bool):
            ap.error(f"{mv['name']}: loop must be boolean")
        if "stance_bookend" in mv and not isinstance(mv["stance_bookend"], bool):
            ap.error(f"{mv['name']}: stance_bookend must be boolean")
        if mv.get("strike") not in (None, "hand", "foot"):
            ap.error(f"{mv['name']}: strike must be hand or foot")
        apex = mv.get("apex")
        if apex is not None:
            key = valid_apex.get(apex.get("kind")) if isinstance(apex, dict) else None
            if key is None or not valid_bound(apex.get(key)):
                ap.error(f"{mv['name']}: invalid apex definition")
            other = "max" if key == "min" else "min"
            if apex.get(other) is not None:
                if not valid_bound(apex[other]) or apex["min"] > apex["max"]:
                    ap.error(f"{mv['name']}: invalid apex bounds (need min <= max)")
        if "height" in mv and mv["height"] not in ("low", "mid", "high"):
            ap.error(f"{mv['name']}: height must be low, mid, or high")
        if mv.get("reach_policy") not in (None, "clamp"):
            ap.error(f"{mv['name']}: reach_policy must be \"clamp\" or omitted "
                     "(strict: unreachable mapped targets fail character QA)")
    if a.samples < 1 or a.steps < 1:
        ap.error("--samples and --steps must be positive")
    if not np.isfinite(a.root_margin) or a.root_margin < 0:
        ap.error("--root-margin must be a finite non-negative number")
    if not isinstance(a.model, str) or not a.model.strip():
        ap.error("--model must be non-empty")
    only = set(a.only.split(",")) if a.only else None
    unknown = only - set(names) if only else set()
    if unknown:
        ap.error("--only contains unknown moves: " + ", ".join(sorted(unknown)))
    selected = [mv for mv in moves if not only or mv["name"] in only]
    if os.path.exists(STANCE_PATH):
        with open(STANCE_PATH) as fp:
            stance = json.load(fp)
    else:
        stance = None
    if stance is not None:
        try:
            rot = np.asarray(stance["local_joints_rot30"], dtype=float)
            root = np.asarray(stance["root_pos"], dtype=float)
            ee = np.asarray(stance["ee_root_rel"], dtype=float)
        except (KeyError, TypeError, ValueError):
            ap.error("out/stance_pose.json is malformed; regenerate it with 'stance'")
        if (rot.shape != (30, 3) or root.shape != (3,) or ee.shape != (5, 3)
                or not np.isfinite(rot).all() or not np.isfinite(root).all()
                or not np.isfinite(ee).all()):
            ap.error("out/stance_pose.json is malformed; regenerate it with 'stance'")
    # The unqualified second pass means "everything else": do not overwrite
    # the idle clip from which the active stance constraint was extracted.
    # An explicit idle-only pass remains the way to replace that source.
    has_bookends = any(mv.get("stance_bookend") for mv in selected)
    if stance is not None and a.only is None and has_bookends:
        selected = [mv for mv in selected if mv["name"] != "idle_stance"]
    elif (stance is not None and has_bookends
          and any(mv["name"] == "idle_stance" for mv in selected)):
        ap.error("generate idle_stance separately before moves that use its stance")
    if stance is None and any(mv.get("stance_bookend") for mv in selected):
        ap.error("bookended moves require out/stance_pose.json; generate idle_stance, then run 'stance' first")

    # ---- prepare + validate every constraint BEFORE the model is loaded:
    # schema, frames, shapes, conflicts, bookend merge, and FK resolution of
    # authored world targets (needs only the skeleton classes, not weights)
    constrained = [mv for mv in selected
                   if mv.get("constraints") is not None
                   or mv.get("constraints_file") is not None
                   or mv.get("stance_bookend")]
    if constrained and fps_spec is None:
        ap.error("the spec must declare fps to author constraints "
                 "(frame indices depend on it): " +
                 ", ".join(mv["name"] for mv in constrained))
    prepared = {mv["name"]: {"constraints": [], "meta": [], "records": []}
                for mv in selected}
    if constrained:
        from kimodo.skeleton.definitions import SOMASkeleton30
        skel30 = SOMASkeleton30()
        spec_dir = os.path.dirname(os.path.abspath(a.spec))
        for mv in constrained:
            nF = int(mv["duration"] * fps_spec)
            try:
                cons, meta = kc.prepare_move_constraints(mv, nF, fps_spec, stance, spec_dir)
                records = kc.resolve_records(mv["name"], cons, meta, skel30) if cons else []
            except kc.ConstraintSpecError as e:
                ap.error(str(e))
            prepared[mv["name"]] = {"constraints": cons, "meta": meta, "records": records}

    model = load_kimodo(a.model)
    if "fps" in spec and int(spec["fps"]) != int(model.fps):
        ap.error(f"spec fps={spec['fps']} does not match model fps={model.fps}")
    too_short = [mv["name"] for mv in selected if int(mv["duration"] * model.fps) < 3]
    if too_short:
        ap.error("moves must produce at least 3 frames: " + ", ".join(too_short))
    failed = []
    for mv in selected:
        if mv["name"] == "idle_stance" and os.path.exists(STANCE_PATH):
            os.remove(STANCE_PATH)
            stance = None
            print("[stance] removed stale stance_pose.json; run 'stance' after idle generation")
        label = (mv.get("prompt") or "<constraint-only>")[:60]
        ncons = len(prepared[mv["name"]]["constraints"])
        print(f"[gen] {mv['name']}: '{label}...' "
              f"({mv['duration']}s x{a.samples}, {ncons} constraints)")
        rep = gen_move(model, mv, stance, prepared[mv["name"]], a.samples, a.seed, a.steps,
                       root_margin=a.root_margin)
        g = rep["gates"]
        print(f"      -> accepted={rep['accepted']} ({rep['num_passing']}/{rep['num_samples']} passing) "
              f"skate={g['foot_skate_mean']} travel_x={g['travel_x']}")
        if not rep["accepted"]:
            failed.append(mv["name"])
    if failed:
        print("[reject] no passing sample for: " + ", ".join(failed), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
