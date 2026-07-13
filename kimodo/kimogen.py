"""kimogen — MK move-set generation on NVIDIA Kimodo (SOMA skeleton).

The Stage 2 generation step (BAKE.md §3-4): each move is
text-prompted (combat is in Kimodo's training distribution), optionally
bookended with a fighting-stance fullbody keyframe constraint at both ends
(so clips chain/crossfade in-game), generated as a best-of-N batch, and
pushed through numeric QA gates. Winner NPZ + gate/frame-data JSON per move.

Usage (venv: kimenv; start `TEXT_ENCODER_DEVICE=cpu kimodo_textencoder` first
or let it fall back to an in-process CPU encoder):

  python kimogen.py gen   --spec moveset_mk.json [--only jab,sweep] [--samples 8] [--seed 42]
  python kimogen.py stance                    # extract stance pose from out/moves/idle_stance.npz
  python kimogen.py report                    # gate table of everything generated so far

Axis conventions (Kimodo): Y-up, XZ ground, heading angle t about Y with
facing dir (sin t, cos t) — i.e. heading 0 faces +Z. Verified by
validate_axes.py on a generated walk. Gates evaluate in a canonical frame
(frame-0 root at origin, frame-0 facing rotated to +X = the baked-clip/game
convention).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
MOVES_OUT = os.path.join(OUT, "moves")
STANCE_PATH = os.path.join(OUT, "stance_pose.json")

# somaskel77 indices are resolved at runtime from the skeleton class
EE_NAMES = ["LeftHand", "RightHand", "LeftFoot", "RightFoot", "Head"]
CONTACT_JOINTS = ["LeftFoot", "LeftToeBase", "LeftToeEnd",
                  "RightFoot", "RightToeBase", "RightToeEnd"]


# ---------------------------------------------------------------- geometry --
def canonicalize(joints, root_pos, idx):
    """Rotate/translate so frame-0 root sits at XZ origin facing +X.

    Frame-0 facing is derived from the hip line: fwd = cross(up, rHip-lHip)
    (verified on Kimodo outputs by validate_axes.py — walk-forward clips
    travel along this vector). Returns (joints', root', R).
    """
    right = joints[0, idx["RightLeg"]] - joints[0, idx["LeftLeg"]]
    fwd = np.cross(np.array([0.0, 1.0, 0.0]), right)
    fwd[1] = 0.0
    fwd /= np.linalg.norm(fwd) + 1e-9
    c, s = fwd[0], fwd[2]                     # rot_y mapping fwd -> +X
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    p0 = root_pos[0] * np.array([1.0, 0.0, 1.0])
    j = (joints - p0) @ R.T
    r = (root_pos - p0) @ R.T
    return j, r, R


# ------------------------------------------------------------------- gates --
def gate_sample(j, r, fc, idx, mv, stance, fps):
    """All-numeric QA for one canonicalized sample. Returns dict of metrics."""
    g = {}
    g["nan"] = bool(np.isnan(j).any() or np.isnan(r).any())

    # foot skate: horizontal speed of contact-labeled foot joints (m/s)
    cj = [idx[n] for n in CONTACT_JOINTS]
    v = np.linalg.norm(np.diff(j[:, cj][:, :, [0, 2]], axis=0), axis=-1) * fps  # [T-1, 6]
    c = (fc[1:] > 0.5) & (fc[:-1] > 0.5)
    g["foot_skate_mean"] = float(v[c].mean()) if c.any() else 0.0
    g["contact_frac"] = float((fc > 0.5).any(axis=-1).mean())

    # jitter: mean 2nd difference of joint positions (m/frame^2)
    g["jitter_mean"] = float(np.linalg.norm(np.diff(j, n=2, axis=0), axis=-1).mean())

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
            g["apex_ok"] = g["apex_val"] >= apex["min"]
        elif apex["kind"] == "root_dip":
            g["apex_val"] = float(y0 - r[:, 1].min())
            g["apex_ok"] = g["apex_val"] >= apex["min"]
        elif apex["kind"] == "root_floor":
            g["apex_val"] = float(r[:, 1].min())
            g["apex_ok"] = g["apex_val"] <= apex["max"]
        elif apex["kind"] == "ankle_height":
            ank = max(float(j[:, idx["LeftFoot"], 1].max()),
                      float(j[:, idx["RightFoot"], 1].max()))
            g["apex_val"] = ank
            g["apex_ok"] = ank >= apex["min"]
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
        g["stance_ok"] = g["stance_err_end"] < 0.22
    else:
        g["stance_ok"] = True

    g["pass"] = (not g["nan"]) and g["travel_ok"] and g["apex_ok"] and g["stance_ok"]
    # score: lower better; only meaningful among passing samples
    g["score"] = (g["foot_skate_mean"] * 2.0 + g["jitter_mean"] * 30.0
                  + g.get("stance_err_end", 0.0) * 3.0)
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


def best_loop(j, r, min_len, max_len):
    """Find (i0, i1) trimming to the best pose-space cycle for loop moves."""
    T = len(j)
    root_rel = j - r[:, None, :] * np.array([1.0, 0.0, 1.0])
    best, pair = 1e9, (0, T - 1)
    for i in range(0, T - min_len):
        jmax = min(T, i + max_len)
        for k in range(i + min_len, jmax):
            d = float(np.linalg.norm(root_rel[i] - root_rel[k], axis=-1).mean())
            if d < best:
                best, pair = d, (i, k)
    return pair, best


# --------------------------------------------------------------- generation --
def load_kimodo(modelname):
    import torch
    from kimodo import load_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(modelname, device=device, default_family="Kimodo")
    return model, device


def build_bookend_constraints(stance, nF):
    """Fullbody stance keyframes at the first and last frames (somaskel30)."""
    rot = stance["local_joints_rot30"]
    rp = stance["root_pos"]
    return [{
        "type": "fullbody",
        "frame_indices": [0, nF - 1],
        "local_joints_rot": [rot, rot],
        "root_positions": [[0.0, rp[1], 0.0], [0.0, rp[1], 0.0]],
    }]


def gen_move(model, device, mv, stance, samples, seed, steps):
    import torch
    from kimodo.constraints import load_constraints_lst
    from kimodo.tools import seed_everything

    fps = model.fps
    nF = int(mv["duration"] * fps)
    constraint_lst = []
    cfg_kwargs = {}
    if stance is not None and mv.get("stance_bookend"):
        cpath = os.path.join(MOVES_OUT, f"{mv['name']}.constraints.json")
        with open(cpath, "w") as fp:
            json.dump(build_bookend_constraints(stance, nF), fp)
        constraint_lst = load_constraints_lst(cpath, model.skeleton)
        cfg_kwargs = {"cfg_type": "separated", "cfg_weight": [2.0, 2.0]}

    seed_everything(seed)
    out = model([mv["prompt"]], [nF],
                constraint_lst=constraint_lst,
                num_denoising_steps=steps,
                num_samples=samples,
                multi_prompt=True,
                post_processing=True,
                return_numpy=True,
                **cfg_kwargs)

    skel77 = model.skeleton.somaskel77
    idx = {n: i for i, n in enumerate(skel77.bone_order_names)}
    root_i = skel77.root_idx

    results = []
    for s in range(out["posed_joints"].shape[0]):
        pj = out["posed_joints"][s]
        rp = out["root_positions"][s] if out["root_positions"].ndim == 3 else pj[:, root_i]
        fc = out["foot_contacts"][s]
        jc, rc, R = canonicalize(pj, rp, idx)
        g = gate_sample(jc, rc, fc, idx, mv, stance, fps)
        results.append((g, s, jc, rc, R))

    passing = [t for t in results if t[0]["pass"]]
    pool = passing if passing else results
    g, s, jc, rc, R = min(pool, key=lambda t: t[0]["score"])

    # loop trim on the canonical winner
    trim = None
    if mv.get("loop"):
        (i0, i1), loop_err = best_loop(jc, rc, int(1.0 * fps), int(3.2 * fps))
        trim = (i0, i1)
        g["loop_err"] = round(loop_err, 4)
        g["loop_trim"] = [i0, i1]

    sl = slice(trim[0], trim[1] + 1) if trim else slice(None)
    grm = out["global_rot_mats"][s][sl]
    # rotate global rotations into the canonical frame
    grm = np.einsum("ij,tnjk->tnik", R, grm)
    lrm = out["local_rot_mats"][s][sl]
    lrm0 = np.einsum("ij,tjk->tik", R, lrm[:, root_i])       # root local = global yaw
    lrm = lrm.copy(); lrm[:, root_i] = lrm0

    np.savez_compressed(
        os.path.join(MOVES_OUT, f"{mv['name']}.npz"),
        posed_joints=jc[sl].astype(np.float32),
        root_positions=rc[sl].astype(np.float32),
        global_rot_mats=grm.astype(np.float32),
        local_rot_mats=lrm.astype(np.float32),
        foot_contacts=out["foot_contacts"][s][sl].astype(np.float32),
        fps=fps)

    fd = frame_data(jc[sl], rc[sl], idx, mv, fps)
    report = {"name": mv["name"], "picked_sample": int(s),
              "num_samples": len(results), "num_passing": len(passing),
              "gates": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in g.items()},
              "all_gates": [{k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in t[0].items()} for t in results],
              "frame_data": fd, "frames": int(jc[sl].shape[0]), "fps": int(fps)}
    with open(os.path.join(MOVES_OUT, f"{mv['name']}.json"), "w") as fp:
        json.dump(report, fp, indent=1)
    return report


# ------------------------------------------------------------------ stance --
def extract_stance():
    """Pick the most representative frame of idle_stance as THE stance pose."""
    import torch
    from kimodo.skeleton.definitions import SOMASkeleton30, SOMASkeleton77

    z = np.load(os.path.join(MOVES_OUT, "idle_stance.npz"))
    j, r, lrm = z["posed_joints"], z["root_positions"], z["local_rot_mats"]
    sk30, sk77 = SOMASkeleton30(), SOMASkeleton77()
    idx = {n: i for i, n in enumerate(sk77.bone_order_names)}

    # medoid frame in root-relative pose space = the pose the idle keeps returning to
    rel = j - r[:, None, :] * np.array([1.0, 0.0, 1.0])
    D = np.linalg.norm(rel[:, None] - rel[None], axis=-1).mean(-1)
    f = int(D.mean(1).argmin())

    lrm30 = sk30.from_SOMASkeleton77(torch.from_numpy(lrm[f][None]))[0].numpy()
    # constraints are authored in Kimodo's native frame (heading 0 = facing
    # +Z); the saved NPZ is canonicalized to face +X — de-rotate the root row
    Rym = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # rot_y(-pi/2)
    lrm30[sk30.root_idx] = Rym @ lrm30[sk30.root_idx]
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
    ap.add_argument("--model", default="Kimodo-SOMA-RP-v1.1")
    a = ap.parse_args()
    os.makedirs(MOVES_OUT, exist_ok=True)

    if a.cmd == "stance":
        return extract_stance()

    if a.cmd == "report":
        for f in sorted(os.listdir(MOVES_OUT)):
            if f.endswith(".json") and not f.endswith("constraints.json"):
                rep = json.load(open(os.path.join(MOVES_OUT, f)))
                g = rep["gates"]
                print(f"{rep['name']:<14} pass={g.get('pass')} "
                      f"skate={g.get('foot_skate_mean')} jitter={g.get('jitter_mean')} "
                      f"travel_x={g.get('travel_x')} apex={g.get('apex_val', '-')} "
                      f"stance_end={g.get('stance_err_end', '-')} "
                      f"({rep['num_passing']}/{rep['num_samples']} passing)")
        return

    spec = json.load(open(a.spec))
    only = set(a.only.split(",")) if a.only else None
    stance = json.load(open(STANCE_PATH)) if os.path.exists(STANCE_PATH) else None
    if stance is None:
        print("[warn] no stance_pose.json yet — bookend constraints disabled this run")

    model, device = load_kimodo(a.model)
    for mv in spec["moves"]:
        if only and mv["name"] not in only:
            continue
        print(f"[gen] {mv['name']}: '{mv['prompt'][:60]}...' "
              f"({mv['duration']}s x{a.samples})")
        rep = gen_move(model, device, mv, stance, a.samples, a.seed, a.steps)
        g = rep["gates"]
        print(f"      -> pass={g['pass']} ({rep['num_passing']}/{rep['num_samples']} passing) "
              f"skate={g['foot_skate_mean']} travel_x={g['travel_x']}")


if __name__ == "__main__":
    main()
