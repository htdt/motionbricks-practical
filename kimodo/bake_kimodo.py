"""bake_kimodo — export Kimodo move NPZs as browser motion JSONs + manifest.

The Stage 2 bake step (BAKE.md §5). Clips are already canonicalized by
kimogen (frame-0 root at origin facing +X) and already Y-up (Kimodo ==
three.js axes, no basis change). Each JSON carries `srcMap` (SOMA-77 joint
names → canonical source roles) so align/retarget.js drives any certified
rig from it without knowing the source family.

Besides the motion channels, each baked clip preserves:
- `contacts` + `contactJoints`: Kimodo's per-frame foot-contact PREDICTIONS
  with their explicit joint mapping (QA/cleanup evidence, never authored
  targets);
- `constraints`: the resolved, canonical-frame constraint records written by
  kimogen (<move>.resolved_constraints.json) — authored world targets with
  family/source/provenance, directly usable by runtime IK and constraint QA
  without reloading the authoring skeleton or repeating FK.

Usage: kimenv/bin/python bake_kimodo.py [--in out/moves] [--web ../web/moves_kimodo]
No GPU needed.
"""
import argparse
import json
import os
import re

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
SAFE_MOVE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

# canonical source roles -> somaskel77 joint names (mirror of SOMA_SRC in
# align/retarget.js; baked into each clip JSON as data.srcMap)
SOMA_SRC = {
    "Hips": "Hips", "Chest": "Chest",
    "LeftHipAnchor": "LeftLeg", "RightHipAnchor": "RightLeg",
    "LeftShoulderAnchor": "LeftArm", "RightShoulderAnchor": "RightArm",
    "LeftUpLeg": "LeftLeg", "LeftLeg": "LeftShin", "LeftFoot": "LeftFoot",
    "RightUpLeg": "RightLeg", "RightLeg": "RightShin", "RightFoot": "RightFoot",
    "LeftArm": "LeftArm", "LeftForeArm": "LeftForeArm", "LeftHand": "LeftHand",
    "RightArm": "RightArm", "RightForeArm": "RightForeArm", "RightHand": "RightHand",
}

# Kimodo foot-contact channel layouts (verified against the installed
# package: 4-channel = SOMA-30 [L heel, L toe, R heel, R toe]; the 6-channel
# layout is the SOMA-77 expansion, toe-end copying toe-base). Contacts are
# model PREDICTIONS about the generated motion, not authored constraints.
CONTACT_JOINTS_BY_WIDTH = {
    4: ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"],
    6: ["LeftFoot", "LeftToeBase", "LeftToeEnd",
        "RightFoot", "RightToeBase", "RightToeEnd"],
}


def minrot(a, b):
    """Minimal rotation matrix taking unit vector a to unit vector b."""
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    if not np.isfinite(an) or not np.isfinite(bn) or an < 1e-9 or bn < 1e-9:
        raise ValueError("cannot align a zero or non-finite vector")
    a, b = a / an, b / bn
    v, c = np.cross(a, b), float(np.clip(a @ b, -1.0, 1.0))
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        basis = np.eye(3)[np.argmin(np.abs(a))]
        axis = np.cross(a, basis)
        axis /= np.linalg.norm(axis)
        return Rot.from_rotvec(axis * np.pi).as_matrix()
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def mats_to_xyzw(m):
    """[...,3,3] rotation matrices -> [...,4] xyzw quats (three.js order)."""
    shape = m.shape[:-2]
    q = Rot.from_matrix(m.reshape(-1, 3, 3)).as_quat()   # xyzw
    return q.reshape(*shape, 4)


def manifest_entry(name, frames, fps, meta, spec_move, constraints=None):
    """Build the runtime contract for one already-validated clip."""
    loop = spec_move.get("loop", meta.get("loop", False))
    entry = {
        "name": name,
        "file": f"{name}.json",
        "frames": int(frames),
        "fps": int(fps),
        "loop": bool(loop),
        "frame_data": meta.get("frame_data"),
        "gates": {k: v for k, v in meta.get("gates", {}).items()
                  if not isinstance(v, (list, dict))},
    }
    if constraints:
        counts = {}
        for r in constraints:
            counts[r["type"]] = counts.get(r["type"], 0) + 1
        entry["constraints"] = {
            "counts": counts,
            "adherence": meta.get("gates", {}).get("constraints"),
        }
        # documented per-move reach policy: "clamp" lets character QA accept
        # explicitly clamped (reported) unreachable targets; default strict
        if spec_move.get("reach_policy") == "clamp":
            entry["reachPolicy"] = "clamp"
    return entry


def has_authored_hand_rotation(constraints):
    """Whether baked stylization must yield to an exact authored wrist.

    End-effector records name one role directly; fullbody records carry their
    constrained roles in the resolved `ee` map.
    """
    for rec in constraints or []:
        if rec.get("rotConstrained") is False:
            continue
        if rec.get("role") in ("LeftHand", "RightHand"):
            return True
        if rec.get("family") == "fullbody" and any(
                role in rec.get("ee", {}) for role in ("LeftHand", "RightHand")):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", "--in", default=os.path.join(HERE, "out/moves"))
    ap.add_argument("--web-dir", "--web", default=os.path.join(HERE, "../web/moves_kimodo"))
    ap.add_argument("--spec", default=os.path.join(HERE, "moveset_mk.json"),
                    help="move spec; defines output order and loop flags")
    ap.add_argument("--all", action="store_true",
                    help="bake every NPZ in in-dir instead of the spec's move list")
    ap.add_argument("--allow-missing", action="store_true",
                    help="allow a partial spec bake (normally every spec move is required)")
    ap.add_argument("--allow-ungated", action="store_true",
                    help="allow NPZs without a generation report (for imported examples only)")
    a = ap.parse_args()
    os.makedirs(a.web_dir, exist_ok=True)

    with open(a.spec) as fp:
        spec = json.load(fp)
    if not isinstance(spec, dict):
        ap.error("spec must be a JSON object")
    spec_moves = spec.get("moves", [])
    if not isinstance(spec_moves, list):
        ap.error("spec 'moves' must be an array")
    if not a.all and not spec_moves:
        ap.error("spec 'moves' must not be empty")
    if any(not isinstance(mv, dict) for mv in spec_moves):
        ap.error("every spec move must be an object")
    spec_by_name = {mv.get("name"): mv for mv in spec_moves}
    if (any(not isinstance(name, str) or not SAFE_MOVE_NAME.fullmatch(name)
            for name in spec_by_name)
            or len(spec_by_name) != len(spec_moves)):
        ap.error("spec move names must be safe and unique")
    if any("loop" in mv and not isinstance(mv["loop"], bool) for mv in spec_moves):
        ap.error("spec loop flags must be boolean")
    spec_fps = spec.get("fps")
    if (spec_fps is not None
            and (isinstance(spec_fps, bool) or not isinstance(spec_fps, (int, float))
                 or not np.isfinite(spec_fps) or spec_fps <= 0
                 or abs(spec_fps - round(spec_fps)) > 1e-9)):
        ap.error("spec fps must be a positive integer")

    from kimodo.skeleton.definitions import SOMASkeleton30, SOMASkeleton77

    def skel_pack(J):
        """names/parents/rest/restQuat for a 77- or 30-joint SOMA export.

        restQuat convention: in Kimodo NPZ outputs the T-pose is the zero
        pose — global_rot_mats of EVERY joint is the IDENTITY at rest (FK of
        identity locals). So the baked rest rotations are the identity
        (yawed to the canonical +X facing), NOT the
        `standard_t_pose_global_offsets_rots` asset — that file is a
        BVH-export convention; using it shipped garbage orientation anchors
        (skewed fists/feet, caught by align/qa_endeffectors.mjs).

        FEET keep the identity anchor as-is: at a generated standing frame
        the source foot global rotation returns to ~identity (measured:
        0.1°) — the T-pose toe droop (~14°) is anatomy shared by every
        frame, and "leveling" it injects exactly that skew. Verified
        empirically — do not re-add foot leveling.

        HANDS get one semantic correction on top: the SOMA T-pose hand bends
        ~18° off the forearm axis, while game rigs bind hands straight along
        the forearm. The retargeter transfers wrist DELTAS from rest — with
        a bent rest, a straight mocap wrist would hyperextend the character's
        straight-bind hand by that 18°. Pre-rotating the hand rest onto the
        forearm axis makes the anchor "straight source wrist ↔ straight
        character hand": absolute bend tracking. (Only valid together with
        retarget.js's world-axes wrist mapping — under the old bone-local
        splice this same correction rotated about wrong axes.)
        """
        sk77 = SOMASkeleton77()
        sk = sk77 if J == 77 else SOMASkeleton30()
        names = sk.bone_order_names
        parents = [-1 if p is None else names.index(p)
                   for _, p in sk.bone_order_names_with_parents]
        nj = sk.neutral_joints.numpy()
        ni = {n: i for i, n in enumerate(names)}

        rots = np.tile(np.eye(3), (len(names), 1, 1))
        for side in ("Left", "Right"):
            mid = f"{side}HandMiddle1" if f"{side}HandMiddle1" in ni else \
                  (f"{side}HandMiddleEnd" if f"{side}HandMiddleEnd" in ni else None)
            if mid:
                hd = nj[ni[mid]] - nj[ni[f"{side}Hand"]]
                fd = nj[ni[f"{side}Hand"]] - nj[ni[f"{side}ForeArm"]]
                rots[ni[f"{side}Hand"]] = minrot(hd, fd)

        # rest pose: standard T-pose, lifted so the lowest foot joint sits on
        # the ground (the retargeter reads rest hip/ankle heights from this),
        # and yawed -90° so it faces +X like every canonicalized clip (the
        # SOMA T-pose faces +Z; the retargeter yaw-rebases from the source
        # REST heading, so rest and clips must agree)
        c, s = 0.0, 1.0                                   # rot_y(pi/2): +Z -> +X
        Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        rest = nj @ Ry.T
        foot_i = [names.index(n) for n in
                  ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]]
        rest[:, 1] -= rest[foot_i, 1].min()
        rest_quat = mats_to_xyzw(np.einsum("ij,njk->nik", Ry, rots))
        return names, parents, rest, rest_quat

    packs = {}

    order = ([mv["name"] for mv in spec_moves] if not a.all else sorted(
        f[:-4] for f in os.listdir(a.in_dir) if f.endswith(".npz")))
    if any(not SAFE_MOVE_NAME.fullmatch(name) for name in order):
        raise SystemExit("[reject] input NPZ names must contain only letters, numbers, '_' and '-'")
    manifest = []
    for name in order:
        src = os.path.join(a.in_dir, f"{name}.npz")
        meta_path = os.path.join(a.in_dir, f"{name}.json")
        if not os.path.exists(src):
            rejected = False
            if os.path.exists(meta_path):
                with open(meta_path) as fp:
                    failed_meta = json.load(fp)
                rejected = failed_meta.get(
                    "accepted", failed_meta.get("gates", {}).get("pass")) is not True
            if a.allow_missing:
                # a deliberate partial bake may skip explicitly rejected moves
                # (e.g. an expected-fail case in a test spec) — loudly
                print(f"[skip] {name} ({'generation rejected' if rejected else 'no npz'})")
                continue
            if rejected:
                raise SystemExit(f"[reject] {name}: generation gates did not pass")
            if a.all:
                print(f"[skip] {name} (no npz)")
                continue
            raise SystemExit(f"[reject] {name}: expected NPZ is missing (use --allow-missing for a partial bake)")
        if os.path.exists(meta_path):
            with open(meta_path) as fp:
                meta = json.load(fp)
        else:
            meta = {}
        if not isinstance(meta, dict) or not isinstance(meta.get("gates", {}), dict):
            raise SystemExit(f"[reject] {name}: generation report must be a JSON object")
        if meta and meta.get("name", name) != name:
            raise SystemExit(f"[reject] {name}: generation report belongs to {meta.get('name')}")
        reported = meta.get("accepted")
        gated = meta.get("gates", {}).get("pass")
        if reported is not None and not isinstance(reported, bool):
            raise SystemExit(f"[reject] {name}: generation report has a malformed accepted flag")
        if gated is not None and not isinstance(gated, bool):
            raise SystemExit(f"[reject] {name}: generation report has a malformed pass gate")
        if reported is not None and gated is not None and reported != gated:
            raise SystemExit(f"[reject] {name}: generation report acceptance flags disagree")
        accepted = reported if reported is not None else gated
        if accepted is False:
            raise SystemExit(f"[reject] {name}: generation gates did not pass")
        if not a.allow_ungated and accepted is not True:
            reason = "missing generation report" if not meta else "generation gates did not pass"
            raise SystemExit(f"[reject] {name}: {reason}; use --allow-ungated only for imported examples")
        if "loop" in meta and not isinstance(meta["loop"], bool):
            raise SystemExit(f"[reject] {name}: report loop flag must be boolean")

        z = np.load(src)
        required = {"posed_joints", "global_rot_mats", "fps"}
        missing = required - set(z.files)
        if missing:
            raise SystemExit(f"[reject] {name}: NPZ missing {', '.join(sorted(missing))}")
        pj, grm = z["posed_joints"], z["global_rot_mats"]
        if pj.ndim != 3 or pj.shape[-1] != 3:
            raise SystemExit(f"[reject] {name}: posed_joints must have shape [T,J,3]")
        F, J = pj.shape[0], pj.shape[1]
        if J not in (30, 77):
            raise SystemExit(f"[reject] {name}: expected a 30- or 77-joint SOMA clip, got {J}")
        if F < 2 or grm.shape[:2] != (F, J) or grm.shape[-2:] != (3, 3):
            raise SystemExit(f"[reject] {name}: inconsistent motion shapes")
        if not np.isfinite(pj).all() or not np.isfinite(grm).all():
            raise SystemExit(f"[reject] {name}: motion contains non-finite values")
        fps_value = np.asarray(z["fps"])
        if fps_value.size != 1:
            raise SystemExit(f"[reject] {name}: fps must be a scalar")
        fps_float = float(fps_value.reshape(-1)[0])
        if (not np.isfinite(fps_float) or fps_float <= 0
                or abs(fps_float - round(fps_float)) > 1e-9):
            raise SystemExit(f"[reject] {name}: fps must be a positive integer")
        fps = int(round(fps_float))
        if spec_fps is not None and fps != int(round(spec_fps)):
            raise SystemExit(f"[reject] {name}: clip fps={fps} does not match spec fps={spec_fps}")
        eye = np.eye(3)
        orth_err = float(np.max(np.abs(np.swapaxes(grm, -1, -2) @ grm - eye)))
        det_err = float(np.max(np.abs(np.linalg.det(grm) - 1.0)))
        if orth_err > 5e-3 or det_err > 5e-3:
            raise SystemExit(
                f"[reject] {name}: invalid rotation matrices "
                f"(orthogonality error={orth_err:.4g}, determinant error={det_err:.4g})")
        if ("frames" in meta and (isinstance(meta["frames"], bool)
                                  or not isinstance(meta["frames"], int))):
            raise SystemExit(f"[reject] {name}: report frames must be an integer")
        if "frames" in meta and meta["frames"] != F:
            raise SystemExit(f"[reject] {name}: report frames={meta['frames']} but NPZ has {F}")
        if ("fps" in meta and (isinstance(meta["fps"], bool)
                               or not isinstance(meta["fps"], int))):
            raise SystemExit(f"[reject] {name}: report fps must be an integer")
        if "fps" in meta and meta["fps"] != fps:
            raise SystemExit(f"[reject] {name}: report fps={meta['fps']} but NPZ has {fps}")
        spec_move = spec_by_name.get(name, {})
        if ("loop" in meta and "loop" in spec_move
                and meta["loop"] != spec_move["loop"]):
            raise SystemExit(f"[reject] {name}: report/spec loop flags disagree; regenerate the clip")
        contacts = None
        if "foot_contacts" in z.files:
            contacts = np.asarray(z["foot_contacts"], dtype=float)
            if (contacts.ndim != 2 or contacts.shape[0] != F
                    or contacts.shape[1] not in CONTACT_JOINTS_BY_WIDTH
                    or not np.isfinite(contacts).all()):
                raise SystemExit(f"[reject] {name}: foot_contacts must be a finite "
                                 f"[{F},4|6] array; got {contacts.shape}")
        z.close()

        # resolved constraint provenance (canonical frame, written by kimogen)
        resolved_path = os.path.join(a.in_dir, f"{name}.resolved_constraints.json")
        constraints = None
        if os.path.exists(resolved_path):
            with open(resolved_path) as fp:
                resolved = json.load(fp)
            if (not isinstance(resolved, dict) or resolved.get("name") != name
                    or resolved.get("space") != "canonical"
                    or not isinstance(resolved.get("records"), list)):
                raise SystemExit(f"[reject] {name}: malformed resolved_constraints file")
            if resolved.get("frames") != F or resolved.get("fps") != fps:
                raise SystemExit(
                    f"[reject] {name}: resolved constraints were written for "
                    f"{resolved.get('frames')} frames @ {resolved.get('fps')} fps but the NPZ "
                    f"has {F} @ {fps} — regenerate the move")
            if any(not isinstance(r, dict) or not isinstance(r.get("frame"), int)
                   or not 0 <= r["frame"] < F for r in resolved["records"]):
                raise SystemExit(f"[reject] {name}: resolved constraint records must "
                                 "carry in-range integer canonical frames")
            constraints = resolved["records"]
        elif meta.get("constraints"):
            raise SystemExit(f"[reject] {name}: the generation report shows constraints "
                             "but resolved_constraints.json is missing; regenerate the move")
        if J not in packs:
            packs[J] = skel_pack(J)
        names, parents, rest, rest_quat = packs[J]
        ni = {n: i for i, n in enumerate(names)}
        root_xz = pj[0, ni["Hips"], [0, 2]]
        hip_right = pj[0, ni["RightLeg"]] - pj[0, ni["LeftLeg"]]
        facing = np.cross(np.array([0.0, 1.0, 0.0]), hip_right)
        facing[1] = 0.0
        facing /= np.linalg.norm(facing) + 1e-12
        if np.linalg.norm(root_xz) > 1e-3 or facing[0] < 0.999:
            raise SystemExit(
                f"[reject] {name}: clip is not canonical (frame-0 root XZ={root_xz.tolist()}, "
                f"facing={facing.tolist()}); regenerate it with the current kimogen.py")
        quat = mats_to_xyzw(grm)

        # a clip with authored hand-rotation constraints must transfer the
        # full authored wrist: a global stylization gain would silently damp
        # an exact target (handFollow stays a stylization knob for
        # prediction-only clips; see the ablation evidence in KIMODO.md)
        hand_constrained = has_authored_hand_rotation(constraints)
        out = {
            "fps": fps, "numFrames": F, "mode": name,
            "names": names, "parents": parents,
            "pos": np.round(pj, 4).tolist(),
            "rest": np.round(rest, 4).tolist(),
            "quat": np.round(quat, 5).tolist(),
            "restQuat": np.round(rest_quat, 5).tolist(),
            "srcMap": SOMA_SRC,
            "source": "kimodo-soma-rp-v1.1",
            # wrist articulation gain for the retargeter: mocap wrist
            # channels read as "broken fists" on fingerless fist meshes —
            # keep 30% of the source wrist, ride the forearm for the rest
            "handFollow": 1.0 if hand_constrained else 0.3,
        }
        if contacts is not None:
            out["contacts"] = np.round(contacts, 3).tolist()
            out["contactJoints"] = CONTACT_JOINTS_BY_WIDTH[contacts.shape[1]]
        if constraints is not None:
            out["constraints"] = constraints
        with open(os.path.join(a.web_dir, f"{name}.json"), "w") as fp:
            json.dump(out, fp, allow_nan=False)

        manifest.append(manifest_entry(
            name, F, fps, meta, spec_move, constraints))
        print(f"[bake] {name}: {F} frames"
              + (f", {len(constraints)} constraint records" if constraints else ""))

    if not manifest:
        raise SystemExit("[reject] no clips were found to bake")
    with open(os.path.join(a.web_dir, "manifest.json"), "w") as fp:
        json.dump({"moves": manifest, "source": "kimodo"}, fp, indent=1,
                  allow_nan=False)
    print(f"[bake] manifest: {len(manifest)} moves -> {a.web_dir}/manifest.json")


if __name__ == "__main__":
    main()
