"""bake_kimodo — export Kimodo move NPZs as browser motion JSONs + manifest.

The Stage 2 bake step (BAKE.md §5). Clips are already canonicalized by
kimogen (frame-0 root at origin facing +X) and already Y-up (Kimodo ==
three.js axes, no basis change). Each JSON carries `srcMap` (SOMA-77 joint
names → canonical source roles) so align/retarget.js drives any certified
rig from it without knowing the source family.

Usage: kimenv/bin/python bake_kimodo.py [--in out/moves] [--web ../web/moves_kimodo]
No GPU needed.
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))

ORDER = ["idle_stance", "walk_fwd", "walk_back", "jump_up", "crouch",
         "block_high", "jab", "punch_heavy", "uppercut", "kick_front",
         "kick_high", "kick_side", "sweep", "hit_head", "hit_heavy",
         "knockdown", "victory"]

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


def minrot(a, b):
    """Minimal rotation matrix taking unit vector a to unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c > 1 - 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def mats_to_xyzw(m):
    """[...,3,3] rotation matrices -> [...,4] xyzw quats (three.js order)."""
    shape = m.shape[:-2]
    q = Rot.from_matrix(m.reshape(-1, 3, 3)).as_quat()   # xyzw
    return q.reshape(*shape, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=os.path.join(HERE, "out/moves"))
    ap.add_argument("--web-dir", default=os.path.join(HERE, "../web/moves_kimodo"))
    ap.add_argument("--all", action="store_true",
                    help="bake every npz in in-dir instead of the MK ORDER list")
    a = ap.parse_args()
    os.makedirs(a.web_dir, exist_ok=True)

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

    order = ORDER if not a.all else sorted(
        f[:-4] for f in os.listdir(a.in_dir) if f.endswith(".npz"))
    manifest = []
    for name in order:
        src = os.path.join(a.in_dir, f"{name}.npz")
        if not os.path.exists(src):
            print(f"[skip] {name} (no npz)")
            continue
        z = np.load(src)
        pj, grm = z["posed_joints"], z["global_rot_mats"]
        F, J = pj.shape[0], pj.shape[1]
        if J not in packs:
            packs[J] = skel_pack(J)
        names, parents, rest, rest_quat = packs[J]
        quat = mats_to_xyzw(grm)

        out = {
            "fps": int(z["fps"]), "numFrames": F, "mode": name,
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
            "handFollow": 0.3,
            # forearm roll from the source's own twist (true pronation)
            # instead of the chest-rebase projection, which spins the fist
            # during big torso pitches (jump crouch)
            "foreRollSrc": True,
        }
        with open(os.path.join(a.web_dir, f"{name}.json"), "w") as fp:
            json.dump(out, fp)

        meta_path = os.path.join(a.in_dir, f"{name}.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        manifest.append({"name": name, "file": f"{name}.json", "frames": F,
                         "frame_data": meta.get("frame_data"),
                         "gates": {k: v for k, v in meta.get("gates", {}).items()
                                   if not isinstance(v, list)}})
        print(f"[bake] {name}: {F} frames")

    with open(os.path.join(a.web_dir, "manifest.json"), "w") as fp:
        json.dump({"moves": manifest, "source": "kimodo"}, fp, indent=1)
    print(f"[bake] manifest: {len(manifest)} moves -> {a.web_dir}/manifest.json")


if __name__ == "__main__":
    main()
