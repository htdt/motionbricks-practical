"""make_e2e_spec — build the deterministic GPU end-to-end constraint suite.

Writes moveset_e2e.json plus a colocated upstream-format fullbody constraint
sidecar: one move per authoring mode (text only, one
arbitrary full-body target, each hand/foot independently, both hands, both
feet, hands+feet, sparse root waypoints, a dense curved path, text + pose +
end-effector + path combined, constraint-only, and a deliberately
unreachable target). Complete valid SOMA poses are lifted from the upstream
demo constraint files, so every example carries real arrays rather than
placeholders, and constraint files saved by the Kimodo demo/API are consumed
verbatim (03_full_body_keyframes is referenced as a constraints_file).

Usage: kimenv/bin/python make_e2e_spec.py [--demo <examples/kimodo-soma-rp>]
"""
import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEMO = os.environ.get(
    "KIMODO_DEMO",
    os.path.expanduser("~/Downloads/ani_test/kimodo/kimodo/assets/demo/examples/kimodo-soma-rp"),
)
FPS = 30


def load(demo, name):
    with open(os.path.join(demo, name, "constraints.json")) as fp:
        return json.load(fp)


def pose_of(entry, k=0):
    """(local_joints_rot, root_positions) row k of an upstream constraint."""
    return entry["local_joints_rot"][k], entry["root_positions"][k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default=DEFAULT_DEMO)
    ap.add_argument("--out", default=os.path.join(HERE, "moveset_e2e.json"))
    a = ap.parse_args()

    ee_demo = load(a.demo, "04_ee_constraint")
    fullbody_demo = load(a.demo, "03_full_body_keyframes")
    by_type = {c["type"]: c for c in ee_demo}
    reach_rot, reach_root = pose_of(by_type["left-hand"], 0)   # natural reach pose
    place_rot, place_root = pose_of(by_type["left-hand"], 1)   # rotated placing pose
    waypoints = load(a.demo, "06_root_waypoints")[0]

    def ee(ctype, frame, rot=reach_rot, root=reach_root):
        return {"type": ctype, "frame_indices": [frame],
                "local_joints_rot": [rot], "root_positions": [root]}

    # dense quarter-circle path, radius 1.5 m, with tangent headings
    T_path = 240
    path, headings = [], []
    for f in range(T_path):
        t = f / (T_path - 1)
        ang = t * math.pi / 2
        path.append([round(1.5 * math.sin(ang), 4), round(1.5 - 1.5 * math.cos(ang), 4)])
        headings.append([round(math.cos(ang), 6), round(math.sin(ang), 6)])

    moves = [
        # text only (the control example; the MK set covers this class widely)
        {"name": "e2e_text", "duration": 3.0, "travel": None,
         "prompt": "A person waves with their right hand."},

        # one arbitrary full-body keyframe mid-clip, from a demo-saved file
        {"name": "e2e_fb_key", "duration": 4.5, "travel": None,
         "prompt": "A person dances energetically.",
         "constraints_file": os.path.basename(
             os.path.splitext(a.out)[0] + "_fullbody_constraints.json")},

        # each hand / foot independently
        {"name": "e2e_lhand", "duration": 3.0, "travel": None, "reach_policy": "clamp",
         "prompt": "A person reaches out with their left hand to touch something.",
         "constraints": [ee("left-hand", 45)]},
        {"name": "e2e_rhand", "duration": 3.0, "travel": None, "reach_policy": "clamp",
         "prompt": "A person reaches out with their right hand to touch something.",
         "constraints": [ee("right-hand", 45)]},
        {"name": "e2e_lfoot", "duration": 3.0, "travel": None, "reach_policy": "clamp",
         "prompt": "A person takes a careful step.",
         "constraints": [ee("left-foot", 45)]},
        {"name": "e2e_rfoot", "duration": 3.0, "travel": None, "reach_policy": "clamp",
         "prompt": "A person takes a careful step.",
         "constraints": [ee("right-foot", 45)]},

        # both hands via the GENERIC end-effector type (exercises the split)
        {"name": "e2e_hands", "duration": 3.0, "travel": None, "reach_policy": "clamp",
         "prompt": "A person reaches out with both hands to grab something.",
         "constraints": [{"type": "end-effector", "frame_indices": [45],
                          "joint_names": ["LeftHand", "RightHand"],
                          "local_joints_rot": [reach_rot], "root_positions": [reach_root]}]},

        # both feet as co-framed shorthands (same pose -> allowed)
        {"name": "e2e_feet", "duration": 3.0, "travel": None, "reach_policy": "clamp",
         "prompt": "A person hops and lands on both feet.",
         "constraints": [ee("left-foot", 60), ee("right-foot", 60)]},

        # hands + feet at one frame
        {"name": "e2e_hands_feet", "duration": 3.0, "travel": None, "reach_policy": "clamp",
         "prompt": "A person crouches and reaches forward with both hands.",
         "constraints": [{"type": "end-effector", "frame_indices": [50],
                          "joint_names": ["LeftHand", "RightHand", "LeftFoot", "RightFoot"],
                          "local_joints_rot": [reach_rot], "root_positions": [reach_root]}]},

        # a second hand keyframe with a different wrist orientation (rotation
        # adherence at two frames of one clip)
        {"name": "e2e_hand_rot", "duration": 4.0, "travel": None,
         "prompt": "A person picks up an object and places it to the side.",
         "constraints": [{"type": "left-hand", "frame_indices": [40, 100],
                          "local_joints_rot": [reach_rot, place_rot],
                          "root_positions": [reach_root, place_root]}]},

        # sparse root waypoints (the demo-saved waypoint file, embedded inline)
        {"name": "e2e_waypoints", "duration": 6.05, "travel": None,
         "prompt": "A person walks around the room.",
         "constraints": [waypoints]},

        # dense curved path with tangent headings
        {"name": "e2e_path", "duration": 8.0, "travel": None,
         "prompt": "A person walks along a curving path.",
         "constraints": [{"type": "root2d", "frame_indices": list(range(T_path)),
                          "smooth_root_2d": path, "global_root_heading": headings}]},

        # text + full-body key + end-effector + sparse path in ONE move.
        # reach_policy "clamp": on characters whose arms cannot span the
        # mapped reach target, the IK clamps explicitly and QA reports the
        # target as unreachable without failing the move (documented policy).
        {"name": "e2e_mixed", "duration": 6.0, "travel": None,
         "reach_policy": "clamp",
         "prompt": "A person walks forward and reaches for an object.",
         "constraints": [
             {"type": "root2d", "frame_indices": [0, 130],
              "smooth_root_2d": [[reach_root[0], reach_root[2]], [0.9, 0.4]]},
             ee("right-hand", 60),
             {"type": "fullbody", "frame_indices": [110],
              "local_joints_rot": [place_rot],
              "root_positions": [[0.62, place_root[1], 0.28]]},
         ]},

        # constraint-only: no text at all (absence represented explicitly)
        {"name": "e2e_constraint_only", "duration": 6.05, "travel": None,
         "constraints": [waypoints]},
        {"name": "e2e_constraint_only_pose", "duration": 3.0, "travel": None,
         "constraints": [{"type": "fullbody", "frame_indices": [0, 89],
                          "local_joints_rot": [reach_rot, place_rot],
                          "root_positions": [reach_root, place_root]}]},

        # deliberately unreachable: a 4.2 m dash in 1 second against slow text.
        # This move is EXPECTED to fail the root-adherence gate; the e2e
        # runner asserts that it does (explicit failure, not silent success).
        {"name": "e2e_unreachable", "duration": 3.0, "travel": None,
         "prompt": "A person stands still, barely moving.",
         "constraints": [{"type": "root2d", "frame_indices": [30, 60],
                          "smooth_root_2d": [[0.0, 0.0], [4.2, 0.0]]}]},
    ]

    spec = {"comment": "Deterministic GPU end-to-end constraint suite "
                       "(generated by make_e2e_spec.py; poses lifted from the "
                       "upstream demo constraint files). e2e_unreachable is "
                       "expected to be rejected by the adherence gates.",
            "fps": FPS, "moves": moves}
    sidecar = os.path.splitext(a.out)[0] + "_fullbody_constraints.json"
    with open(sidecar, "w") as fp:
        json.dump(fullbody_demo, fp, indent=1)
    with open(a.out, "w") as fp:
        json.dump(spec, fp, indent=1)
    print(f"[e2e] {len(moves)} moves -> {a.out} (+ {sidecar})")


if __name__ == "__main__":
    main()
