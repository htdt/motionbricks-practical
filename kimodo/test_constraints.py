"""Fast, model-free tests for kimoconstraints: schema validation for every
constraint family, invalid shape/value/frame cases, normalization, bookend
merging + conflict detection, mixed families, native +Z -> canonical +X
conversion, and loop-trim frame shifting. Needs the kimodo skeleton classes
(no weights, no GPU): run inside kimenv from the repo root.
"""
import json
import math
import os
import tempfile
import unittest

import numpy as np

try:
    from kimodo import kimoconstraints as kc
    from kimodo.kimogen import best_loop, measure_constraints
except ImportError:
    import kimoconstraints as kc
    from kimogen import best_loop, measure_constraints

NF = 90
FPS = 30
POSE = np.zeros((30, 3))          # identity pose = the SOMA T-pose
ROOT = [0.0, 0.95, 0.0]


def ee(ctype="left-hand", frames=(30,), joint_names=None, **over):
    c = {"type": ctype, "frame_indices": list(frames),
         "local_joints_rot": [POSE.tolist()] * len(frames),
         "root_positions": [list(ROOT)] * len(frames)}
    if joint_names is not None:
        c["joint_names"] = joint_names
    c.update(over)
    return c


def root2d(frames=(0, 45, 89), xz=((0, 0), (0.4, 0.2), (0.9, 0.2)), **over):
    c = {"type": "root2d", "frame_indices": list(frames),
         "smooth_root_2d": [list(p) for p in xz]}
    c.update(over)
    return c


def fullbody(frames=(0,), roots=None, **over):
    roots = roots if roots is not None else [list(ROOT)] * len(frames)
    c = {"type": "fullbody", "frame_indices": list(frames),
         "local_joints_rot": [POSE.tolist()] * len(frames),
         "root_positions": roots}
    c.update(over)
    return c


def prep(constraints):
    return kc.normalize_constraints("mv", constraints, NF, "inline")


def check(constraints):
    norm, meta = prep(constraints)
    kc.check_conflicts("mv", norm, meta, FPS)
    return norm, meta


class ValidationTests(unittest.TestCase):
    def assertRejects(self, constraints, fragment):
        with self.assertRaises(kc.ConstraintSpecError) as ctx:
            check(constraints)
        self.assertIn(fragment, str(ctx.exception))

    def test_every_family_validates(self):
        norm, meta = check([
            root2d(),
            fullbody(frames=(10,)),
            ee("left-hand", frames=(20,)),
            ee("right-hand", frames=(21,)),
            ee("left-foot", frames=(22,)),
            ee("right-foot", frames=(23,)),
            ee("end-effector", frames=(40,), joint_names=["LeftHand", "RightFoot"]),
        ])
        self.assertEqual(len(norm), 8)      # end-effector split into 2 shorthands

    def test_unknown_type(self):
        self.assertRejects([{"type": "warp", "frame_indices": [0]}], "unknown constraint type")

    def test_frames_must_be_sorted_unique_ints_in_range(self):
        self.assertRejects([root2d(frames=(45, 30, 89))], "sorted")
        self.assertRejects([root2d(frames=(0, 0, 89))], "sorted")
        self.assertRejects([root2d(frames=(0, 45, NF))], "lie in [0, 89]")
        self.assertRejects([root2d(frames=(0, 45.0, 89))], "integers")
        self.assertRejects([root2d(frames=(0, True, 89))], "integers")

    def test_shape_mismatches(self):
        self.assertRejects([root2d(frames=(0, 45), xz=((0, 0),))], "shape [2,2]")
        bad = ee(); bad["local_joints_rot"] = [np.zeros((29, 3)).tolist()]
        self.assertRejects([bad], "shape [1,30|77,3]")
        bad = ee(); bad["root_positions"] = [[0.0, 0.95]]
        self.assertRejects([bad], "shape [1,3]")

    def test_nonfinite_and_invalid_values(self):
        bad = root2d(); bad["smooth_root_2d"][1][0] = float("nan")
        self.assertRejects([bad], "non-finite")
        bad = ee(); bad["local_joints_rot"][0][5][0] = 90.0   # degrees, not radians
        self.assertRejects([bad], "radians")
        bad = ee(); bad["root_positions"][0][1] = 5.0
        self.assertRejects([bad], "hip height")
        self.assertRejects([root2d(frames=(0, 1), xz=((0, 0), (0, 0)),
                                   global_root_heading=[[1, 0], [0, 0]])],
                           "zero heading")

    def test_end_effector_vocabulary(self):
        self.assertRejects([ee("end-effector", joint_names=["left_hand"])], "exact casing")
        self.assertRejects([ee("end-effector", joint_names=["Hips"])], "root intent")
        self.assertRejects([ee("end-effector", joint_names=[])], "non-empty")
        self.assertRejects([ee("end-effector", joint_names=["LeftHand", "LeftHand"])], "unique")
        self.assertRejects([ee("left-hand", joint_names=["LeftHand"])],
                           "unsupported fields")

    def test_sparse_frame_limit(self):
        frames = tuple(range(20))
        self.assertRejects([ee("left-hand", frames=frames)], "guidance limit")
        # root2d dense paths are exempt
        n = NF
        check([root2d(frames=tuple(range(n)),
                      xz=tuple((0.01 * i, 0.0) for i in range(n)))])

    def test_same_type_frame_overlap_rejected(self):
        self.assertRejects([ee(frames=(30,)), ee(frames=(30,))], "both constrain frame 30")

    def test_fullbody_vs_ee_overlap_rejected(self):
        self.assertRejects([fullbody(frames=(30,)), ee(frames=(30,))],
                           "already pins every end-effector")

    def test_coframed_ee_must_share_root(self):
        other = ee("right-hand", frames=(30,))
        other["root_positions"] = [[0.5, 0.95, 0.0]]
        self.assertRejects([ee("left-hand", frames=(30,)), other], "disagree on the root")
        # identical pose rows are fine (the demo-04 pattern)
        check([ee("left-hand", frames=(30,)), ee("right-hand", frames=(30,))])

    def test_root2d_vs_pose_xz_conflict(self):
        self.assertRejects(
            [root2d(frames=(30,), xz=((0.5, 0.5),)), ee(frames=(30,))],
            "resolve the contradiction")
        check([root2d(frames=(30,), xz=((0.0, 0.0),)), ee(frames=(30,))])

    def test_root_speed_reachability(self):
        self.assertRejects([root2d(frames=(0, 3), xz=((0, 0), (2.0, 0)))], "unreachable")

    def test_inline_and_file_are_mutually_exclusive(self):
        mv = {"name": "mv", "constraints": [root2d()], "constraints_file": "x.json"}
        with self.assertRaises(kc.ConstraintSpecError) as ctx:
            kc.prepare_move_constraints(mv, NF, FPS, None, ".")
        self.assertIn("mutually exclusive", str(ctx.exception))

    def test_file_loading_matches_inline(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.json")
            with open(path, "w") as fp:
                json.dump([root2d()], fp)
            mv = {"name": "mv", "duration": 3.0, "constraints_file": path}
            cons, meta = kc.prepare_move_constraints(mv, NF, FPS, None, d)
            self.assertEqual(meta[0]["source"], "file")
            mv2 = {"name": "mv", "constraints": [root2d()]}
            cons2, meta2 = kc.prepare_move_constraints(mv2, NF, FPS, None, ".")
            self.assertEqual(cons, cons2)
            self.assertEqual(meta2[0]["source"], "inline")


class BookendTests(unittest.TestCase):
    STANCE = {"local_joints_rot30": POSE.tolist(), "root_pos": ROOT,
              "ee_root_rel": np.zeros((5, 3)).tolist()}

    def test_bookend_merges_with_authored_constraints(self):
        mv = {"name": "mv", "stance_bookend": True,
              "constraints": [ee(frames=(45,))]}
        cons, meta = kc.prepare_move_constraints(mv, NF, FPS, self.STANCE, ".")
        self.assertEqual(len(cons), 2)
        self.assertEqual(cons[-1]["type"], "fullbody")
        self.assertEqual(cons[-1]["frame_indices"], [0, NF - 1])
        self.assertEqual(meta[-1]["source"], "stance_bookend")

    def test_bookend_conflict_is_reported_not_overwritten(self):
        mv = {"name": "mv", "stance_bookend": True,
              "constraints": [ee(frames=(0,))]}
        with self.assertRaises(kc.ConstraintSpecError) as ctx:
            kc.prepare_move_constraints(mv, NF, FPS, self.STANCE, ".")
        self.assertIn("stance_bookend", str(ctx.exception))

    def test_bookend_requires_stance(self):
        mv = {"name": "mv", "stance_bookend": True}
        with self.assertRaises(kc.ConstraintSpecError):
            kc.prepare_move_constraints(mv, NF, FPS, None, ".")


class ResolveAndTransformTests(unittest.TestCase):
    """FK resolution + the native +Z -> canonical +X transform path."""

    @classmethod
    def setUpClass(cls):
        try:
            from kimodo.skeleton.definitions import SOMASkeleton30
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(
                "upstream Kimodo skeleton package is not installed; run inside kimenv") from exc
        cls.sk = SOMASkeleton30()

    def resolve(self, constraints):
        norm, meta = check(constraints)
        return kc.resolve_records("mv", norm, meta, self.sk)

    def test_identity_pose_resolves_to_neutral_fk(self):
        recs = self.resolve([ee("left-hand", frames=(30,))])
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual(r["role"], "LeftHand")
        self.assertEqual(r["frame"], 30)
        # identity pose: world position = neutral joint offset + root
        j = self.sk.bone_index["LeftHand"]
        expected = self.sk.neutral_joints[j].numpy() + np.asarray(ROOT)
        np.testing.assert_allclose(r["pos"], expected, atol=1e-5)
        # identity pose: world rotation = identity quaternion
        self.assertAlmostEqual(abs(r["quat"][3]), 1.0, places=5)
        # the EE constraint also pins root XZ / Y / heading
        np.testing.assert_allclose(r["rootXZ"], [0, 0], atol=1e-9)
        self.assertAlmostEqual(r["rootY"], 0.95)
        # identity pose faces +Z (the SOMA neutral hip line is ~0.07° off exact)
        np.testing.assert_allclose(r["facing"], [0, 1], atol=5e-3)
        self.assertTrue(r["posConstrained"] and r["rotConstrained"])
        self.assertEqual(r["provenance"], "conditioned+corrected")

    def test_native_to_canonical_transform(self):
        recs = self.resolve([ee("left-hand", frames=(30,)),
                             root2d(frames=(0, 45, 89),
                                    global_root_heading=[[1, 0], [1, 0], [0, 1]])])
        # canonicalization for a +Z-facing frame-0: yaw +Z onto +X, shift p0
        c, s = 0.0, 1.0
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        p0 = np.array([0.25, 0.0, -0.5])
        Rt, tt = kc.compose_canonical(R, p0)
        out = kc.transform_records(recs, Rt, tt)
        r = out[0]
        src = recs[0]
        # position: R @ (p - p0)
        np.testing.assert_allclose(
            r["pos"], R @ (np.asarray(src["pos"]) - p0), atol=1e-6)
        # facing +Z becomes +X (up to the SOMA neutral hip-line asymmetry)
        np.testing.assert_allclose(r["facing"], [1, 0], atol=5e-3)
        # heights are untouched by the yaw transform
        self.assertEqual(r["rootY"], src["rootY"])
        # root2d waypoint transforms like a ground point
        w = out[1]
        np.testing.assert_allclose(
            w["rootXZ"],
            (R @ (np.array([0, 0, 0]) - p0))[[0, 2]], atol=1e-6)
        # heading pin [cos,sin]=[0,1] (facing +X) rotates onto -Z... i.e. [0,-1] facing
        h = out[3]                                     # root2d frame 89
        np.testing.assert_allclose(h["facing"], [0, -1], atol=1e-6)

    def test_transform_rejects_non_yaw(self):
        recs = self.resolve([ee("left-hand", frames=(30,))])
        bad = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)  # pitch
        with self.assertRaises(ValueError):
            kc.transform_records(recs, bad, np.zeros(3))
        with self.assertRaises(ValueError):
            kc.transform_records(recs, np.diag([2.0, 1.0, 2.0]), np.zeros(3))

    def test_quaternion_transform_matches_matrix_transform(self):
        pose = POSE.copy()
        pose[13] = [0.3, -0.2, 0.9]                    # bend the left wrist
        c = ee("left-hand", frames=(30,))
        c["local_joints_rot"] = [pose.tolist()]
        recs = self.resolve([c])
        theta = 0.7
        R = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0],
                      [-math.sin(theta), 0, math.cos(theta)]])
        out = kc.transform_records(recs, R, np.zeros(3))
        from scipy.spatial.transform import Rotation as Rot
        got = Rot.from_quat(out[0]["quat"]).as_matrix()
        want = R @ Rot.from_quat(recs[0]["quat"]).as_matrix()
        np.testing.assert_allclose(got, want, atol=1e-6)

    def test_heading_conflict_detected_at_resolution(self):
        # pose implies facing +Z; root2d pins facing +X at the same frame
        norm, meta = check([ee("left-hand", frames=(30,)),
                            root2d(frames=(30,), xz=((0.0, 0.0),),
                                   global_root_heading=[[0, 1]])])
        with self.assertRaises(kc.ConstraintSpecError) as ctx:
            kc.resolve_records("mv", norm, meta, self.sk)
        self.assertIn("two different headings", str(ctx.exception))

    def test_matching_coframed_headings_are_accepted(self):
        # The identity pose faces +Z, represented as [cos(t), sin(t)] = [1, 0]
        # in the upstream root2d schema. Matching pose/root headings must not
        # be mistaken for a conflict.
        norm, meta = check([ee("left-hand", frames=(30,)),
                            root2d(frames=(30,), xz=((0.0, 0.0),),
                                   global_root_heading=[[1, 0]])])
        recs = kc.resolve_records("mv", norm, meta, self.sk)
        self.assertEqual(len(recs), 2)

    def test_demo_constraint_files_load_without_rewriting(self):
        demo = os.environ.get(
            "KIMODO_DEMO",
            os.path.expanduser(
                "~/Downloads/ani_test/kimodo/kimodo/assets/demo/examples/kimodo-soma-rp"))
        if not os.path.isdir(demo):
            self.skipTest("upstream demo examples not available")
        for name, frames in [("04_ee_constraint", 120), ("03_full_body_keyframes", 135),
                             ("06_root_waypoints", 181), ("05_root_path", 300),
                             ("07_mixed_constraints", 152)]:
            raw = kc.load_constraints_file("demo", os.path.join(demo, name, "constraints.json"), ".")
            norm, meta = kc.normalize_constraints("demo", raw, frames, "file")
            kc.check_conflicts("demo", norm, meta, FPS)
            recs = kc.resolve_records("demo", norm, meta, self.sk)
            self.assertTrue(recs, name)


class TrimTests(unittest.TestCase):
    def records(self, frames, required=True):
        return [{"family": "end-effector", "type": "left-hand", "source": "inline",
                 "originalType": "left-hand", "required": required, "frame": f,
                 "posConstrained": True, "rotConstrained": True} for f in frames]

    def test_shift_keeps_and_shifts_retained_frames(self):
        kept = kc.shift_records("mv", self.records([10, 40]), 10, 50)
        self.assertEqual([r["frame"] for r in kept], [0, 30])

    def test_trim_removing_required_constraint_rejected(self):
        with self.assertRaises(kc.ConstraintSpecError) as ctx:
            kc.shift_records("mv", self.records([5, 40]), 10, 50)
        self.assertIn("would remove required constraint", str(ctx.exception))

    def test_non_required_records_outside_trim_are_dropped(self):
        kept = kc.shift_records("mv", self.records([5, 40], required=False), 10, 50)
        self.assertEqual([r["frame"] for r in kept], [30])

    def test_best_loop_must_span_constrained_frames(self):
        rng = np.random.default_rng(7)
        j = rng.normal(size=(120, 4, 3))
        j[100] = j[20]                          # perfect cycle 20..100
        j[80] = j[40]                           # competing shorter cycle 40..80
        r = np.zeros((120, 3))
        (i0, i1), _ = best_loop(j, r, 30, 96)
        self.assertEqual((i0, i1), (20, 100))
        # a required frame at 90 excludes the 40..80 window
        (i0, i1), _ = best_loop(j, r, 30, 96, must_span=(20, 90))
        self.assertLessEqual(i0, 20)
        self.assertGreaterEqual(i1, 90)
        with self.assertRaises(ValueError):
            best_loop(j, r, 10, 20, must_span=(0, 119))


class AdherenceTests(unittest.TestCase):
    """measure_constraints: the authored-target -> final-SOMA stage metric."""

    @classmethod
    def setUpClass(cls):
        try:
            from kimodo.skeleton.definitions import SOMASkeleton30, SOMASkeleton77
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(
                "upstream Kimodo skeleton package is not installed; run inside kimenv") from exc
        cls.sk = SOMASkeleton30()
        cls.sk77 = SOMASkeleton77()
        cls.idx = {n: i for i, n in enumerate(cls.sk77.bone_order_names)}

    def perfect_motion(self, recs, T=NF):
        """Synthesize 77-joint motion that exactly satisfies the records."""
        pj = np.tile(self.sk77.neutral_joints.numpy()[None], (T, 1, 1))
        rp = np.zeros((T, 3)); rp[:, 1] = 0.95
        pj += rp[:, None, :]
        grm = np.tile(np.eye(3)[None, None], (T, 77, 1, 1))
        heading = np.tile(np.array([1.0, 0.0]), (T, 1))
        return pj, rp, grm, heading

    def test_perfect_and_missed_targets(self):
        norm, meta = check([ee("left-hand", frames=(30,))])
        recs = kc.resolve_records("mv", norm, meta, self.sk)
        pj, rp, grm, heading = self.perfect_motion(recs)
        per, summary = measure_constraints(recs, pj, rp, grm, heading, self.idx)
        self.assertTrue(summary["constraint_ok"])
        self.assertLessEqual(summary["ee_pos_max"], 1e-4)
        self.assertLessEqual(summary["ee_rot_max_deg"], 0.01)
        # a 2 cm miss on the wrist must fail the 5 mm gate
        pj2 = pj.copy(); pj2[30, self.idx["LeftHand"]] += [0.02, 0, 0]
        _, s2 = measure_constraints(recs, pj2, rp, grm, heading, self.idx)
        self.assertFalse(s2["constraint_ok"])
        self.assertAlmostEqual(s2["ee_pos_max"], 0.02, places=3)
        # a 5° wrist rotation error must fail the 2° gate
        from scipy.spatial.transform import Rotation as Rot
        grm3 = grm.copy()
        grm3[30, self.idx["LeftHand"]] = Rot.from_euler("y", 5, degrees=True).as_matrix()
        _, s3 = measure_constraints(recs, pj, rp, grm3, heading, self.idx)
        self.assertFalse(s3["constraint_ok"])
        self.assertAlmostEqual(s3["ee_rot_max_deg"], 5.0, places=2)

    def test_root2d_gate(self):
        norm, meta = check([root2d(frames=(0, 45), xz=((0, 0), (0.4, 0.2)))])
        recs = kc.resolve_records("mv", norm, meta, self.sk)
        pj, rp, grm, heading = self.perfect_motion(recs)
        rp = rp.copy(); rp[45, [0, 2]] = [0.4, 0.2]
        _, s = measure_constraints(recs, pj, rp, grm, heading, self.idx)
        self.assertTrue(s["constraint_ok"])
        rp2 = rp.copy(); rp2[45, 0] += 0.05
        _, s2 = measure_constraints(recs, pj, rp2, grm, heading, self.idx)
        self.assertFalse(s2["constraint_ok"])
        self.assertAlmostEqual(s2["root_xz_max"], 0.05, places=3)

    def test_fullbody_end_effector_position_and_rotation_gates(self):
        norm, meta = check([fullbody(frames=(30,))])
        recs = kc.resolve_records("mv", norm, meta, self.sk)
        pj, rp, grm, heading = self.perfect_motion(recs)
        _, perfect = measure_constraints(recs, pj, rp, grm, heading, self.idx)
        self.assertTrue(perfect["constraint_ok"])

        left = self.idx["LeftHand"]
        pj_miss = pj.copy()
        pj_miss[30, left, 0] += 0.02
        _, pos_miss = measure_constraints(recs, pj_miss, rp, grm, heading, self.idx)
        self.assertFalse(pos_miss["constraint_ok"])
        self.assertAlmostEqual(pos_miss["fullbody_ee_pos_max"], 0.02, places=3)

        from scipy.spatial.transform import Rotation as Rot
        grm_miss = grm.copy()
        grm_miss[30, left] = Rot.from_euler("x", 5, degrees=True).as_matrix()
        _, rot_miss = measure_constraints(recs, pj, rp, grm_miss, heading, self.idx)
        self.assertFalse(rot_miss["constraint_ok"])
        self.assertAlmostEqual(rot_miss["fullbody_ee_rot_max_deg"], 5.0, places=2)

    def test_metric_failure_is_loud(self):
        norm, meta = check([ee("left-hand", frames=(80,))])
        recs = kc.resolve_records("mv", norm, meta, self.sk)
        pj, rp, grm, heading = self.perfect_motion(recs, T=40)
        with self.assertRaises(ValueError):
            measure_constraints(recs, pj, rp, grm, heading, self.idx)


class AgentGuideVocabularyTests(unittest.TestCase):
    """ANIMATION_AGENT.md is tested against the supported spec vocabulary so
    it cannot claim controls the wrapper does not implement (and cannot
    silently omit ones it does)."""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "ANIMATION_AGENT.md")
        with open(path) as fp:
            cls.guide = fp.read()

    def test_every_supported_constraint_type_is_documented(self):
        for ctype in kc.KNOWN_TYPES:
            self.assertIn(f"`{ctype}`", self.guide,
                          f"guide must document constraint type {ctype}")

    def test_every_end_effector_token_is_documented_exactly(self):
        for token in kc.EE_TOKENS:
            self.assertIn(f"`{token}`", self.guide,
                          f"guide must state the exact token {token}")

    def test_every_move_spec_control_is_documented(self):
        for field in ("prompt", "constraints", "constraints_file",
                      "stance_bookend", "frame_indices", "smooth_root_2d",
                      "local_joints_rot", "root_positions", "joint_names",
                      "global_root_heading", "duration"):
            self.assertIn(field, self.guide,
                          f"guide must mention the spec field {field}")

    def test_guide_claims_no_unknown_constraint_types(self):
        import re
        # every backticked "type":"..." style token that looks like a
        # constraint type must be a supported one
        for m in re.finditer(r'"type"\s*:\s*"([a-z0-9-]+)"', self.guide):
            self.assertIn(m.group(1), kc.KNOWN_TYPES)

    def test_guide_states_the_actual_limits(self):
        self.assertIn(str(kc.SPARSE_FRAME_LIMIT), self.guide)
        self.assertIn("5 m/s", self.guide)          # MAX_ROOT_SPEED
        self.assertEqual(kc.MAX_ROOT_SPEED, 5.0)


if __name__ == "__main__":
    unittest.main()
