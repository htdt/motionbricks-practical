"""Fast, model-free tests for the Kimodo generation and bake contracts."""
import unittest

import numpy as np

from kimodo.bake_kimodo import has_authored_hand_rotation, manifest_entry, minrot
from kimodo.kimogen import best_loop, canonicalize, gate_sample


class KimodoToolTests(unittest.TestCase):
    def test_canonicalize_reanchors_root_and_heading(self):
        idx = {"LeftLeg": 0, "RightLeg": 1, "Hips": 2}
        joints = np.array([
            [[6.0, 0.8, 7.0], [4.0, 0.8, 7.0], [5.0, 1.0, 7.0]],
            [[7.0, 0.8, 7.0], [5.0, 0.8, 7.0], [6.0, 1.0, 7.0]],
        ])
        roots = joints[:, 2].copy()

        out, root, _ = canonicalize(joints, roots, idx)

        np.testing.assert_allclose(root[0, [0, 2]], 0.0, atol=1e-12)
        right = out[0, idx["RightLeg"]] - out[0, idx["LeftLeg"]]
        facing = np.cross([0.0, 1.0, 0.0], right)
        facing /= np.linalg.norm(facing)
        np.testing.assert_allclose(facing, [1.0, 0.0, 0.0], atol=1e-12)

    def test_stance_gate_checks_both_bookends_and_accepts_four_contacts(self):
        names = [
            "LeftHand", "RightHand", "LeftFoot", "RightFoot", "Head",
            "LeftToeBase", "RightToeBase",
        ]
        idx = {name: i for i, name in enumerate(names)}
        joints = np.zeros((4, len(names), 3), dtype=float)
        joints[0, idx["LeftHand"], 0] = 2.0  # bad start, good end
        roots = np.zeros((4, 3), dtype=float)
        stance = {"ee_root_rel": np.zeros((5, 3)).tolist()}
        contacts = np.ones((4, 4), dtype=float)

        gates = gate_sample(
            joints, roots, contacts, idx,
            {"travel": "in_place", "stance_bookend": True}, stance, 30)

        self.assertFalse(gates["stance_ok"])
        self.assertFalse(gates["pass"])
        self.assertGreater(gates["stance_err_start"], gates["stance_err_end"])

    def test_gate_accepts_six_contact_layout(self):
        names = [
            "LeftHand", "RightHand", "LeftFoot", "RightFoot", "Head",
            "LeftToeBase", "LeftToeEnd", "RightToeBase", "RightToeEnd",
        ]
        idx = {name: i for i, name in enumerate(names)}
        joints = np.zeros((4, len(names), 3), dtype=float)
        gates = gate_sample(
            joints, np.zeros((4, 3)), np.ones((4, 6)), idx,
            {"travel": "in_place"}, None, 30)
        self.assertTrue(gates["pass"])

    def test_gate_hard_rejects_skate_and_jitter(self):
        names = [
            "LeftHand", "RightHand", "LeftFoot", "RightFoot", "Head",
            "LeftToeBase", "LeftToeEnd", "RightToeBase", "RightToeEnd",
        ]
        idx = {name: i for i, name in enumerate(names)}
        joints = np.zeros((4, len(names), 3), dtype=float)
        joints[1, :, 0] = 0.3
        joints[2, :, 0] = -0.3
        gates = gate_sample(
            joints, np.zeros((4, 3)), np.ones((4, 6)), idx,
            {"travel": None}, None, 30)
        self.assertFalse(gates["foot_skate_ok"])
        self.assertFalse(gates["jitter_ok"])
        self.assertFalse(gates["pass"])

    def test_best_loop_handles_short_clips(self):
        joints = np.zeros((3, 2, 3), dtype=float)
        roots = np.zeros((3, 3), dtype=float)
        pair, error = best_loop(joints, roots, 30, 96)
        self.assertEqual(pair, (0, 2))
        self.assertEqual(error, 0.0)

    def test_manifest_contains_runtime_metadata(self):
        entry = manifest_entry(
            "walk", 42, 30,
            {"gates": {"pass": True, "loop_trim": [2, 43]}, "frame_data": None},
            {"loop": True})
        self.assertEqual(entry["fps"], 30)
        self.assertTrue(entry["loop"])
        self.assertNotIn("loop_trim", entry["gates"])

    def test_authored_hand_rotation_disables_wrist_stylization(self):
        self.assertTrue(has_authored_hand_rotation([
            {"family": "end-effector", "role": "LeftHand", "rotConstrained": True}]))
        self.assertTrue(has_authored_hand_rotation([
            {"family": "fullbody", "rotConstrained": True,
             "ee": {"LeftHand": {"quat": [0, 0, 0, 1]}}}]))
        self.assertFalse(has_authored_hand_rotation([
            {"family": "end-effector", "role": "LeftFoot", "rotConstrained": True}]))

    def test_minrot_handles_opposite_vectors(self):
        rotation = minrot(np.array([1.0, 0.0, 0.0]),
                          np.array([-1.0, 0.0, 0.0]))
        np.testing.assert_allclose(rotation @ [1.0, 0.0, 0.0],
                                   [-1.0, 0.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)


if __name__ == "__main__":
    unittest.main()
