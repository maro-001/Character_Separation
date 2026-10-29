import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

import torch
import torch.nn.functional as F


COMFY_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = COMFY_ROOT.parent
spec = importlib.util.spec_from_file_location("bone_prompt_core", COMFY_ROOT / "custom_nodes/bone_prompt/core.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def attention(q, k, v, heads, **kwargs):
    batch, length, dim = q.shape

    def split(tensor):
        return tensor.reshape(batch, -1, heads, dim // heads).transpose(1, 2)

    result = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=kwargs.get("mask"))
    return result.transpose(1, 2).reshape(batch, length, dim)


class BonePromptTests(unittest.TestCase):
    def setUp(self):
        self.pose = json.loads((WORK_ROOT / "other_scripts/research/assets/openpose_2characters_arms_hips.json").read_text())
        self.bindings = [{"person": 0, "bone": "right_forearm", "text": "red gauntlet"}]

    def test_anatomical_bone_and_person_identity(self):
        first, = core.resolve_bindings(self.pose, self.bindings, 3, 0.1)
        self.assertEqual(first["points"], ((112 / 1024, 340 / 1024), (86 / 1024, 307.5 / 1024), (60 / 1024, 275 / 1024)))
        self.bindings[0]["person"] = 1
        second, = core.resolve_bindings(self.pose, self.bindings, 3, 0.1)
        self.assertEqual(second["points"][0], (552 / 1024, 382 / 1024))
        self.assertEqual(second["points"][-1], (612 / 1024, 495 / 1024))

    def test_missing_joint_does_not_route_to_origin(self):
        self.pose["people"][0]["pose_keypoints_2d"][14] = 0
        with self.assertRaisesRegex(ValueError, "visible"):
            core.resolve_bindings(self.pose, self.bindings, 8, 0)

    def test_bad_layout_person_bone_and_nonfinite_input(self):
        for key, value in (("person", -1), ("person", 2), ("bone", "forearm"), ("strength", float("nan"))):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                core.resolve_bindings(self.pose, [{**self.bindings[0], key: value}], 8, 0.1)
        self.pose["people"][0]["pose_keypoints_2d"] += [0, 0, 0]
        with self.assertRaisesRegex(ValueError, "COCO-18"):
            core.resolve_bindings(self.pose, self.bindings, 8, 0.1)

    def test_outside_points_are_skipped_and_collisions_deduplicated(self):
        points = ((0, 0), (0.01, 0.01), (0.5, 0.5), (-0.1, 0.5), (1, 0.5))
        self.assertEqual(core.sample_indices(points, 4, 8), [0, 20])

    def attention_case(self, strengths=(1.0,)):
        q = torch.zeros(1, 8, 4)
        k = torch.zeros(1, 2 + len(strengths), 4)
        v = torch.zeros_like(k)
        v[:, :2] = 2
        v[:, 2] = 6
        if len(strengths) > 1:
            v[:, 3] = 10
        options = {"n_heads": 2, "activations_shape": [1, 4, 2, 4], "bone_prompt": {
            "base_tokens": 2, "bone_tokens": len(strengths), "bindings": [
                {"points": ((0.25 + i * 0.5, 0),), "start": i, "end": i + 1, "strength": strength}
                for i, strength in enumerate(strengths)
            ],
        }}
        return q, k, v, options

    def test_only_bone_feature_queries_change_without_global_text_leak(self):
        q, k, v, options = self.attention_case()
        result = core.inject_attention(q, k, v, options, attention)
        self.assertTrue(torch.equal(result[:, 1], torch.full((1, 4), 6.0)))
        self.assertTrue(torch.equal(result[:, [0, 2, 3, 4, 5, 6, 7]], torch.full((1, 7, 4), 2.0)))

    def test_independent_bone_text_and_strength(self):
        q, k, v, options = self.attention_case((0.5, 1.0))
        result = core.inject_attention(q, k, v, options, attention)
        self.assertTrue(torch.equal(result[:, 1], torch.full((1, 4), 4.0)))
        self.assertTrue(torch.equal(result[:, 3], torch.full((1, 4), 10.0)))

    def test_zero_strength_matches_unmodified_attention_exactly(self):
        q, k, v, options = self.attention_case((0.0,))
        result = core.inject_attention(q, k, v, options, attention)
        expected = attention(q, k[:, :2], v[:, :2], 2)
        self.assertTrue(torch.equal(result, expected))

    def test_negative_without_bone_metadata_matches_native_attention(self):
        q, k, v, options = self.attention_case()
        options.pop("bone_prompt")
        self.assertTrue(torch.equal(core.inject_attention(q, k, v, options, attention), attention(q, k, v, 2)))

    def test_bone_uses_one_pooled_query(self):
        q = torch.zeros(1, 8, 4)
        q[:, 1, 0], q[:, 2, 0] = -2, 2
        k = torch.zeros(1, 4, 4)
        k[:, 2, 0], k[:, 3, 0] = 3, -3
        v = torch.full_like(k, 2)
        v[:, 2], v[:, 3] = 6, 10
        _, _, _, options = self.attention_case()
        options["bone_prompt"]["bone_tokens"] = 2
        options["bone_prompt"]["bindings"][0].update({
            "points": ((0.25, 0), (0.5, 0)), "end": 2,
        })
        result = core.inject_attention(q, k, v, options, attention)
        torch.testing.assert_close(result[:, [1, 2]], torch.full((1, 2, 4), 8.0))

    def test_bone_update_preserves_local_feature_differences(self):
        q, k, v, options = self.attention_case()
        q[:, 1, 0], q[:, 2, 0] = -2, 2
        k[:, 0, 0], k[:, 1, 0] = -3, 3
        v[:, 0], v[:, 1] = 0, 2
        options["bone_prompt"]["bindings"][0]["points"] = ((0.25, 0), (0.5, 0))
        baseline = attention(q, k[:, :2], v[:, :2], 2)
        result = core.inject_attention(q, k, v, options, attention)
        delta = result[:, [1, 2]] - baseline[:, [1, 2]]
        torch.testing.assert_close(delta[:, 0], delta[:, 1])
        torch.testing.assert_close(result[:, [1, 2]].mean(dim=1), torch.full((1, 4), 6.0))

    def test_shared_joint_deltas_are_averaged(self):
        q, k, v, options = self.attention_case((0.5, 0.5))
        options["bone_prompt"]["bindings"][1]["points"] = ((0.25, 0),)
        result = core.inject_attention(q, k, v, options, attention)
        self.assertTrue(torch.equal(result[:, 1], torch.full((1, 4), 5.0)))

    def test_zero_strength_binding_does_not_dilute_shared_joint(self):
        q, k, v, options = self.attention_case((0.0, 1.0))
        options["bone_prompt"]["bindings"][1]["points"] = ((0.25, 0),)
        result = core.inject_attention(q, k, v, options, attention)
        self.assertTrue(torch.equal(result[:, 1], torch.full((1, 4), 10.0)))

    def test_attention_preserves_batch_dtype_and_input_tensors(self):
        q, k, v, options = self.attention_case()
        for dtype in (torch.float32, torch.float64):
            qb, kb, vb = [tensor.to(dtype).repeat(2, 1, 1) for tensor in (q, k, v)]
            originals = [tensor.clone() for tensor in (qb, kb, vb)]
            output = core.inject_attention(qb, kb, vb, options, attention)
            self.assertEqual(output.shape, qb.shape)
            self.assertEqual(output.dtype, dtype)
            for tensor, original in zip((qb, kb, vb), originals):
                self.assertTrue(torch.equal(tensor, original))

    def test_changed_context_length_fails_clearly(self):
        q, k, v, options = self.attention_case()
        with self.assertRaisesRegex(ValueError, "context length"):
            core.inject_attention(q, k[:, :2], v[:, :2], options, attention)

    def test_reference_text_is_preserved_and_validated(self):
        binding = {**self.bindings[0], "reference_text": "steel armored forearm"}
        resolved, = core.resolve_bindings(self.pose, [binding], 8, 0.1)
        self.assertEqual(resolved["reference_text"], binding["reference_text"])
        with self.assertRaisesRegex(ValueError, "reference_text"):
            core.resolve_bindings(self.pose, [{**binding, "reference_text": 3}], 8, 0.1)

    def test_reference_attention_subtracts_neutral_instead_of_scene(self):
        q, k, v, options = self.attention_case((0.5, 1.0))
        # Target value 6 minus neutral value 10, added to scene value 2.
        options["bone_prompt"]["bindings"] = [{
            **options["bone_prompt"]["bindings"][0], "reference_start": 1, "reference_end": 2,
        }]
        result = core.inject_attention(q, k, v, options, attention)
        torch.testing.assert_close(result[:, 1], torch.zeros(1, 4))
        torch.testing.assert_close(result[:, [0, 2, 3, 4, 5, 6, 7]], torch.full((1, 7, 4), 2.0))

    def test_identical_target_and_reference_leave_scene_unchanged(self):
        q, k, v, options = self.attention_case((1.5, 1.0))
        k[:, 3], v[:, 3] = k[:, 2], v[:, 2]
        options["bone_prompt"]["bindings"] = [{
            **options["bone_prompt"]["bindings"][0], "reference_start": 1, "reference_end": 2,
        }]
        result = core.inject_attention(q, k, v, options, attention)
        torch.testing.assert_close(result, attention(q, k[:, :2], v[:, :2], 2), rtol=0, atol=0)

    def test_minimum_grid_skips_coarse_features(self):
        q, k, v, options = self.attention_case()
        binding = options["bone_prompt"]["bindings"][0]
        binding["min_grid_size"] = 3
        expected = attention(q, k[:, :2], v[:, :2], 2)
        torch.testing.assert_close(core.inject_attention(q, k, v, options, attention), expected, rtol=0, atol=0)
        binding["min_grid_size"] = 2
        self.assertFalse(torch.equal(core.inject_attention(q, k, v, options, attention), expected))

    def test_batch_gates_do_not_dilute_overlapping_active_bones(self):
        q, k, v, options = self.attention_case((1.0, 1.0))
        q, k, v = [tensor.repeat(2, 1, 1) for tensor in (q, k, v)]
        first, second = options["bone_prompt"]["bindings"]
        second["points"] = first["points"]
        first["gate"] = torch.tensor([False, True]).reshape(2, 1, 1)
        second["gate"] = torch.tensor([True, False]).reshape(2, 1, 1)
        result = core.inject_attention(q, k, v, options, attention)
        torch.testing.assert_close(result[0, 1], torch.full((4,), 10.0))
        torch.testing.assert_close(result[1, 1], torch.full((4,), 6.0))

    def test_zero_batch_gate_matches_scene(self):
        q, k, v, options = self.attention_case()
        options["bone_prompt"]["bindings"][0]["gate"] = torch.tensor([False]).reshape(1, 1, 1)
        expected = attention(q, k[:, :2], v[:, :2], 2)
        torch.testing.assert_close(core.inject_attention(q, k, v, options, attention), expected, rtol=0, atol=0)

    def test_tuning_parameters_are_preserved_without_changing_text(self):
        binding = {**self.bindings[0], "min_grid_size": 64, "start_percent": 0.25, "end_percent": 0.85}
        resolved, = core.resolve_bindings(self.pose, [binding], 8, 0.1)
        for key in ("text", "min_grid_size", "start_percent", "end_percent"):
            self.assertEqual(resolved[key], binding[key])
        for key in ("min_grid_size", "start_percent", "end_percent"):
            with self.assertRaisesRegex(ValueError, key):
                core.resolve_bindings(self.pose, [{**binding, key: float("nan")}], 8, 0.1)

    def test_native_comfy_attention_and_model_patcher_integration(self):
        # Isolate ComfyUI's CPU startup policy from other pytest modules.
        result = subprocess.run([sys.executable, "-c", NATIVE_CHECK], cwd=COMFY_ROOT,
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_token_selection_validation(self):
        binding = {**self.bindings[0], "reference_text": "silver", "token_indices": [2, 8, 13],
                   "reference_token_indices": [2, 8, 14]}
        resolved, = core.resolve_bindings(self.pose, [binding], 8, 0.1)
        self.assertEqual(resolved["token_indices"], [2, 8, 13])
        self.assertIsNot(resolved["token_indices"], binding["token_indices"])
        for indices in ([], [1, 1], [-1], [True], [1.5], "2"):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "token_indices"):
                core.resolve_bindings(self.pose, [{**binding, "token_indices": indices}], 8, 0.1)
        with self.assertRaisesRegex(ValueError, "requires reference_text"):
            core.resolve_bindings(self.pose, [{**self.bindings[0], "reference_token_indices": [2]}], 8, 0.1)

    def test_opposite_routes_same_person_and_protects_explicit_target(self):
        binding = {**self.bindings[0], "opposite_strength": 0.5}
        resolved, = core.resolve_bindings(self.pose, [binding], 8, 0.1)
        opposite, = core.resolve_bindings(self.pose, [{**self.bindings[0], "bone": "left_forearm"}], 8, 0.1)
        self.assertEqual(resolved["opposite_bone"], "left_forearm")
        self.assertEqual(resolved["opposite_points"], opposite["points"])
        explicit = core.resolve_bindings(self.pose, [binding, {**binding, "bone": "left_forearm"}], 8, 0.1)
        self.assertTrue(all("opposite_points" not in item for item in explicit))
        with self.assertRaisesRegex(ValueError, "no anatomical opposite"):
            core.resolve_bindings(self.pose, [{**binding, "bone": "neck"}], 8, 0.1)

    def test_competition_subtracts_other_attribute_only_at_target_samples(self):
        q, k, v, options = self.attention_case((1.0, 1.0))
        options["bone_prompt"]["bindings"][0]["competition_strength"] = 1
        result = core.inject_attention(q, k, v, options, attention)
        torch.testing.assert_close(result[:, 1], torch.full((1, 4), -2.0))
        torch.testing.assert_close(result[:, 3], torch.full((1, 4), 10.0))
        torch.testing.assert_close(result[:, [0, 2, 4, 5, 6, 7]], torch.full((1, 6, 4), 2.0))

    def test_competition_excludes_inactive_and_negative_bindings_per_batch(self):
        q, k, v, options = self.attention_case((1.0, 1.0))
        q, k, v = [tensor.repeat(2, 1, 1) for tensor in (q, k, v)]
        first, second = options["bone_prompt"]["bindings"]
        first["competition_strength"] = 1
        second["gate"] = torch.tensor([False, True]).reshape(2, 1, 1)
        result = core.inject_attention(q, k, v, options, attention)
        torch.testing.assert_close(result[:, 1, 0], torch.tensor([6.0, -2.0]))
        second["strength"] = -0.5
        result = core.inject_attention(q, k, v, options, attention)
        torch.testing.assert_close(result[:, 1], torch.full((2, 4), 6.0))
        torch.testing.assert_close(result[:, 3, 0], torch.tensor([2.0, -2.0]))

    def test_invalid_tuning_windows_and_suppression_strengths(self):
        for settings in ({"start_percent": 0.7, "end_percent": 0.3}, {"start_percent": -0.1},
                         {"end_percent": 1.1}, {"competition_strength": -1}, {"opposite_strength": -1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                core.resolve_bindings(self.pose, [{**self.bindings[0], **settings}], 8, 0.1)

    def test_sample_queries_keep_local_attention_instead_of_repeating_bone_mean(self):
        q, k, v, options = self.attention_case((1.0, 1.0))
        q[:, 1, 0], q[:, 2, 0] = -2, 2
        k[:, 2, 0], k[:, 3, 0] = -3, 3
        options["bone_prompt"]["bindings"] = [{
            **options["bone_prompt"]["bindings"][0], "end": 2,
            "points": ((0.25, 0), (0.5, 0)), "query_mode": "sample",
        }]
        result = core.inject_attention(q, k, v, options, attention)
        expected = attention(q[:, [1, 2]], k[:, 2:], v[:, 2:], 2)
        torch.testing.assert_close(result[:, [1, 2]], expected)
        self.assertFalse(torch.allclose(result[:, 1], result[:, 2]))
        torch.testing.assert_close(result[:, [0, 3, 4, 5, 6, 7]], torch.full((1, 6, 4), 2.0))

    def test_block_selection_excludes_inactive_competitors(self):
        q, k, v, options = self.attention_case((1.0, 1.0))
        first, second = options["bone_prompt"]["bindings"]
        first.update(competition_strength=1, block_types=["output"])
        second["block_types"] = ["input"]
        options["block"] = ("output", 0)
        result = core.inject_attention(q, k, v, options, attention)
        torch.testing.assert_close(result[:, 1], torch.full((1, 4), 6.0))
        torch.testing.assert_close(result[:, 3], torch.full((1, 4), 2.0))
        options["block"] = ("input", 0)
        result = core.inject_attention(q, k, v, options, attention)
        torch.testing.assert_close(result[:, 1], torch.full((1, 4), 2.0))
        torch.testing.assert_close(result[:, 3], torch.full((1, 4), 10.0))
        options["block"] = ("middle", 0)
        torch.testing.assert_close(core.inject_attention(q, k, v, options, attention),
                                   attention(q, k[:, :2], v[:, :2], 2), rtol=0, atol=0)

    def test_sample_queries_use_matching_reference_query_for_each_location(self):
        torch.manual_seed(19)
        q, k, v = [torch.randn(2, length, 8, dtype=torch.float64) for length in (8, 6, 6)]
        binding = {"points": ((0.25, 0), (0.5, 0)), "start": 0, "end": 2,
                   "reference_start": 2, "reference_end": 4, "query_mode": "sample", "strength": 0.5}
        options = {"n_heads": 2, "activations_shape": [2, 8, 2, 4],
                   "bone_prompt": {"base_tokens": 2, "bone_tokens": 4, "bindings": [binding]}}
        baseline = attention(q, k[:, :2], v[:, :2], 2)
        expected = baseline.clone()
        expected[:, [1, 2]] += 0.5 * (attention(q[:, [1, 2]], k[:, 2:4], v[:, 2:4], 2)
                                             - attention(q[:, [1, 2]], k[:, 4:6], v[:, 4:6], 2))
        torch.testing.assert_close(core.inject_attention(q, k, v, options, attention), expected)

    def test_query_mode_and_block_types_validation(self):
        binding = {**self.bindings[0], "query_mode": "sample", "block_types": ["output"]}
        resolved, = core.resolve_bindings(self.pose, [binding], 8, 0.1)
        self.assertEqual(resolved["query_mode"], "sample")
        self.assertEqual(resolved["block_types"], ["output"])
        self.assertIsNot(resolved["block_types"], binding["block_types"])
        for fields in ({"query_mode": "pixel"}, {"block_types": []},
                       {"block_types": "output"}, {"block_types": ["input", "unknown"]}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                core.resolve_bindings(self.pose, [{**binding, **fields}], 8, 0.1)

    def test_tubular_sampling_and_unbound_arm_isolation_points(self):
        resolved, = core.resolve_bindings(self.pose, self.bindings, 8, 0.1)
        self.assertIn("tube_points", resolved)
        self.assertIn("unbound_points", resolved)
        self.assertGreater(len(resolved["tube_points"]), len(resolved["points"]))
        bound_idx = set(core.sample_indices(resolved["tube_points"], 64, 64))
        unbound_idx = set(core.sample_indices(resolved["unbound_points"], 64, 64))
        self.assertTrue(bound_idx)
        self.assertTrue(unbound_idx - bound_idx)

    def test_self_attention_isolation_prevents_bound_bone_leak_to_scene_and_other_limbs(self):
        q = torch.zeros(1, 4, 4)
        k = torch.zeros(1, 4, 4)
        v = torch.tensor([[[1.0, 0, 0, 0], [100.0, 0, 0, 0], [200.0, 0, 0, 0], [3.0, 0, 0, 0]]])
        options = {
            "n_heads": 2,
            "activations_shape": [1, 4, 2, 2],
            "bone_prompt": {
                "base_tokens": 2,
                "bone_tokens": 2,
                "bindings": [
                    {"points": ((0.75, 0.25),), "strength": 1.0, "unbound_points": ((0.75, 0.75),)},
                    {"points": ((0.25, 0.75),), "strength": 1.0, "unbound_points": ((0.75, 0.75),)},
                ],
            },
        }
        result = core.isolate_self_attention(q, k, v, options, attention)
        # Unbound indices [0, 3] only attend to [0, 3] -> mean of 1.0 and 3.0 is 2.0 (no leak from 100 or 200)
        torch.testing.assert_close(result[:, [0, 3], 0], torch.tensor([[2.0, 2.0]]))
        # Bound index 1 attends to [0, 1] (excluding other bound index 2 and unbound arm index 3) -> mean(1, 100) = 50.5
        torch.testing.assert_close(result[:, 1, 0], torch.tensor([50.5]))

    def test_attribute_indices_reinforce_weaker_binding_delta(self):
        q = torch.zeros(1, 4, 4)
        k = torch.zeros(1, 6, 4)
        v = torch.zeros(1, 6, 4)
        v[:, :2] = 1.0
        v[:, 2:4] = torch.tensor([[[4.0, 0, 0, 0], [8.0, 0, 0, 0]]])
        v[:, 4:6] = 0.0
        options = {
            "n_heads": 2,
            "activations_shape": [1, 4, 2, 2],
            "bone_prompt": {
                "base_tokens": 2,
                "bone_tokens": 4,
                "bindings": [{
                    "points": ((0.25, 0.25),),
                    "start": 0,
                    "end": 2,
                    "reference_start": 2,
                    "reference_end": 4,
                    "strength": 1.0,
                    "attribute_indices": [1],
                    "attribute_strength": 0.5,
                }],
            },
        }
        result = core.inject_attention(q, k, v, options, attention)
        # Base is 1.0, full delta is mean(4, 8) - 0 = 6.0, attr delta at index 1 is 8.0 - 0 = 8.0 * 0.5 = 4.0 -> total 11.0
        torch.testing.assert_close(result[:, 0, 0], torch.tensor([11.0]))



NATIVE_CHECK = r'''
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import torch
import comfy.cli_args
comfy.cli_args.args.cpu = True
import comfy.conds
import comfy.model_base
import comfy.model_patcher
import comfy.sample
import comfy.supported_models_base
from comfy.ldm.modules.attention import BasicTransformerBlock
from custom_nodes import bone_prompt

torch.manual_seed(7)
layer = BasicTransformerBlock(8, 2, 4, context_dim=2048, operations=torch.nn)
x = torch.randn(1, 8, 8)
scene = torch.randn(1, 3, 2048)
bone = torch.randn(1, 2, 2048)
layout = [{"points": ((0.25, 0),), "start": 0, "end": 2, "strength": 0.5}]
options = {"activations_shape": [1, 8, 2, 4], "block": ("input", 0)}
baseline = layer(x, context=scene, transformer_options=options.copy())
patched_options = {**options, "bone_prompt": {"base_tokens": 3, "bone_tokens": 2, "bindings": layout},
                   "patches_replace": {"attn2": {("input", 0): bone_prompt.sparse_cross_attention}}}
actual = layer(x, context=torch.cat((scene, bone), dim=1), transformer_options=patched_options)
untouched = [0, 2, 3, 4, 5, 6, 7]
torch.testing.assert_close(actual[:, untouched], baseline[:, untouched], rtol=1e-5, atol=1e-6)
assert not torch.allclose(actual[:, 1], baseline[:, 1])

class Clip:
    def tokenize(self, text):
        return text
    def encode_from_tokens_scheduled(self, tokens):
        return [[bone, {"start_percent": 0, "end_percent": 0.5}],
                [bone * 2, {"start_percent": 0.5, "end_percent": 1}]]

metadata = {"pooled_output": torch.ones(1, 1280), "model_conds": {"existing": comfy.conds.CONDConstant(1)}}
binding = {"person": 0, "bone": "right_forearm", "text": "red", **layout[0]}
encoded = bone_prompt.encode_bindings(Clip(), [[scene, metadata]], [binding])
assert len(encoded) == 2
assert encoded[0][1]["end_percent"] == 0.5
assert encoded[1][1]["start_percent"] == 0.5
assert "bone_prompt_embeddings" not in metadata["model_conds"]
assert encoded[0][0] is scene
assert encoded[0][1]["pooled_output"] is metadata["pooled_output"]

class ReferenceClip(Clip):
    def encode_from_tokens_scheduled(self, tokens):
        if tokens == "neutral":
            return [[torch.full((1, 3, 2048), 7.0), {"start_percent": 0.25, "end_percent": 0.75}]]
        return super().encode_from_tokens_scheduled(tokens)

reference_encoded = bone_prompt.encode_bindings(ReferenceClip(), [[scene, metadata]], [
    {**binding, "reference_text": "neutral"}, {**binding, "person": 1},
])
assert len(reference_encoded) == 2
assert [(meta["start_percent"], meta["end_percent"]) for _, meta in reference_encoded] == [(0.25, 0.5), (0.5, 0.75)]
for _, meta in reference_encoded:
    conditions = meta["model_conds"]
    embeddings = conditions[bone_prompt.EMBEDDING_KEY].cond
    first, second = conditions[bone_prompt.LAYOUT_KEY].cond
    assert (first["start"], first["end"], first["reference_start"], first["reference_end"]) == (0, 2, 2, 5)
    assert (second["start"], second["end"]) == (5, 7)
    assert "reference_start" not in second
    assert embeddings.shape == (1, 7, 2048)
    torch.testing.assert_close(embeddings[:, 2:5], torch.full((1, 3, 2048), 7.0))

config = comfy.supported_models_base.BASE({
    "image_size": 8, "in_channels": 4, "out_channels": 4, "model_channels": 32,
    "num_res_blocks": [1], "channel_mult": [1], "use_spatial_transformer": True,
    "transformer_depth": [1], "transformer_depth_output": [1, 1],
    "transformer_depth_middle": 1, "context_dim": 2048, "dtype": torch.float32,
})
config.unet_config["num_head_channels"] = 8
model = comfy.model_base.BaseModel(config, device=torch.device("cpu"))
for parameter in model.parameters():
    torch.nn.init.normal_(parameter, std=0.02)
patcher = comfy.model_patcher.ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
pose = Path("../other_scripts/research/assets/openpose_2characters_arms_hips.json").read_text()
with TemporaryDirectory() as directory, patch.object(bone_prompt.folder_paths, "get_input_directory", return_value=directory):
    Path(directory, "pose.json").write_text(pose)
    patched, positive = bone_prompt.BonePromptInject().inject(patcher, Clip(), [[scene, metadata]], "pose.json",
        json.dumps([{"person": 0, "bone": "right_forearm", "text": "red"}]), 8, 0.1)
    assert "patches_replace" not in patcher.model_options["transformer_options"]
    assert patched is not patcher
    assert patched.model_options["transformer_options"]["patches_replace"]["attn2"]
    before = bone_prompt.BonePromptInject.IS_CHANGED("pose.json")
    Path(directory, "pose.json").write_text(pose + "\n")
    assert before != bone_prompt.BonePromptInject.IS_CHANGED("pose.json")
    for filename in ("../outside.json", "/tmp/outside.json", "pose.png"):
        try:
            bone_prompt.pose_path(filename)
        except ValueError:
            pass
        else:
            raise AssertionError(filename)
    cond = positive[0][1]["model_conds"]
    assert not comfy.conds.CONDConstant(cond[bone_prompt.LAYOUT_KEY].cond).can_concat(comfy.conds.CONDConstant([]))
    captured = {}
    def executor(x, t, **kwargs):
        captured.update(kwargs)
        return x
    kwargs = {key: value.cond for key, value in cond.items() if key != "existing"}
    bone_prompt.apply_bone_context(executor, x, torch.ones(1), c_crossattn=scene, transformer_options={}, **kwargs)
    assert captured["c_crossattn"].shape[1] == scene.shape[1] + bone.shape[1]
    assert bone_prompt.LAYOUT_KEY not in captured
    assert "bone_prompt" in captured["transformer_options"]
    assert all(not isinstance(value, torch.Tensor) for value in patched.model_options["transformer_options"].values())
    sampled = comfy.sample.sample(patched, torch.randn(1, 4, 8, 8), 2, 4.0, "euler", "normal", positive,
        [[torch.randn_like(scene), {}]], torch.zeros(1, 4, 8, 8), disable_pbar=True, seed=7)
    assert sampled.shape == (1, 4, 8, 8)
    assert torch.isfinite(sampled).all()
    reference_patched, reference_positive = bone_prompt.BonePromptInject().inject(
        patcher, ReferenceClip(), [[scene, metadata]], "pose.json",
        json.dumps([{"person": 0, "bone": "right_forearm", "text": "red", "reference_text": "neutral"}]), 8, 0.1)
    reference_sampled = comfy.sample.sample(reference_patched, torch.randn(1, 4, 8, 8), 2, 4.0,
        "euler", "normal", reference_positive, [[torch.randn_like(scene), {}]],
        torch.zeros(1, 4, 8, 8), disable_pbar=True, seed=7)
    assert reference_sampled.shape == (1, 4, 8, 8)
    assert torch.isfinite(reference_sampled).all()
    scheduled_patched, scheduled_positive = bone_prompt.BonePromptInject().inject(
        patcher, ReferenceClip(), [[scene, metadata]], "pose.json",
        json.dumps([{"person": 0, "bone": "right_forearm", "text": "red", "reference_text": "neutral",
                     "min_grid_size": 2, "start_percent": 0.25, "end_percent": 0.85}]), 8, 0.1)
    scheduled_conds = scheduled_positive[0][1]["model_conds"]
    scheduled_layout = scheduled_conds[bone_prompt.LAYOUT_KEY].cond
    item, = scheduled_layout
    assert item["min_grid_size"] == 2
    assert item["start_sigma"] > item["end_sigma"]
    midpoint = (item["start_sigma"] + item["end_sigma"]) / 2
    bone_prompt.apply_bone_context(executor, x.repeat(2, 1, 1),
        torch.tensor([midpoint, item["start_sigma"] * 2]), c_crossattn=scene.repeat(2, 1, 1),
        transformer_options={}, **{
            bone_prompt.LAYOUT_KEY: scheduled_layout,
            bone_prompt.EMBEDDING_KEY: scheduled_conds[bone_prompt.EMBEDDING_KEY].cond.repeat(2, 1, 1),
        })
    torch.testing.assert_close(captured["transformer_options"]["bone_prompt"]["bindings"][0]["gate"].flatten(),
                               torch.tensor([True, False]))
    assert "gate" not in item
    scheduled_sampled = comfy.sample.sample(scheduled_patched, torch.randn(1, 4, 8, 8), 4, 4.0,
        "euler", "normal", scheduled_positive, [[torch.randn_like(scene), {}]],
        torch.zeros(1, 4, 8, 8), disable_pbar=True, seed=7)
    assert scheduled_sampled.shape == (1, 4, 8, 8)
    assert torch.isfinite(scheduled_sampled).all()

    class TokenClip(Clip):
        def encode_from_tokens_scheduled(self, tokens):
            offset = 10 if tokens == "neutral" else 0
            values = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1).expand(1, 5, 2048) + offset
            return [[values, {}]]

    routed_bindings = [{"person": 0, "bone": "right_forearm", "text": "red", "reference_text": "neutral",
                        "token_indices": [3, 1], "reference_token_indices": [2], "strength": 1,
                        "opposite_strength": 0.5, "competition_strength": 1,
                        "query_mode": "sample", "block_types": ["output"]}]
    routed_patched, routed_positive = bone_prompt.BonePromptInject().inject(
        patcher, TokenClip(), [[scene, metadata]], "pose.json", json.dumps(routed_bindings), 8, 0.1)
    conditions = routed_positive[0][1]["model_conds"]
    target, opposite = conditions[bone_prompt.LAYOUT_KEY].cond
    assert (target["start"], target["end"], target["reference_start"], target["reference_end"]) == (0, 2, 2, 3)
    assert opposite["bone"] == "left_forearm" and opposite["strength"] == -0.5
    assert opposite["competition_strength"] == 0 and opposite["points"] != target["points"]
    assert target["query_mode"] == opposite["query_mode"] == "sample"
    assert target["block_types"] == opposite["block_types"] == ["output"]
    assert opposite["start"] == target["start"] and opposite["reference_start"] == target["reference_start"]
    torch.testing.assert_close(conditions[bone_prompt.EMBEDDING_KEY].cond[0, :, 0], torch.tensor([3., 1., 12.]))
    routed_sampled = comfy.sample.sample(routed_patched, torch.randn(1, 4, 8, 8), 2, 4.0,
        "euler", "normal", routed_positive, [[torch.randn_like(scene), {}]],
        torch.zeros(1, 4, 8, 8), disable_pbar=True, seed=7)
    assert torch.isfinite(routed_sampled).all()
    paired, paired_positive = bone_prompt.BonePromptInject().inject(
        patcher, TokenClip(), [[scene, metadata]], "pose.json",
        json.dumps(routed_bindings + [{**routed_bindings[0], "person": 1, "bone": "left_forearm", "text": "blue"}]), 8, 0.1)
    paired_layout = paired_positive[0][1]["model_conds"][bone_prompt.LAYOUT_KEY].cond
    assert [(item["person"], item["bone"]) for item in paired_layout] == [
        (0, "right_forearm"), (0, "left_forearm"), (1, "left_forearm"), (1, "right_forearm")]
    assert paired_layout[2]["start"] == 3 and paired_layout[3]["start"] == 3
    paired_sampled = comfy.sample.sample(paired, torch.randn(1, 4, 8, 8), 2, 4.0,
        "euler", "normal", paired_positive, [[torch.randn_like(scene), {}]],
        torch.zeros(1, 4, 8, 8), disable_pbar=True, seed=7)
    assert torch.isfinite(paired_sampled).all()
    try:
        bone_prompt.encode_bindings(TokenClip(), [[scene, metadata]], [{**binding, "token_indices": [5]}])
    except ValueError as error:
        assert "exceeds encoded length" in str(error)
    else:
        raise AssertionError("Out-of-range token selection was silently accepted")

    class GraphClip(Clip):
        def __init__(self):
            self.calls = []
        def encode_from_tokens_scheduled(self, tokens):
            self.calls.append(tokens)
            return super().encode_from_tokens_scheduled(tokens)

    graph_clip = GraphClip()
    body_bones = ("right_upper_arm", "left_upper_arm", "right_forearm", "left_forearm", "right_torso",
                  "left_torso", "right_thigh", "left_thigh", "right_shin", "left_shin")
    graph_config = [{"person": person, "bone": bone_name, "text": "red" if index % 2 else "blue",
                     "routing_mode": "graph", "strength": 0.8, "radius": 0.02, "overlap_mode": "nearest"}
                    for person in range(2) for index, bone_name in enumerate(body_bones)]
    graph_patched, graph_positive = bone_prompt.BonePromptInject().inject(
        patcher, graph_clip, [[scene, metadata]], "pose.json", json.dumps(graph_config), 8, 0.1)
    assert graph_clip.calls == ["blue", "red"]
    assert len(graph_positive) == 2
    graph_conditions = graph_positive[0][1]["model_conds"]
    graph_layout = graph_conditions[bone_prompt.LAYOUT_KEY].cond
    assert len(graph_layout) == 20
    assert all(item["overlap_mode"] == "nearest" for item in graph_layout)
    assert all(item["routing_mode"] == "graph" and "tube_points" not in item for item in graph_layout)
    assert graph_conditions[bone_prompt.EMBEDDING_KEY].cond.shape == (1, 4, 2048)
    assert graph_layout[0]["start"] == graph_layout[2]["start"]
    assert graph_patched.model_options["transformer_options"]["patches_replace"]["attn1"]
    graph_kwargs = {key: value.cond for key, value in graph_conditions.items() if key != "existing"}
    bone_prompt.apply_bone_context(executor, x, torch.ones(1), c_crossattn=scene,
                                  transformer_options={}, **graph_kwargs)
    first_cache = captured["transformer_options"]["bone_prompt"]["route_cache"]
    assert all(item["overlap_mode"] == "nearest"
               for item in captured["transformer_options"]["bone_prompt"]["bindings"])
    first_cache[(8, 8)] = "sentinel"
    bone_prompt.apply_bone_context(executor, x, torch.ones(1), c_crossattn=scene,
                                  transformer_options={}, **graph_kwargs)
    assert captured["transformer_options"]["bone_prompt"]["route_cache"] == {}
    assert captured["transformer_options"]["bone_prompt"]["route_cache"] is not first_cache
    graph_noise = torch.randn(1, 4, 8, 8)
    graph_negative = [[torch.randn_like(scene), {}]]
    graph_sampled = comfy.sample.sample(graph_patched, graph_noise, 2, 4.0,
        "euler", "normal", graph_positive, graph_negative,
        torch.zeros(1, 4, 8, 8), disable_pbar=True, seed=7)
    assert torch.isfinite(graph_sampled).all() and graph_sampled.shape == (1, 4, 8, 8)
    assert "route_cache" not in graph_layout[0]
    sampled_routes_config = [{**binding, "route_samples": 4} for binding in graph_config]
    sampled_routes_model, sampled_routes_positive = bone_prompt.BonePromptInject().inject(
        patcher, graph_clip, [[scene, metadata]], "pose.json", json.dumps(sampled_routes_config), 8, 0.1)
    sampled_routes_layout = sampled_routes_positive[0][1]["model_conds"][bone_prompt.LAYOUT_KEY].cond
    assert all(item["route_samples"] == 4 for item in sampled_routes_layout)
    sampled_routes_result = comfy.sample.sample(sampled_routes_model, graph_noise, 2, 4.0,
        "euler", "normal", sampled_routes_positive, graph_negative,
        torch.zeros(1, 4, 8, 8), disable_pbar=True, seed=7)
    assert torch.isfinite(sampled_routes_result).all()
    zero_bias_config = [{**binding, "token_bias": 0, "token_bias_indices": [1]} for binding in graph_config]
    zero_bias_patched, zero_bias_positive = bone_prompt.BonePromptInject().inject(
        patcher, graph_clip, [[scene, metadata]], "pose.json", json.dumps(zero_bias_config), 8, 0.1)
    zero_bias_sampled = comfy.sample.sample(zero_bias_patched, graph_noise, 2, 4.0,
        "euler", "normal", zero_bias_positive, graph_negative,
        torch.zeros(1, 4, 8, 8), disable_pbar=True, seed=7)
    torch.testing.assert_close(zero_bias_sampled, graph_sampled, rtol=0, atol=0)
    biased_config = [{**binding, "token_bias": 2, "token_bias_indices": [1]} for binding in graph_config]
    biased_patched, biased_positive = bone_prompt.BonePromptInject().inject(
        patcher, graph_clip, [[scene, metadata]], "pose.json", json.dumps(biased_config), 8, 0.1)
    biased_layout = biased_positive[0][1]["model_conds"][bone_prompt.LAYOUT_KEY].cond
    assert all(item["token_bias"] == 2 and item["token_bias_indices"] == [1] for item in biased_layout)
    biased_sampled = comfy.sample.sample(biased_patched, torch.randn(1, 4, 8, 8), 2, 4.0,
        "euler", "normal", biased_positive, [[torch.randn_like(scene), {}]],
        torch.zeros(1, 4, 8, 8), disable_pbar=True, seed=7)
    assert torch.isfinite(biased_sampled).all()
    graph_clip.calls.clear()
    contrast_config = [{**binding, "reference_binding": [binding["person"], "right_torso"],
                        "delta_scale": 0.5} for binding in biased_config]
    contrast_patched, contrast_positive = bone_prompt.BonePromptInject().inject(
        patcher, graph_clip, [[scene, metadata]], "pose.json", json.dumps(contrast_config), 8, 0.1)
    assert graph_clip.calls == ["blue", "red"]
    contrast_conditions = contrast_positive[0][1]["model_conds"]
    contrast_layout = contrast_conditions[bone_prompt.LAYOUT_KEY].cond
    references = {(item["person"], item["bone"]): item for item in contrast_layout}
    for item in contrast_layout:
        reference = references[item["person"], "right_torso"]
        assert (item["reference_start"], item["reference_end"]) == (reference["start"], reference["end"])
        assert item["reference_token_bias"] == 2 and item["reference_token_bias_indices"] == [1]
        assert item["delta_scale"] == 0.5
    assert contrast_conditions[bone_prompt.EMBEDDING_KEY].cond.shape == (1, 4, 2048)
    contrast_sampled = comfy.sample.sample(contrast_patched, graph_noise, 2, 4.0,
        "euler", "normal", contrast_positive, graph_negative,
        torch.zeros(1, 4, 8, 8), disable_pbar=True, seed=7)
    assert torch.isfinite(contrast_sampled).all()
    target_config = [{**binding, "reference_attention": "target"} for binding in contrast_config]
    target_patched, target_positive = bone_prompt.BonePromptInject().inject(
        patcher, graph_clip, [[scene, metadata]], "pose.json", json.dumps(target_config), 8, 0.1)
    target_layout = target_positive[0][1]["model_conds"][bone_prompt.LAYOUT_KEY].cond
    assert all(item["reference_attention"] == "target" for item in target_layout)
    target_sampled = comfy.sample.sample(target_patched, graph_noise, 2, 4.0,
        "euler", "normal", target_positive, graph_negative,
        torch.zeros(1, 4, 8, 8), disable_pbar=True, seed=7)
    assert torch.isfinite(target_sampled).all()
    try:
        bone_prompt.BonePromptInject().inject(patcher, graph_clip, [[scene, metadata]],
            "pose.json", json.dumps([{**binding, "token_indices": [0], "token_bias_indices": [0]} if index == 0 else binding
                                      for index, binding in enumerate(target_config)]), 8, 0.1)
    except ValueError as error:
        assert "equal target/reference context lengths" in str(error)
    else:
        raise AssertionError("Unequal target/reference value sequences were accepted")
    partition_config = [{**binding, "self_attention_scope": "partition", "self_attention_strength": 0.25,
                         "joint_falloff": 0.2}
                        for binding in contrast_config]
    partition_patched, partition_positive = bone_prompt.BonePromptInject().inject(
        patcher, graph_clip, [[scene, metadata]], "pose.json", json.dumps(partition_config), 8, 0.1)
    partition_layout = partition_positive[0][1]["model_conds"][bone_prompt.LAYOUT_KEY].cond
    assert all(item["self_attention_scope"] == "partition" for item in partition_layout)
    assert all(item["joint_falloff"] == 0.2 for item in partition_layout)
    partition_sampled = comfy.sample.sample(partition_patched, graph_noise, 2, 4.0,
        "euler", "normal", partition_positive, graph_negative,
        torch.zeros(1, 4, 8, 8), disable_pbar=True, seed=7)
    assert torch.isfinite(partition_sampled).all()
    for fields in ({"token_bias_indices": [2]}, {"token_indices": [1], "token_bias_indices": [1]}):
        try:
            bone_prompt.BonePromptInject().inject(patcher, graph_clip, [[scene, metadata]],
                "pose.json", json.dumps([{**biased_config[0], **fields}]), 8, 0.1)
        except ValueError as error:
            assert "attended context length" in str(error)
        else:
            raise AssertionError("Out-of-range attention bias was silently accepted")
    graph_clip.calls.clear()
    selected_config = [{**binding, "token_indices": [1]} for binding in graph_config]
    _, selected_positive = bone_prompt.BonePromptInject().inject(
        patcher, graph_clip, [[scene, metadata]], "pose.json", json.dumps(selected_config), 8, 0.1)
    assert graph_clip.calls == ["blue", "red"]
    selected_conditions = selected_positive[0][1]["model_conds"]
    assert selected_conditions[bone_prompt.EMBEDDING_KEY].cond.shape == (1, 2, 2048)
    selected_layout = selected_conditions[bone_prompt.LAYOUT_KEY].cond
    assert all(item["end"] - item["start"] == 1 for item in selected_layout)
    try:
        bone_prompt.BonePromptInject().inject(patcher, graph_clip, [[scene, metadata]], "pose.json",
            json.dumps([{**graph_config[0], "token_indices": [999]}]), 8, 0.1)
    except ValueError as error:
        assert "Graph token index" in str(error)
    else:
        raise AssertionError("Graph accepted an out-of-range attribute token")
    probe_spec = importlib.util.spec_from_file_location("bone_attention_probe",
        Path.cwd().parent / "other_scripts/research/37_bone_attention_probe.py")
    probe = importlib.util.module_from_spec(probe_spec)
    probe_spec.loader.exec_module(probe)
    probe_config = [{**graph_config[index], "token_bias": 2, "token_bias_indices": [1],
                     "start_percent": 0, "end_percent": 1} for index in (2, 4, 13, 14)]
    plain, plain_positive = bone_prompt.BonePromptInject().inject(
        patcher, graph_clip, [[scene, metadata]], "pose.json", json.dumps(probe_config), 8, 0.1)
    shim = SimpleNamespace(NODE_CLASS_MAPPINGS={"BonePromptInject": bone_prompt.BonePromptInject})
    probe_rows = probe.install_probe(shim, lambda rows: None)
    measured, measured_positive = shim.NODE_CLASS_MAPPINGS["BonePromptInject"]().inject(
        model=patcher, clip=graph_clip, positive=[[scene, metadata]], pose_file="pose.json",
        bindings_json=json.dumps(probe_config), samples_per_bone=8, confidence_threshold=0.1)
    plain_sampled = comfy.sample.sample(plain, graph_noise, 2, 4.0,
        "euler", "normal", plain_positive, graph_negative,
        torch.zeros(1, 4, 8, 8), disable_pbar=True, seed=7)
    measured_sampled = comfy.sample.sample(measured, graph_noise, 2, 4.0,
        "euler", "normal", measured_positive, graph_negative,
        torch.zeros(1, 4, 8, 8), disable_pbar=True, seed=7)
    torch.testing.assert_close(measured_sampled, plain_sampled, rtol=0, atol=0)
    assert probe_rows and all(0 <= row["color_attention_after_bias"] <= 1 for row in probe_rows)
'''


if __name__ == "__main__":
    unittest.main()
