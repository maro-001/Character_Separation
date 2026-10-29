import importlib.util
import json
import math
from copy import deepcopy
from pathlib import Path
import sys
import types
import unittest

import torch

from test_bone_prompt import attention


ROOT = Path(__file__).resolve().parents[2] / "custom_nodes/bone_prompt"
package = types.ModuleType("bone_graph_test")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
for name in ("core", "graph"):
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    setattr(package, name, module)
graph = package.graph
core = package.core


class BoneGraphTests(unittest.TestCase):
    def bindings(self):
        return [
            {"person": 0, "bone": "right_upper_arm", "joints": (2, 3),
             "points": ((0.08, 0.25), (0.25, 0.25)), "radius": 0.035,
             "start": 0, "end": 1, "strength": 0.8, "self_attention_strength": 1, "routing_mode": "graph"},
            {"person": 0, "bone": "right_forearm", "joints": (3, 4),
             "points": ((0.25, 0.25), (0.45, 0.25)), "radius": 0.035,
             "start": 1, "end": 2, "strength": 0.8, "self_attention_strength": 1, "routing_mode": "graph"},
            {"person": 1, "bone": "left_forearm", "joints": (6, 7),
             "points": ((0.65, 0.25), (0.9, 0.25)), "radius": 0.035,
             "start": 2, "end": 3, "strength": 0.8, "self_attention_strength": 1, "routing_mode": "graph"},
        ]

    def case(self):
        bindings = self.bindings()
        q = torch.zeros(2, 8 * 16, 4, dtype=torch.float64)
        k = torch.zeros(2, 2 + len(bindings), 4, dtype=torch.float64)
        v = k.clone()
        v[:, :2] = 2
        for index, value in enumerate((6, 10, 20)):
            v[:, index + 2] = value
        options = {"n_heads": 2, "activations_shape": [2, 4, 8, 16], "block": ("output", 0),
                   "bone_prompt": {"base_tokens": 2, "bone_tokens": 3, "bindings": bindings,
                                   "routing_mode": "graph", "route_cache": {}}}
        return q, k, v, options

    def test_partition_covers_every_cell_once_and_is_order_independent(self):
        bindings = self.bindings()
        plan = graph.build_routes(bindings, 8, 16)
        indices = list(plan["background"]) + [index for group in plan["owned"] for index in group]
        self.assertEqual(sorted(indices), list(range(128)))
        self.assertEqual(len(indices), len(set(indices)))
        reversed_plan = graph.build_routes(bindings[::-1], 8, 16)
        for i, binding in enumerate(bindings):
            self.assertEqual(plan["owned"][i], reversed_plan["owned"][2 - i])

    def test_joint_neighbors_share_keys_without_cross_person_keys(self):
        plan = graph.build_routes(self.bindings(), 8, 16)
        self.assertTrue(set(plan["owned"][1]).issubset(plan["allowed"][0]))
        self.assertFalse(set(plan["owned"][2]).intersection(plan["allowed"][0]))
        self.assertFalse(set(plan["owned"][0]).intersection(plan["allowed"][2]))

    def test_joint_falloff_reduces_endpoint_weights_without_changing_ownership(self):
        binding = {"person": 0, "bone": "right_upper_arm", "joints": (2, 3),
                   "points": ((0.2, 0.5), (0.8, 0.5)), "radius": 0.04,
                   "overlap_mode": "nearest"}
        plain = graph.build_routes([binding], 32, 32)
        tapered = graph.build_routes([{**binding, "joint_falloff": 0.25}], 32, 32)
        self.assertEqual(plain["owned"], tapered["owned"])
        indices, weights = plain["spatial"][0]
        tapered_indices, tapered_weights = tapered["spatial"][0]
        self.assertEqual(indices, tapered_indices)
        for index, original, actual in zip(indices, weights, tapered_weights):
            x = (index % 32 + 0.5) / 32
            position = (x - 0.2) / 0.6
            expected_factor = max(0.0, min(1.0, position / 0.25, (1 - position) / 0.25))
            self.assertAlmostEqual(actual, original * expected_factor)
        self.assertTrue(any(actual < original for original, actual in zip(weights, tapered_weights)))
        self.assertTrue(any(actual == original for original, actual in zip(weights, tapered_weights)))

    def test_joint_falloff_defaults_to_zero_and_rejects_invalid_values(self):
        prepared = graph.prepare_graph_bindings(self.bindings(), [{}, {}, {}])
        self.assertTrue(all(binding["joint_falloff"] == 0 for binding in prepared))
        for value in (-0.1, 0.51, float("nan")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "joint_falloff"):
                graph.prepare_graph_bindings(self.bindings(), [{"joint_falloff": value}, {}, {}])

    def test_supersampled_routes_measure_coverage_and_keep_unique_owners(self):
        bindings = self.bindings()
        for binding in bindings:
            binding.update(overlap_mode="nearest", route_samples=4)
        plan = graph.build_routes(bindings, 8, 16)
        reverse = graph.build_routes(bindings[::-1], 8, 16)
        all_owned = [index for group in plan["owned"] for index in group]
        self.assertEqual(len(all_owned), len(set(all_owned)))
        self.assertEqual(len(all_owned) + len(plan["background"]), 128)
        for i, (indices, weights) in enumerate(plan["spatial"]):
            self.assertEqual(set(indices), set(plan["owned"][i]))
            self.assertEqual(plan["owned"][i], reverse["owned"][2 - i])
            self.assertTrue(all(0 < weight <= 1 for weight in weights))
        self.assertTrue(any(0 < weight < 0.5 for _, weights in plan["spatial"] for weight in weights))

    def test_supersampled_routing_responds_to_subcell_position(self):
        binding = {"person": 0, "bone": "right_upper_arm", "joints": (2, 3),
                   "points": ((0.2, 0.25), (0.8, 0.25)), "radius": 0.04,
                   "overlap_mode": "nearest", "route_samples": 8}
        shifted = {**binding, "points": ((0.2, 0.30), (0.8, 0.30))}
        original = graph.build_routes([binding], 8, 16)
        moved = graph.build_routes([shifted], 8, 16)
        self.assertNotEqual(original["spatial"], moved["spatial"])
        self.assertTrue(original["spatial"][0][0])
        self.assertTrue(moved["spatial"][0][0])

    def test_supersampled_cross_attention_scales_change_by_coverage(self):
        binding = {"person": 0, "bone": "right_upper_arm", "joints": (2, 3),
                   "points": ((0.2, 0.25), (0.8, 0.25)), "radius": 0.04,
                   "overlap_mode": "nearest", "route_samples": 4,
                   "start": 0, "end": 1, "strength": 0.5}
        q = torch.zeros(1, 8 * 16, 4, dtype=torch.float32)
        k = torch.zeros(1, 2, 4, dtype=q.dtype)
        v = torch.tensor([[[2.0] * 4, [10.0] * 4]], dtype=q.dtype)
        options = {"n_heads": 2, "activations_shape": [1, 4, 8, 16], "block": ("output", 0),
                   "bone_prompt": {"base_tokens": 1, "bone_tokens": 1, "bindings": [binding],
                                   "route_cache": {}}}
        actual = graph.graph_cross_attention(q, k, v, options, attention)
        expected = torch.full_like(actual, 2.0)
        indices, weights = graph.build_routes([binding], 8, 16)["spatial"][0]
        for index, weight in zip(indices, weights):
            expected[:, index] += weight * 0.5 * 8.0
        torch.testing.assert_close(actual, expected)

    def test_route_samples_preserves_legacy_default_and_validates_configuration(self):
        default = graph.prepare_graph_bindings(self.bindings(), [{}, {}, {}])
        self.assertTrue(all(binding["route_samples"] == 1 for binding in default))
        explicit = graph.prepare_graph_bindings(self.bindings(), [{"route_samples": 1}] * 3)
        self.assertEqual(graph.build_routes(default, 8, 16), graph.build_routes(explicit, 8, 16))
        for value in (0, 9, 2.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "route_samples"):
                graph.prepare_graph_bindings(self.bindings(), [{"route_samples": value}] * 3)
        with self.assertRaisesRegex(ValueError, "nearest"):
            graph.prepare_graph_bindings(self.bindings(), [{"route_samples": 4}] * 3)
        with self.assertRaisesRegex(ValueError, "share"):
            graph.prepare_graph_bindings(self.bindings(), [{"route_samples": 4, "overlap_mode": "nearest"},
                                                          {"route_samples": 2, "overlap_mode": "nearest"},
                                                          {"route_samples": 4, "overlap_mode": "nearest"}])

    def test_cross_attention_is_bounded_and_background_is_unchanged(self):
        q, k, v, options = self.case()
        result = graph.graph_cross_attention(q, k, v, options, attention)
        self.assertEqual(result.dtype, q.dtype)
        self.assertGreaterEqual(result.min().item(), 2)
        self.assertLessEqual(result.max().item(), 20)
        background = graph.build_routes(self.bindings(), 8, 16)["background"]
        torch.testing.assert_close(result[:, list(background)], torch.full((2, len(background), 4), 2.0, dtype=q.dtype))
        self.assertTrue(torch.equal(v[:, :2], torch.full_like(v[:, :2], 2)))

    def test_grouped_contexts_match_separate_contexts_and_reduce_attention_calls(self):
        for dtype in (torch.float32, torch.float64):
            for mode in ("plain", "independent", "target"):
                for overlap in ("blend", "nearest"):
                    with self.subTest(dtype=dtype, mode=mode, overlap=overlap):
                        torch.manual_seed(61)
                        q, _, _, options = self.case()
                        q = torch.randn(q.shape, dtype=dtype)
                        k = torch.randn(2, 6, 4, dtype=dtype)
                        v = torch.randn_like(k)
                        bindings = options["bone_prompt"]["bindings"]
                        options["bone_prompt"]["bone_tokens"] = 4
                        for i, binding in enumerate(bindings):
                            binding.update(start=0, end=2, overlap_mode=overlap,
                                           token_bias=math.log(4 if i < 2 else 8),
                                           token_bias_indices=[0] if i < 2 else [1],
                                           strength=(0.2, 0.65, 0.8)[i],
                                           gate=q.new_tensor([1, 0]).reshape(2, 1, 1))
                            if mode != "plain":
                                binding.update(reference_start=2, reference_end=4,
                                               reference_attention=mode, reference_token_bias=math.log(2),
                                               reference_token_bias_indices=[1], delta_scale=(0.5, 2, 1.2)[i])
                        calls = []

                        def counted_attention(*args, **kwargs):
                            calls.append(args[0].shape[1])
                            return attention(*args, **kwargs)

                        actual = graph.graph_cross_attention(q, k, v, options, counted_attention)
                        grouped_calls = len(calls)
                        split = deepcopy(options)
                        split["bone_prompt"]["route_cache"] = {}
                        split["bone_prompt"]["bone_tokens"] = 12
                        for i, binding in enumerate(split["bone_prompt"]["bindings"]):
                            binding.update(start=4 * i, end=4 * i + 2)
                            if mode != "plain":
                                binding.update(reference_start=4 * i + 2, reference_end=4 * i + 4)
                        split_k = torch.cat([k[:, :2], *[k[:, 2:] for _ in bindings]], dim=1)
                        split_v = torch.cat([v[:, :2], *[v[:, 2:] for _ in bindings]], dim=1)
                        calls.clear()
                        expected = graph.graph_cross_attention(q, split_k, split_v, split, counted_attention)
                        torch.testing.assert_close(actual, expected)
                        self.assertEqual(grouped_calls, 4 if mode == "independent" else 3)
                        self.assertEqual(len(calls), 7 if mode == "independent" else 4)
                        torch.testing.assert_close(actual[1], attention(q, k[:, :2], v[:, :2], 2)[1], rtol=0, atol=0)

    def test_foreign_context_cannot_change_another_person_bone(self):
        q, k, v, options = self.case()
        expected = graph.graph_cross_attention(q, k, v, options, attention)
        changed = v.clone()
        changed[:, 4] = 1000
        actual = graph.graph_cross_attention(q, k, changed, options, attention)
        plan = graph.build_routes(self.bindings(), 8, 16)
        indices = plan["owned"][0] + plan["owned"][1]
        torch.testing.assert_close(actual[:, indices], expected[:, indices], rtol=0, atol=0)

    def test_overlapping_people_are_assigned_to_one_person(self):
        bindings = self.bindings()
        bindings[2]["points"] = bindings[0]["points"]
        plan = graph.build_routes(bindings, 8, 16)
        self.assertFalse(set(plan["spatial"][0][0]).intersection(plan["spatial"][2][0]))

    def test_nearest_overlap_routes_each_cell_to_its_owning_bone(self):
        bindings = self.bindings()
        blended = graph.build_routes(bindings, 8, 16)
        self.assertTrue(set(blended["spatial"][0][0]).intersection(blended["spatial"][1][0]))
        for binding in bindings:
            binding["overlap_mode"] = "nearest"
        plan = graph.build_routes(bindings, 8, 16)
        reversed_plan = graph.build_routes(bindings[::-1], 8, 16)
        self.assertEqual(plan["background"], blended["background"])
        self.assertEqual(plan["owned"], blended["owned"])
        for index, (indices, weights) in enumerate(plan["spatial"]):
            self.assertEqual(list(indices), plan["owned"][index])
            self.assertEqual(plan["spatial"][index], reversed_plan["spatial"][2 - index])
            original_weights = dict(zip(*blended["spatial"][index]))
            self.assertEqual(weights, tuple(original_weights[cell] for cell in indices))

    def test_nearest_overlap_blocks_direct_context_leak_from_same_person_bone(self):
        for mode in ("blend", "nearest"):
            with self.subTest(mode=mode):
                q, k, v, options = self.case()
                bindings = options["bone_prompt"]["bindings"]
                for binding in bindings:
                    binding["overlap_mode"] = mode
                expected = graph.graph_cross_attention(q, k, v, options, attention)
                changed = v.clone()
                changed[:, 3] = 1000
                actual = graph.graph_cross_attention(q, k, changed, options, attention)
                protected = graph.build_routes(bindings, 8, 16)["owned"][0]
                if mode == "nearest":
                    torch.testing.assert_close(actual[:, protected], expected[:, protected], rtol=0, atol=0)
                else:
                    self.assertFalse(torch.equal(actual[:, protected], expected[:, protected]))

    def test_nearest_overlap_ignores_zero_strength_competitors(self):
        q, k, v, options = self.case()
        bindings = options["bone_prompt"]["bindings"]
        for binding in bindings:
            binding["overlap_mode"] = "nearest"
        bindings[1]["strength"] = 0
        actual = graph.graph_cross_attention(q, k, v, options, attention)
        bindings[1]["points"] = ((0.7, 0.9), (0.9, 0.9))
        options["bone_prompt"]["route_cache"] = {}
        expected = graph.graph_cross_attention(q, k, v, options, attention)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_nearest_overlap_preserves_coverage_for_existing_ten_poses(self):
        research = ROOT.parents[2] / "other_scripts/research"
        workflow = json.loads((research / "workflows/06_pony_v6_openpose_quality_refined.api.json").read_text())
        config = json.loads(workflow["18"]["inputs"]["bindings_json"])
        for source in config:
            source["overlap_mode"] = "blend"
        for pose_path in sorted((research / "assets/14_pose100").glob("*.json")):
            pose = json.loads(pose_path.read_text())
            for size in (32, 64, 128):
                with self.subTest(pose=pose_path.name, grid=size):
                    bindings = graph.prepare_graph_bindings(core.resolve_bindings(pose, config, 16, 0.1), config)
                    blended = graph.build_routes(bindings, size, size)
                    for binding in bindings:
                        binding["overlap_mode"] = "nearest"
                    nearest = graph.build_routes(bindings, size, size)
                    self.assertEqual(nearest["background"], blended["background"])
                    self.assertEqual(nearest["owned"], blended["owned"])
                    for before, (indices, _) in zip(blended["spatial"], nearest["spatial"]):
                        if before[0]:
                            self.assertTrue(indices)
                    routed = [index for indices, _ in nearest["spatial"] for index in indices]
                    self.assertEqual(len(routed), len(set(routed)))
                    self.assertEqual(len(routed) + len(nearest["background"]), size * size)

    def test_token_bias_uses_softmax_prior_and_keeps_query_dependence(self):
        q, k, v, options = self.case()
        binding = options["bone_prompt"]["bindings"][0]
        options["bone_prompt"]["bindings"] = [binding]
        options["bone_prompt"]["bone_tokens"] = 2
        binding["end"] = 2
        k, v = k[:, :4].clone(), v[:, :4].clone()
        v[:, 2], v[:, 3] = 6, 0
        unbiased = graph.graph_cross_attention(q, k, v, options, attention)
        binding.update(token_bias=0, token_bias_indices=[0])
        torch.testing.assert_close(graph.graph_cross_attention(q, k, v, options, attention),
                                   unbiased, rtol=0, atol=0)
        binding["token_bias"] = math.log(4)
        biased = graph.graph_cross_attention(q, k, v, options, attention)
        # Native scene=2, unbiased local=3, biased local=(4*6+0)/5=4.8.
        torch.testing.assert_close(biased, 2 + 2.8 * (unbiased - 2))
        self.assertGreater(biased.max().item(), unbiased.max().item())
        self.assertGreaterEqual(biased.min().item(), 2)
        self.assertLessEqual(biased.max().item(), 6)
        k[:, 2], k[:, 3] = 1, -1
        varied = graph.graph_cross_attention(torch.ones_like(q), k, v, options, attention)
        self.assertFalse(torch.equal(varied, biased))
        background = list(graph.build_routes([binding], 8, 16)["background"])
        torch.testing.assert_close(biased[:, background], unbiased[:, background], rtol=0, atol=0)

    def test_full_local_self_attention_blocks_foreign_bone_copy(self):
        q, _, _, options = self.case()
        k, v = torch.zeros_like(q), torch.zeros_like(q)
        plan = graph.build_routes(self.bindings(), 8, 16)
        v[:, plan["owned"][2]] = 100
        result = graph.graph_self_attention(q, k, v, options, attention)
        protected = plan["owned"][0] + plan["owned"][1]
        torch.testing.assert_close(result[:, protected], torch.zeros_like(result[:, protected]), rtol=0, atol=0)
        self.assertGreater(result[:, plan["owned"][2]].max().item(), 0)

    def test_reference_contrast_calibrates_magnitude_and_preserves_native_scene(self):
        q, k, v, options = self.case()
        bindings = options["bone_prompt"]["bindings"]
        for binding in bindings:
            binding.update(overlap_mode="nearest", reference_start=2, reference_end=3,
                           reference_token_bias=0, reference_token_bias_indices=[], delta_scale=0.2)
        v[:, 2], v[:, 3], v[:, 4] = 4.1, 5, 4
        actual = graph.graph_cross_attention(q, k, v, options, attention)
        expected = q.new_full(q.shape, 2)
        routes = graph.build_routes(bindings, 8, 16)
        for binding, (indices, weights) in zip(bindings[:2], routes["spatial"][:2]):
            expected[:, list(indices)] += 0.4 * binding["strength"] * q.new_tensor(weights).reshape(1, -1, 1)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertLessEqual((actual - 2).abs().max().item(), 0.2 * 2 * 0.8)

    def test_reference_contrast_zero_difference_zero_scale_and_zero_scene_are_finite(self):
        for mode in ("same", "zero_scale", "zero_scene"):
            with self.subTest(mode=mode):
                q, k, v, options = self.case()
                for binding in options["bone_prompt"]["bindings"]:
                    binding.update(reference_start=2, reference_end=3, reference_token_bias=0,
                                   reference_token_bias_indices=[], delta_scale=0 if mode == "zero_scale" else 0.5)
                if mode == "same":
                    v[:, 2:] = 4
                if mode == "zero_scene":
                    v[:, :2] = 0
                actual = graph.graph_cross_attention(q, k, v, options, attention)
                expected = torch.zeros_like(actual) if mode == "zero_scene" else torch.full_like(actual, 2)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_reference_contrast_caps_gain_for_tiny_differences(self):
        q, k, v, options = self.case()
        for binding in options["bone_prompt"]["bindings"]:
            binding.update(reference_start=2, reference_end=3, reference_token_bias=0,
                           reference_token_bias_indices=[], delta_scale=1)
        v[:, 2:4], v[:, 4] = 4.001, 4
        actual = graph.graph_cross_attention(q, k, v, options, attention)
        self.assertGreater((actual - 2).max().item(), 0)
        self.assertLessEqual((actual - 2).abs().max().item(), 4 * 0.001 * 0.8 + 1e-12)

    def test_target_reference_attention_ignores_reference_keys_but_responds_to_target_queries(self):
        q, k, v, options = self.case()
        binding = options["bone_prompt"]["bindings"][0]
        options["bone_prompt"].update(bindings=[binding], bone_tokens=4)
        binding.update(start=0, end=2, reference_start=2, reference_end=4,
                       reference_attention="target", reference_token_bias=0,
                       reference_token_bias_indices=[], delta_scale=10)
        k = q.new_zeros((2, 6, 4))
        v = k.clone()
        v[:, :2] = 2
        v[:, 2], v[:, 3], v[:, 4], v[:, 5] = 4, 6, 3, 7
        # Equal target weights cancel the opposite +1/-1 value differences,
        # even when reference keys would strongly prefer the first token.
        k[:, 2], k[:, 3], k[:, 4], k[:, 5] = 1, -1, 100, -100
        zero_query = graph.graph_cross_attention(q, k, v, options, attention)
        torch.testing.assert_close(zero_query, torch.full_like(q, 2), rtol=0, atol=0)
        actual = graph.graph_cross_attention(torch.ones_like(q), k, v, options, attention)
        changed = k.clone()
        changed[:, 4:] *= -1
        torch.testing.assert_close(graph.graph_cross_attention(torch.ones_like(q), changed, v, options, attention),
                                   actual, rtol=0, atol=0)
        self.assertGreater(actual.max().item(), 2)
        self.assertLessEqual((actual - 2).abs().max().item(), 4 * binding["strength"])
        torch.testing.assert_close(graph.graph_cross_attention(-torch.ones_like(q), k, v, options, attention),
                                   4 - actual)
        binding["reference_attention"] = "independent"
        independent = graph.graph_cross_attention(torch.ones_like(q), k, v, options, attention)
        self.assertFalse(torch.equal(actual, independent))
        binding["reference_attention"] = "target"
        v[:, 4:] = v[:, 2:4]
        torch.testing.assert_close(graph.graph_cross_attention(torch.ones_like(q), k, v, options, attention),
                                   torch.full_like(q, 2), rtol=0, atol=0)

    def test_soft_self_attention_preserves_background_and_blends_with_native(self):
        q, _, _, options = self.case()
        k, v = torch.randn_like(q), torch.randn_like(q)
        native = attention(q, k, v, 2)
        routed = graph.graph_self_attention(q, k, v, options, attention)
        for binding in options["bone_prompt"]["bindings"]:
            binding["self_attention_strength"] = 0.125
        actual = graph.graph_self_attention(q, k, v, options, attention)
        torch.testing.assert_close(actual, native + 0.125 * (routed - native))
        background = list(graph.build_routes(self.bindings(), 8, 16)["background"])
        torch.testing.assert_close(actual[:, background], native[:, background], rtol=0, atol=0)
        for binding in options["bone_prompt"]["bindings"]:
            binding["self_attention_strength"] = 0
        torch.testing.assert_close(graph.graph_self_attention(q, k, v, options, attention), native, rtol=0, atol=0)

    def test_partition_self_attention_blocks_copy_between_every_bone_and_background(self):
        q, _, _, options = self.case()
        bindings = options["bone_prompt"]["bindings"]
        for binding in bindings:
            binding["self_attention_scope"] = "partition"
        plan = graph.build_routes(bindings, 8, 16)
        k, v = torch.zeros_like(q), torch.full_like(q, 3)
        for indices, value in zip(plan["owned"], (7, 11, 19)):
            v[:, indices] = value
        expected = graph.graph_self_attention(q, k, v, options, attention)
        torch.testing.assert_close(expected, v, rtol=0, atol=1e-12)
        changed = v.clone()
        changed[:, plan["owned"][0]] += 100
        actual = graph.graph_self_attention(q, k, changed, options, attention)
        protected = plan["owned"][1] + plan["owned"][2] + list(plan["background"])
        torch.testing.assert_close(actual[:, protected], expected[:, protected], rtol=0, atol=1e-12)
        torch.testing.assert_close(actual[:, plan["owned"][0]], expected[:, plan["owned"][0]] + 100)

    def test_soft_partition_self_attention_blends_background_and_respects_batch_gate(self):
        q, _, _, options = self.case()
        bindings = options["bone_prompt"]["bindings"]
        for binding in bindings:
            binding["self_attention_scope"] = "partition"
        plan = graph.build_routes(bindings, 8, 16)
        k, v = torch.zeros_like(q), torch.zeros_like(q)
        v[:, plan["owned"][0]] = 100
        native = attention(q, k, v, 2)
        partition = graph.graph_self_attention(q, k, v, options, attention)
        for binding in bindings:
            binding.update(self_attention_strength=0.25, gate=torch.tensor([False, True]).reshape(2, 1, 1))
        actual = graph.graph_self_attention(q, k, v, options, attention)
        torch.testing.assert_close(actual[0], native[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], native[1] + 0.25 * (partition[1] - native[1]))
        background = list(plan["background"])
        torch.testing.assert_close(actual[1, background], 0.75 * native[1, background])
        for binding in bindings:
            binding["self_attention_strength"] = 0
        torch.testing.assert_close(graph.graph_self_attention(q, k, v, options, attention), native, rtol=0, atol=0)

    def test_soft_self_attention_accepts_half_precision_with_batch_gate(self):
        q, _, _, options = self.case()
        q = q.to(torch.float16)
        k, v = torch.zeros_like(q), torch.randn_like(q)
        for binding in options["bone_prompt"]["bindings"]:
            binding.update(self_attention_scope="partition", self_attention_strength=0.05,
                           gate=torch.tensor([False, True]).reshape(2, 1, 1))
        result = graph.graph_self_attention(q, k, v, options, attention)
        self.assertEqual(result.dtype, q.dtype)
        native = attention(q, k, v, 2)
        torch.testing.assert_close(result[0], native[0], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(result).all())

    def test_inactive_batch_gate_matches_native_attention(self):
        q, k, v, options = self.case()
        for binding in options["bone_prompt"]["bindings"]:
            binding["gate"] = torch.tensor([False, True]).reshape(2, 1, 1)
        cross = graph.graph_cross_attention(q, k, v, options, attention)
        torch.testing.assert_close(cross[0], attention(q, k[:, :2], v[:, :2], 2)[0], rtol=0, atol=0)
        sk, sv = torch.randn_like(q), torch.randn_like(q)
        self_out = graph.graph_self_attention(q, sk, sv, options, attention)
        torch.testing.assert_close(self_out[0], attention(q, sk, sv, 2)[0], rtol=0, atol=0)

    def test_zero_strength_and_excluded_blocks_use_native_attention(self):
        for mode in ("zero", "block", "grid"):
            q, k, v, options = self.case()
            for binding in options["bone_prompt"]["bindings"]:
                if mode == "zero":
                    binding["strength"] = 0
                elif mode == "block":
                    binding["block_types"] = ["input"]
                else:
                    binding["min_grid_size"] = 16
            torch.testing.assert_close(graph.graph_cross_attention(q, k, v, options, attention),
                                       attention(q, k[:, :2], v[:, :2], 2), rtol=0, atol=0)
            sk, sv = torch.zeros_like(q), torch.randn_like(q)
            torch.testing.assert_close(graph.graph_self_attention(q, sk, sv, options, attention),
                                       attention(q, sk, sv, 2), rtol=0, atol=0)

    def test_graph_config_rejects_ambiguous_or_incompatible_settings(self):
        bindings = self.bindings()
        config = [{"radius": 0.02} for _ in bindings]
        prepared = graph.prepare_graph_bindings(bindings, config)
        self.assertTrue(all(binding["radius"] == 0.02 for binding in prepared))
        self.assertTrue(all(binding["overlap_mode"] == "blend" for binding in prepared))
        nearest = graph.prepare_graph_bindings(self.bindings(), [{"overlap_mode": "nearest"}] * 3)
        self.assertTrue(all(binding["overlap_mode"] == "nearest" for binding in nearest))
        for fields in ({"radius": 0}, {"radius": float("nan")}, {"reference_text": "neutral"},
                       {"attribute_strength": 1}, {"query_mode": "bone"},
                       {"self_attention_strength": -0.1}, {"self_attention_strength": float("nan")},
                       {"self_attention_scope": "unknown"},
                       {"overlap_mode": "unknown"}, {"token_bias": float("nan")},
                       {"token_bias": 2}, {"token_bias_indices": [1, 1]},
                       {"token_bias_indices": [-1]}, {"token_bias_indices": [True]},
                       {"reference_binding": [9, "right_torso"]}, {"reference_binding": []},
                       {"delta_scale": 0.5},
                       {"reference_attention": "target"},
                       {"reference_binding": [0, "right_forearm"], "reference_attention": "unknown"},
                       {"reference_binding": [0, "right_forearm"], "delta_scale": -1},
                       {"reference_binding": [0, "right_forearm"], "delta_scale": float("nan")}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                graph.prepare_graph_bindings(self.bindings(), [fields, {}, {}])
        for strength in (-0.1, 1.1):
            bindings = self.bindings()
            bindings[0]["strength"] = strength
            with self.assertRaisesRegex(ValueError, "blend fraction"):
                graph.prepare_graph_bindings(bindings, [{}, {}, {}])
        bindings = self.bindings()
        bindings[1]["start_percent"] = 0.5
        with self.assertRaisesRegex(ValueError, "share"):
            graph.prepare_graph_bindings(bindings, [{}, {}, {}])
        with self.assertRaisesRegex(ValueError, "share"):
            graph.prepare_graph_bindings(self.bindings(), [{"overlap_mode": "nearest"}, {}, {}])
        with self.assertRaisesRegex(ValueError, "share"):
            graph.prepare_graph_bindings(self.bindings(), [{"self_attention_scope": "partition"}, {}, {}])
        bindings = self.bindings()
        with self.assertRaisesRegex(ValueError, "unique"):
            graph.prepare_graph_bindings([bindings[0], bindings[0]], [{}, {}])

    def test_mode_selection_rejects_mixed_algorithms(self):
        pose = {"canvas_width": 1024, "canvas_height": 1024, "people": []}
        for modes in (("graph", "delta"), ("unknown",), ([],)):
            config = [{"person": 0, "bone": "right_forearm", "text": "red", "routing_mode": mode} for mode in modes]
            with self.subTest(modes=modes), self.assertRaisesRegex(ValueError, "routing_mode"):
                core.resolve_bindings(pose, config, 8, 0.1)


if __name__ == "__main__":
    unittest.main()
