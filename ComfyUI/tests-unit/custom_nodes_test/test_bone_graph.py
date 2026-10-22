import importlib.util
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

    def test_cross_attention_is_bounded_and_background_is_unchanged(self):
        q, k, v, options = self.case()
        result = graph.graph_cross_attention(q, k, v, options, attention)
        self.assertEqual(result.dtype, q.dtype)
        self.assertGreaterEqual(result.min().item(), 2)
        self.assertLessEqual(result.max().item(), 20)
        background = graph.build_routes(self.bindings(), 8, 16)["background"]
        torch.testing.assert_close(result[:, list(background)], torch.full((2, len(background), 4), 2.0, dtype=q.dtype))
        self.assertTrue(torch.equal(v[:, :2], torch.full_like(v[:, :2], 2)))

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

    def test_full_local_self_attention_blocks_foreign_bone_copy(self):
        q, _, _, options = self.case()
        k, v = torch.zeros_like(q), torch.zeros_like(q)
        plan = graph.build_routes(self.bindings(), 8, 16)
        v[:, plan["owned"][2]] = 100
        result = graph.graph_self_attention(q, k, v, options, attention)
        protected = plan["owned"][0] + plan["owned"][1]
        torch.testing.assert_close(result[:, protected], torch.zeros_like(result[:, protected]), rtol=0, atol=0)
        self.assertGreater(result[:, plan["owned"][2]].max().item(), 0)

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
        for fields in ({"radius": 0}, {"radius": float("nan")}, {"reference_text": "neutral"},
                       {"attribute_strength": 1}, {"query_mode": "bone"},
                       {"self_attention_strength": -0.1}, {"self_attention_strength": float("nan")}):
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
