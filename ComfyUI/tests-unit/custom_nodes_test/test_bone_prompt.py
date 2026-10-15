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

    result = F.scaled_dot_product_attention(split(q), split(k), split(v))
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

    def test_native_comfy_attention_and_model_patcher_integration(self):
        # Isolate ComfyUI's CPU startup policy from other pytest modules.
        result = subprocess.run([sys.executable, "-c", NATIVE_CHECK], cwd=COMFY_ROOT,
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


NATIVE_CHECK = r'''
import json
from pathlib import Path
from tempfile import TemporaryDirectory
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
'''


if __name__ == "__main__":
    unittest.main()
