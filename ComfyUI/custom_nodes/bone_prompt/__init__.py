import hashlib
import json
from pathlib import Path

import torch

import comfy.conds
from comfy.ldm.modules.attention import optimized_attention
from comfy.patcher_extension import WrappersMP
import folder_paths

from .core import inject_attention, isolate_self_attention, resolve_bindings
from .graph import graph_cross_attention, graph_self_attention, prepare_graph_bindings


LAYOUT_KEY = "bone_prompt_layout"
EMBEDDING_KEY = "bone_prompt_embeddings"
DEFAULT_BINDINGS = json.dumps([
    {"person": 0, "bone": "right_forearm", "text": "red steel gauntlet", "strength": 0.5},
    {"person": 1, "bone": "left_forearm", "text": "blue steel gauntlet", "strength": 0.5},
], indent=2)


def pose_path(pose_file):
    root = Path(folder_paths.get_input_directory()).resolve()
    path = (root / pose_file).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() != ".json":
        raise ValueError("Pose file must be JSON inside ComfyUI/input.")
    return path


def sparse_cross_attention(q, k, v, extra_options):
    if extra_options.get("bone_prompt", {}).get("routing_mode") == "graph":
        return graph_cross_attention(q, k, v, extra_options, optimized_attention)
    return inject_attention(q, k, v, extra_options, optimized_attention)


def sparse_self_attention(q, k, v, extra_options):
    if extra_options.get("bone_prompt", {}).get("routing_mode") == "graph":
        return graph_self_attention(q, k, v, extra_options, optimized_attention)
    return isolate_self_attention(q, k, v, extra_options, optimized_attention)


def _align_reference_and_detect_attributes(t_tokens, r_tokens, e_target, e_ref):
    if not (isinstance(t_tokens, dict) and isinstance(r_tokens, dict) and "l" in t_tokens and "l" in r_tokens):
        return e_ref, None
    if e_target.shape[1] != e_ref.shape[1] or e_target.shape[1] < 4:
        return e_ref, None
    t_ids = [tok[0] for tok in t_tokens["l"][0]]
    r_ids = [tok[0] for tok in r_tokens["l"][0]]
    seq_len = min(len(t_ids), len(r_ids), e_target.shape[1])
    eos_id = 49407
    t_len = t_ids.index(eos_id) if eos_id in t_ids else seq_len
    r_len = r_ids.index(eos_id) if eos_id in r_ids else seq_len
    aligned = e_ref.clone()
    aligned_r_ids = list(r_ids[:seq_len])
    if t_len != r_len:
        # Find the last shared anchor token (e.g. comma) before the length shift
        prefix_anchor = 0
        for idx_i in range(min(t_len, r_len)):
            if t_ids[idx_i] == r_ids[idx_i]:
                prefix_anchor = idx_i + 1
            elif idx_i + 1 < min(t_len, r_len) and t_ids[idx_i + 1] == r_ids[idx_i + 1]:
                continue
            else:
                break
        shift = r_len - t_len
        if shift > 0 and prefix_anchor < t_len:
            aligned[:, prefix_anchor] = e_ref[:, prefix_anchor:prefix_anchor + 1 + shift].mean(dim=1)
            count = min(seq_len - (prefix_anchor + 1), seq_len - (prefix_anchor + 1 + shift))
            aligned[:, prefix_anchor + 1:prefix_anchor + 1 + count] = (
                e_ref[:, prefix_anchor + 1 + shift:prefix_anchor + 1 + shift + count]
            )
            for offset_i in range(count):
                aligned_r_ids[prefix_anchor + 1 + offset_i] = r_ids[prefix_anchor + 1 + shift + offset_i]
        elif shift < 0 and prefix_anchor < r_len:
            count = min(seq_len - (prefix_anchor - shift), seq_len - prefix_anchor)
            aligned[:, prefix_anchor - shift:prefix_anchor - shift + count] = (
                e_ref[:, prefix_anchor:prefix_anchor + count]
            )
            for offset_i in range(count):
                aligned_r_ids[prefix_anchor - shift + offset_i] = r_ids[prefix_anchor + offset_i]
    if t_len + 1 < seq_len:
        aligned[:, t_len + 1:] = e_target[:, t_len + 1:]
    diff_positions = [i for i in range(1, min(t_len, seq_len)) if t_ids[i] != aligned_r_ids[i]]
    if not diff_positions:
        return aligned, None
    counts = {}
    for pos in diff_positions:
        counts[t_ids[pos]] = counts.get(t_ids[pos], 0) + 1
    max_count = max(counts.values())
    if max_count > 1:
        attr_indices = [pos for pos in diff_positions if counts[t_ids[pos]] == max_count]
    else:
        attr_indices = diff_positions
    return aligned, attr_indices


def apply_bone_context(executor, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
    layout = kwargs.pop(LAYOUT_KEY, None)
    embeddings = kwargs.pop(EMBEDDING_KEY, None)
    if layout is not None:
        transformer_options = transformer_options.copy()
        bindings = []
        for binding in layout:
            item = binding.copy()
            if "start_sigma" in binding:
                item["gate"] = ((t <= binding["start_sigma"]) & (t >= binding["end_sigma"])).reshape(-1, 1, 1)
            bindings.append(item)
        transformer_options["bone_prompt"] = {
            "base_tokens": c_crossattn.shape[1],
            "bone_tokens": embeddings.shape[1],
            "bindings": bindings,
        }
        if layout[0].get("routing_mode") == "graph":
            transformer_options["bone_prompt"].update(routing_mode="graph", route_cache={})
        c_crossattn = torch.cat((c_crossattn, embeddings.to(c_crossattn)), dim=1)
    return executor(x, t, c_concat=c_concat, c_crossattn=c_crossattn, control=control,
                    transformer_options=transformer_options, **kwargs)


def encode_bindings(clip, conditioning, bindings):
    states = [(embedding, metadata.copy(), [], []) for embedding, metadata in conditioning]
    for binding in bindings:
        t_tokens = clip.tokenize(binding["text"])
        r_tokens = clip.tokenize(binding["reference_text"]) if "reference_text" in binding else None
        for text_key in ("text", "reference_text"):
            if text_key not in binding:
                continue
            tokens = t_tokens if text_key == "text" else r_tokens
            encoded = clip.encode_from_tokens_scheduled(tokens)
            indices = binding.get("token_indices" if text_key == "text" else "reference_token_indices")
            if indices is not None:
                selected = []
                for embedding, schedule in encoded:
                    if max(indices) >= embedding.shape[1]:
                        raise ValueError(f"Bone {text_key} token index exceeds encoded length {embedding.shape[1]}.")
                    selected.append((embedding[:, indices], schedule))
                encoded = selected
            next_states = []
            for base, metadata, chunks, layout in states:
                for embedding, schedule in encoded:
                    start = max(metadata.get("start_percent", 0.0), schedule.get("start_percent", 0.0))
                    end = min(metadata.get("end_percent", 1.0), schedule.get("end_percent", 1.0))
                    if start >= end:
                        continue
                    if embedding.shape[-1] != base.shape[-1]:
                        raise ValueError("Bone CLIP and scene conditioning must use the same text encoder.")
                    meta = metadata.copy()
                    if "start_percent" in metadata or "start_percent" in schedule:
                        meta["start_percent"] = start
                    if "end_percent" in metadata or "end_percent" in schedule:
                        meta["end_percent"] = end
                    offset = sum(chunk.shape[1] for chunk in chunks)
                    chunk_emb = embedding
                    if text_key == "text":
                        next_layout = layout + [{
                            "person": binding["person"], "bone": binding["bone"],
                            "points": binding["points"], "strength": binding["strength"],
                            "start": offset, "end": offset + chunk_emb.shape[1],
                        }]
                        next_layout[-1].update({
                            key: binding[key]
                            for key in (
                                "min_grid_size",
                                "start_sigma",
                                "end_sigma",
                                "competition_strength",
                                "query_mode",
                                "block_types",
                                "tube_points",
                                "unbound_points",
                                "attribute_strength",
                            )
                            if key in binding
                        })
                    else:
                        attr_indices = None
                        if binding.get("token_indices") is None and binding.get("reference_token_indices") is None and chunks:
                            chunk_emb, attr_indices = _align_reference_and_detect_attributes(
                                t_tokens, r_tokens, chunks[-1], embedding
                            )
                        updated_item = {
                            **layout[-1],
                            "reference_start": offset,
                            "reference_end": offset + chunk_emb.shape[1],
                        }
                        if attr_indices:
                            updated_item["attribute_indices"] = attr_indices
                        next_layout = layout[:-1] + [updated_item]
                    next_states.append((base, meta, chunks + [chunk_emb], next_layout))
            states = next_states
        if "opposite_points" in binding:
            for _, _, _, layout in states:
                layout.append({**layout[-1], "bone": binding["opposite_bone"], "points": binding["opposite_points"],
                               "strength": -binding["opposite_strength"], "competition_strength": 0})
    result = []
    for base, metadata, chunks, layout in states:
        model_conds = metadata.get("model_conds", {}).copy()
        # Keep text tensors in normal conditioning, never on the model patcher.
        model_conds[EMBEDDING_KEY] = comfy.conds.CONDRegular(torch.cat([
            chunk.to(base).expand(base.shape[0], -1, -1) for chunk in chunks
        ], dim=1))
        model_conds[LAYOUT_KEY] = comfy.conds.CONDConstant(layout)
        metadata["model_conds"] = model_conds
        result.append([base, metadata])
    return result


def encode_graph_bindings(clip, conditioning, bindings):
    states = [(embedding, metadata.copy(), [], {}) for embedding, metadata in conditioning]
    contexts = [(binding["text"], tuple(binding["token_indices"]) if "token_indices" in binding else None)
                for binding in bindings]
    by_identity = {(binding["person"], binding["bone"]): (binding, context)
                   for binding, context in zip(bindings, contexts)}
    for text, indices in dict.fromkeys(contexts):
        encoded = clip.encode_from_tokens_scheduled(clip.tokenize(text))
        if indices is not None:
            selected = []
            for embedding, schedule in encoded:
                if max(indices) >= embedding.shape[1]:
                    raise ValueError(f"Graph token index exceeds encoded length {embedding.shape[1]}.")
                selected.append((embedding[:, list(indices)], schedule))
            encoded = selected
        next_states = []
        for base, metadata, chunks, spans in states:
            for embedding, schedule in encoded:
                start = max(metadata.get("start_percent", 0), schedule.get("start_percent", 0))
                end = min(metadata.get("end_percent", 1), schedule.get("end_percent", 1))
                if start >= end:
                    continue
                if embedding.shape[-1] != base.shape[-1]:
                    raise ValueError("Bone CLIP and scene conditioning must use the same text encoder.")
                meta = metadata.copy()
                if "start_percent" in metadata or "start_percent" in schedule:
                    meta["start_percent"] = start
                if "end_percent" in metadata or "end_percent" in schedule:
                    meta["end_percent"] = end
                offset = sum(chunk.shape[1] for chunk in chunks)
                next_states.append((base, meta, chunks + [embedding],
                                    {**spans, (text, indices): (offset, offset + embedding.shape[1])}))
        states = next_states
    result = []
    for base, metadata, chunks, spans in states:
        layout = []
        for binding, context in zip(bindings, contexts):
            item = {key: binding[key] for key in (
                "person", "bone", "points", "strength", "radius", "joint_falloff", "route_samples", "joints", "routing_mode",
                "min_grid_size", "block_types", "start_sigma", "end_sigma", "self_attention_strength",
                "self_attention_scope",
                "overlap_mode",
                "token_bias", "token_bias_indices",
                "delta_scale",
                "reference_attention",
            ) if key in binding}
            item["start"], item["end"] = spans[context]
            if "token_bias_indices" in item and max(item["token_bias_indices"]) >= item["end"] - item["start"]:
                raise ValueError("Graph token_bias_indices exceed the attended context length after token selection.")
            if "reference_binding" in binding:
                reference, ref_context = by_identity[binding["reference_binding"]]
                item["reference_start"], item["reference_end"] = spans[ref_context]
                item["reference_token_bias"] = reference.get("token_bias", 0)
                item["reference_token_bias_indices"] = reference.get("token_bias_indices", []).copy()
                if (item["reference_attention"] == "target"
                        and item["end"] - item["start"] != item["reference_end"] - item["reference_start"]):
                    raise ValueError("Graph target reference_attention requires equal target/reference context lengths.")
            layout.append(item)
        model_conds = metadata.get("model_conds", {}).copy()
        model_conds[EMBEDDING_KEY] = comfy.conds.CONDRegular(torch.cat([
            chunk.to(base).expand(base.shape[0], -1, -1) for chunk in chunks
        ], dim=1))
        model_conds[LAYOUT_KEY] = comfy.conds.CONDConstant(layout)
        metadata["model_conds"] = model_conds
        result.append([base, metadata])
    return result


class BonePromptInject:
    @classmethod
    def INPUT_TYPES(cls):
        root = Path(folder_paths.get_input_directory())
        files = sorted(str(p.relative_to(root)) for p in root.rglob("*")
                       if p.is_file() and p.suffix.lower() == ".json")
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "positive": ("CONDITIONING",),
            "pose_file": (files,),
            "bindings_json": ("STRING", {"multiline": True, "default": DEFAULT_BINDINGS}),
            "samples_per_bone": ("INT", {"default": 8, "min": 2, "max": 256}),
            "confidence_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "positive")
    FUNCTION = "inject"
    CATEGORY = "model/patch/unet"
    DESCRIPTION = "Experimental training-free SDXL bone conditioning. routing_mode=graph (graph_v3) directly blends each bone's text attention at local queries and normalizes overlaps. Identical contexts batch local queries (grouped_contexts_v1), preserving per-bone weights and gates. overlap_mode=nearest routes each covered cell to its closest bone; default blend mixes same-person bones in overlaps. Optional token_indices select contextual attribute tokens without changing the text. token_bias adds an attention-logit bias to token_bias_indices in the attended context, preserving the other tokens. Optional reference_binding=[person,bone] injects a contrast against an existing binding; delta_scale (default0.5) limits contrast RMS relative to scene attention, with gain capped at4. Optional reference_attention=target shares target attention weights for token-aligned value contrasts and requires equal sequence lengths; default independent uses separately normalized target/reference attention. Self-attention defaults to joint neighbors with native background; self_attention_scope=partition softly mixes native attention with independent bone/background groups (self_dtype_fix_v1). strength and self_attention_strength (default 0.12) are fractions 0..1; radius sets normalized coverage; joint_falloff optionally tapers cross-attention near bone endpoints (joint_falloff_v1); route_samples optionally estimates subcell coverage for nearest routing (subcell_routes_v1). Graph bindings share schedule, block selection, self-attention strength and overlap_mode. Default delta retains target/reference feature injection. Use both outputs in the base sampling path."

    def inject(self, model, clip, positive, pose_file, bindings_json, samples_per_bone, confidence_threshold):
        config = model.get_model_object("model_config").unet_config
        if config.get("context_dim") != 2048 or config.get("use_temporal_attention", False):
            raise ValueError("BonePromptInject currently supports SDXL/Pony 2D U-Net models.")
        patches = model.model_options.get("transformer_options", {}).get("patches_replace", {})
        if patches.get("attn2"):
            raise ValueError("BonePromptInject cannot replace an existing attn2 replacement. Use a separate model branch.")
        for _, metadata in positive:
            if "mask" in metadata or "area" in metadata:
                raise ValueError("BonePromptInject uses full-frame coordinates; masked or cropped scene conditioning is unsupported.")
            if LAYOUT_KEY in metadata.get("model_conds", {}):
                raise ValueError("Add all bone bindings to one BonePromptInject node.")
        with pose_path(pose_file).open(encoding="utf-8") as source:
            pose = json.load(source)
        config_bindings = json.loads(bindings_json)
        bindings = resolve_bindings(pose, config_bindings, samples_per_bone, confidence_threshold)
        graph = bindings[0].get("routing_mode") == "graph"
        if graph:
            if patches.get("attn1"):
                raise ValueError("Graph bone routing cannot replace an existing attn1 replacement.")
            bindings = prepare_graph_bindings(bindings, config_bindings)
        if any("start_percent" in binding or "end_percent" in binding for binding in bindings):
            sampling = model.get_model_object("model_sampling")
            for binding in bindings:
                if "start_percent" in binding or "end_percent" in binding:
                    binding["start_sigma"] = sampling.percent_to_sigma(binding.get("start_percent", 0.0))
                    binding["end_sigma"] = sampling.percent_to_sigma(binding.get("end_percent", 1.0))
        positive = encode_graph_bindings(clip, positive, bindings) if graph else encode_bindings(clip, positive, bindings)
        patched = model.clone()
        patched.add_wrapper_with_key(WrappersMP.APPLY_MODEL, "bone_prompt", apply_bone_context)
        patch_attn1 = not patches.get("attn1")
        for kind in ("input", "output"):
            count = len(model.get_model_object(f"diffusion_model.{kind}_blocks"))
            for index in range(count):
                patched.set_model_attn2_replace(sparse_cross_attention, kind, index)
                if patch_attn1:
                    patched.set_model_attn1_replace(sparse_self_attention, kind, index)
        patched.set_model_attn2_replace(sparse_cross_attention, "middle", 0)
        if patch_attn1:
            patched.set_model_attn1_replace(sparse_self_attention, "middle", 0)
        return patched, positive

    @classmethod
    def IS_CHANGED(cls, pose_file, **kwargs):
        return hashlib.sha256(pose_path(pose_file).read_bytes()).hexdigest()


NODE_CLASS_MAPPINGS = {"BonePromptInject": BonePromptInject}
NODE_DISPLAY_NAME_MAPPINGS = {"BonePromptInject": "Bone Prompt Inject (Experimental SDXL)"}

