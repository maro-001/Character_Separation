import math


BONES = {
    "neck": (1, 0),
    "right_shoulder": (1, 2),
    "right_upper_arm": (2, 3),
    "right_forearm": (3, 4),
    "left_shoulder": (1, 5),
    "left_upper_arm": (5, 6),
    "left_forearm": (6, 7),
    "right_torso": (2, 8),
    "right_thigh": (8, 9),
    "right_shin": (9, 10),
    "left_torso": (5, 11),
    "left_thigh": (11, 12),
    "left_shin": (12, 13),
    "right_eye": (0, 14),
    "right_ear": (14, 16),
    "left_eye": (0, 15),
    "left_ear": (15, 17),
}


ARM_BONES = {
    "right_forearm",
    "left_forearm",
    "right_upper_arm",
    "left_upper_arm",
    "right_shoulder",
    "left_shoulder",
}


def _segment_projection(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(px - ax, py - ay), 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    tc = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + tc * dx), py - (ay + tc * dy)), t


def _compute_skeleton_tubes(pose, bound_pairs, confidence_threshold, radius=0.012, grid_size=128):
    width, height = pose["canvas_width"], pose["canvas_height"]
    segments = []
    for person_idx, person in enumerate(pose["people"]):
        flat = person["pose_keypoints_2d"]
        if len(flat) != 54 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in flat):
            continue
        for bone_name, (i, j) in BONES.items():
            start, end = flat[i * 3:i * 3 + 3], flat[j * 3:j * 3 + 3]
            if all(p[2] > 0 and p[2] >= confidence_threshold for p in (start, end)):
                segments.append((
                    person_idx,
                    bone_name,
                    start[0] / width,
                    start[1] / height,
                    end[0] / width,
                    end[1] / height,
                ))
    tubes = {}
    unbound_arm_points = []
    for person_idx, bone_name, ax, ay, bx, by in segments:
        is_bound = (person_idx, bone_name) in bound_pairs
        t_min = -0.15 if is_bound else 0.18
        t_max = 0.96 if is_bound else 0.95
        pts = []
        for gy in range(grid_size):
            py = (gy + 0.5) / grid_size
            for gx in range(grid_size):
                px = (gx + 0.5) / grid_size
                dist, t_proj = _segment_projection(px, py, ax, ay, bx, by)
                if dist <= radius and t_min <= t_proj <= t_max:
                    if is_bound:
                        pts.append((px, py))
                    else:
                        other_min = min(
                            (
                                _segment_projection(px, py, ox0, oy0, ox1, oy1)[0]
                                for op, ob, ox0, oy0, ox1, oy1 in segments
                                if not (op == person_idx and ob == bone_name)
                            ),
                            default=float("inf"),
                        )
                        if dist <= other_min:
                            pts.append((px, py))
        tubes[(person_idx, bone_name)] = tuple(pts)
        if not is_bound and bone_name in ARM_BONES:
            unbound_arm_points.extend(pts)
    return tubes, tuple(unbound_arm_points)


def resolve_bindings(pose, bindings, samples_per_bone, confidence_threshold):
    width, height = pose["canvas_width"], pose["canvas_height"]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("Pose canvas_width and canvas_height must be positive pixel dimensions.")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("bindings_json must contain a nonempty list of bone/text bindings.")
    if samples_per_bone < 2:
        raise ValueError("samples_per_bone must be at least 2.")
    modes = [binding.get("routing_mode", "delta") for binding in bindings]
    if any(mode not in ("delta", "graph") for mode in modes) or len(set(modes)) != 1:
        raise ValueError("Use one routing_mode, delta or graph, for all bindings in a node.")
    resolved = []
    for binding in bindings:
        person, bone = binding["person"], binding["bone"]
        if not isinstance(person, int) or not 0 <= person < len(pose["people"]):
            raise ValueError(f"Invalid person index: {person}")
        if bone not in BONES:
            raise ValueError(f"Unknown COCO-18 bone: {bone}. Choices: {', '.join(BONES)}")
        flat = pose["people"][person]["pose_keypoints_2d"]
        if len(flat) != 54:
            raise ValueError("Expected COCO-18 pose_keypoints_2d with 54 pixel x/y/confidence values.")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in flat):
            raise ValueError("Pose coordinates and confidences must be finite numbers.")
        start, end = [flat[i * 3:i * 3 + 3] for i in BONES[bone]]
        if any(p[2] <= 0 or p[2] < confidence_threshold for p in (start, end)):
            raise ValueError(f"Both endpoints of person {person}, {bone} must be visible.")
        text = binding["text"]
        strength = binding.get("strength", 0.5)
        if not isinstance(text, str):
            raise ValueError("Each bone binding needs a text string.")
        if "reference_text" in binding and not isinstance(binding["reference_text"], str):
            raise ValueError("Each bone reference_text must be a text string.")
        if not isinstance(strength, (int, float)) or not math.isfinite(strength):
            raise ValueError("Each bone strength must be a finite number.")
        points = tuple(((start[0] + (end[0] - start[0]) * i / (samples_per_bone - 1)) / width,
                        (start[1] + (end[1] - start[1]) * i / (samples_per_bone - 1)) / height)
                       for i in range(samples_per_bone))
        if not any(0 <= x < 1 and 0 <= y < 1 for x, y in points):
            raise ValueError(f"No samples of person {person}, {bone} lie inside the pose canvas.")
        resolved.append({"person": person, "bone": bone, "text": text,
                         "strength": strength, "points": points})
        if modes[0] == "graph":
            resolved[-1]["routing_mode"] = "graph"
        if "reference_text" in binding:
            resolved[-1]["reference_text"] = binding["reference_text"]
        if "query_mode" in binding:
            if binding["query_mode"] not in ("bone", "sample"):
                raise ValueError("Each bone query_mode must be bone or sample.")
            resolved[-1]["query_mode"] = binding["query_mode"]
        if "block_types" in binding:
            blocks = binding["block_types"]
            if (not isinstance(blocks, list) or not blocks
                    or any(block not in ("input", "middle", "output") for block in blocks)):
                raise ValueError("Each bone block_types must be a nonempty list of input, middle or output.")
            resolved[-1]["block_types"] = blocks.copy()
        for key in ("token_indices", "reference_token_indices"):
            if key in binding:
                indices = binding[key]
                if (not isinstance(indices, list) or not indices
                        or any(type(index) is not int or index < 0 for index in indices)
                        or len(set(indices)) != len(indices)):
                    raise ValueError(f"Each bone {key} must be a nonempty list of unique nonnegative integers.")
                if key == "reference_token_indices" and "reference_text" not in binding:
                    raise ValueError("reference_token_indices requires reference_text.")
                resolved[-1][key] = indices.copy()
        for key in ("min_grid_size", "start_percent", "end_percent", "competition_strength", "opposite_strength", "attribute_strength"):
            if key in binding:
                value = binding[key]
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Each bone {key} must be a finite number.")
                if key in ("competition_strength", "opposite_strength", "min_grid_size", "attribute_strength") and value < 0:
                    raise ValueError(f"Each bone {key} must be nonnegative.")
                if key in ("start_percent", "end_percent") and not 0 <= value <= 1:
                    raise ValueError(f"Each bone {key} must be between 0 and 1.")
                resolved[-1][key] = value
        if binding.get("start_percent", 0) >= binding.get("end_percent", 1):
            raise ValueError("Each bone start_percent must be less than end_percent.")
    bound = {(binding["person"], binding["bone"]) for binding in resolved}
    if modes[0] != "graph":
        tubes, unbound_arm_points = _compute_skeleton_tubes(pose, bound, confidence_threshold)
        for binding in resolved:
            tube = tubes.get((binding["person"], binding["bone"]))
            if tube:
                binding["tube_points"] = tube
            if unbound_arm_points:
                binding["unbound_points"] = unbound_arm_points
    for binding in resolved:
        if binding.get("opposite_strength", 0) == 0:
            continue
        bone = binding["bone"]
        if bone.startswith("right_"):
            opposite = "left_" + bone[len("right_"):]
        elif bone.startswith("left_"):
            opposite = "right_" + bone[len("left_"):]
        else:
            raise ValueError(f"Bone {bone} has no anatomical opposite.")
        if (binding["person"], opposite) in bound:
            continue
        other, = resolve_bindings(pose, [{"person": binding["person"], "bone": opposite, "text": binding["text"]}],
                                  samples_per_bone, confidence_threshold)
        binding["opposite_bone"] = opposite
        binding["opposite_points"] = other["points"]
    return resolved


def sample_indices(points, height, width):
    return sorted({int(y * height) * width + int(x * width)
                   for x, y in points if 0 <= x < 1 and 0 <= y < 1})


def isolate_self_attention(q, k, v, options, attention):
    call_options = {"attn_precision": options.get("attn_precision"), "transformer_options": options}
    heads = options["n_heads"]
    layout = options.get("bone_prompt")
    if layout is None:
        return attention(q, k, v, heads, **call_options)
    height, width = options["activations_shape"][-2:]
    if height * width != q.shape[1]:
        return attention(q, k, v, heads, **call_options)
    output = attention(q, k, v, heads, **call_options)
    active_bindings = [b for b in layout["bindings"] if b.get("strength", 0) > 0]
    if not active_bindings:
        return output
    bound_idx_list = [
        sample_indices(b.get("tube_points", b["points"]), height, width)
        for b in active_bindings
    ]
    all_bound = sorted({i for sub in bound_idx_list for i in sub})
    if not all_bound:
        return output
    bound_set = set(all_bound)
    rest_indices = [i for i in range(q.shape[1]) if i not in bound_set]
    output = output.clone()
    if rest_indices:
        output[:, rest_indices] = attention(
            q[:, rest_indices], k[:, rest_indices], v[:, rest_indices], heads, **call_options
        )
    unbound_pts = active_bindings[0].get("unbound_points", ())
    unbound_arm_set = set(sample_indices(unbound_pts, height, width)) if unbound_pts else set()
    for b_idx, b_indices in enumerate(bound_idx_list):
        if not b_indices:
            continue
        other_bound = {i for j, sub in enumerate(bound_idx_list) if j != b_idx for i in sub}
        excluded = (other_bound | unbound_arm_set) - set(b_indices)
        allowed = [i for i in range(q.shape[1]) if i not in excluded]
        if allowed:
            output[:, b_indices] = attention(
                q[:, b_indices], k[:, allowed], v[:, allowed], heads, **call_options
            )
    return output


def inject_attention(q, k, v, options, attention):
    call_options = {"attn_precision": options.get("attn_precision"), "transformer_options": options}
    heads = options["n_heads"]
    layout = options.get("bone_prompt")
    if layout is None:
        return attention(q, k, v, heads, **call_options)
    base_tokens = layout["base_tokens"]
    if k.shape[1] != base_tokens + layout["bone_tokens"]:
        raise ValueError("Bone prompt context length changed; check other attention patches.")
    output = attention(q, k[:, :base_tokens], v[:, :base_tokens], heads, **call_options)
    height, width = options["activations_shape"][-2:]
    if height * width != q.shape[1]:
        raise ValueError("Bone prompt injection requires full-frame 2D spatial attention.")
    change = output.new_zeros(output.shape)
    overlap_weight = output.new_zeros((output.shape[0], output.shape[1], 1))
    for binding in layout["bindings"]:
        if binding["strength"] == 0:
            continue
        if min(height, width) < binding.get("min_grid_size", 0):
            continue
        if "block_types" in binding and options["block"][0] not in binding["block_types"]:
            continue
        spatial_points = binding.get("tube_points", binding["points"]) if binding["strength"] > 0 else binding["points"]
        indices = sample_indices(spatial_points, height, width)
        if not indices:
            continue
        start = base_tokens + binding["start"]
        end = base_tokens + binding["end"]
        sample_queries = binding.get("query_mode", "bone") == "sample"
        bone_query = q[:, indices] if sample_queries else q[:, indices].mean(dim=1, keepdim=True)
        bone_text = attention(bone_query, k[:, start:end], v[:, start:end], heads, **call_options)
        if "reference_start" in binding:
            ref_start = base_tokens + binding["reference_start"]
            ref_end = base_tokens + binding["reference_end"]
            reference = attention(bone_query, k[:, ref_start:ref_end], v[:, ref_start:ref_end], heads, **call_options)
        else:
            reference = output[:, indices] if sample_queries else output[:, indices].mean(dim=1, keepdim=True)
        raw_delta = bone_text - reference
        delta = raw_delta
        competition = binding.get("competition_strength", 0)
        max_other_norm = None
        if competition != 0:
            opposition = delta.new_zeros(delta.shape)
            opposition_weight = delta.new_zeros((delta.shape[0], 1, 1))
            for other in layout["bindings"]:
                if other is binding or other["strength"] <= 0 or min(height, width) < other.get("min_grid_size", 0):
                    continue
                if "block_types" in other and options["block"][0] not in other["block_types"]:
                    continue
                other_start, other_end = base_tokens + other["start"], base_tokens + other["end"]
                other_text = attention(bone_query, k[:, other_start:other_end], v[:, other_start:other_end], heads, **call_options)
                if "reference_start" in other:
                    other_start = base_tokens + other["reference_start"]
                    other_end = base_tokens + other["reference_end"]
                    other_reference = attention(bone_query, k[:, other_start:other_end], v[:, other_start:other_end], heads, **call_options)
                else:
                    other_reference = output[:, indices] if sample_queries else output[:, indices].mean(dim=1, keepdim=True)
                other_diff = other_text - other_reference
                other_norm = other_diff.norm(dim=-1, keepdim=True)
                max_other_norm = other_norm if max_other_norm is None else max_other_norm.maximum(other_norm)
                other_gate = other.get("gate", 1.0)
                opposition += other_gate * other_diff
                opposition_weight += other_gate
            delta = delta - competition * opposition / opposition_weight.clamp_min_(1)
        attr_indices = binding.get("attribute_indices")
        if attr_indices and "reference_start" in binding and binding["strength"] > 0:
            attr_pos = [start + i for i in attr_indices]
            ref_attr_pos = [base_tokens + binding["reference_start"] + i for i in attr_indices]
            attr_text = attention(bone_query, k[:, attr_pos], v[:, attr_pos], heads, **call_options)
            attr_ref = attention(bone_query, k[:, ref_attr_pos], v[:, ref_attr_pos], heads, **call_options)
            attr_delta = attr_text - attr_ref
            if competition != 0:
                attr_opp = attr_delta.new_zeros(attr_delta.shape)
                attr_opp_w = attr_delta.new_zeros((attr_delta.shape[0], 1, 1))
                for other in layout["bindings"]:
                    if other is binding or other["strength"] <= 0 or min(height, width) < other.get("min_grid_size", 0):
                        continue
                    if "block_types" in other and options["block"][0] not in other["block_types"]:
                        continue
                    o_attr = other.get("attribute_indices")
                    if not o_attr or "reference_start" not in other:
                        continue
                    o_pos = [base_tokens + other["start"] + i for i in o_attr]
                    o_ref_pos = [base_tokens + other["reference_start"] + i for i in o_attr]
                    o_text = attention(bone_query, k[:, o_pos], v[:, o_pos], heads, **call_options)
                    o_ref = attention(bone_query, k[:, o_ref_pos], v[:, o_ref_pos], heads, **call_options)
                    o_gate = other.get("gate", 1.0)
                    attr_opp += o_gate * (o_text - o_ref)
                    attr_opp_w += o_gate
                attr_delta = attr_delta - min(competition, 0.5) * attr_opp / attr_opp_w.clamp_min_(1)
            if "attribute_strength" in binding:
                attr_scale = binding["attribute_strength"] / max(binding["strength"], 1e-6)
            elif binding["strength"] > 2.0:
                attr_scale = 0.72
            elif max_other_norm is not None:
                self_norm = raw_delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                deficit = ((max_other_norm - self_norm) / self_norm).clamp(min=0.0, max=1.0)
                attr_scale = 4.5 * deficit
            else:
                attr_scale = 0.0
            delta = delta + attr_scale * attr_delta
        gate = binding.get("gate", 1.0)
        change[:, indices] += binding["strength"] * gate * delta
        overlap_weight[:, indices] += gate
    change.div_(overlap_weight.clamp_min_(1))
    return output + change

