import math

import torch

from .core import BONES, _segment_projection


def resolve_bias_phrase(clip, text, phrase):
    """Resolve a unique text phrase to matching SDXL CLIP token positions."""
    if not isinstance(phrase, str) or not phrase.strip():
        raise ValueError("token_bias_text must be a nonempty phrase.")
    full, part = clip.tokenize(text), clip.tokenize(phrase)
    matches = []
    for name in ("l", "g"):
        if name not in full or name not in part or len(full[name]) != 1 or len(part[name]) != 1:
            raise ValueError("token_bias_text requires single-block SDXL text in both encoders.")
        full_ids = [token[0] for token in full[name][0]]
        part_ids = [token[0] for token in part[name][0]]
        if not all(isinstance(token, int) for token in full_ids + part_ids):
            raise ValueError("token_bias_text does not support custom embedding tokens.")
        if 49407 not in full_ids or 49407 not in part_ids:
            raise ValueError("token_bias_text requires standard CLIP end tokens.")
        full_ids = full_ids[1:full_ids.index(49407)]
        part_ids = part_ids[1:part_ids.index(49407)]
        if not part_ids:
            raise ValueError("token_bias_text must contain at least one token.")
        starts = [i + 1 for i in range(len(full_ids) - len(part_ids) + 1)
                  if full_ids[i:i + len(part_ids)] == part_ids]
        if len(starts) != 1:
            raise ValueError("token_bias_text must occur exactly once in the full prompt.")
        matches.append(list(range(starts[0], starts[0] + len(part_ids))))
    if matches[0] != matches[1]:
        raise ValueError("token_bias_text must align in both SDXL encoders.")
    return matches[0]


def prepare_graph_bindings(bindings, config):
    identities = [(binding["person"], binding["bone"]) for binding in bindings]
    if len(set(identities)) != len(identities):
        raise ValueError("Graph routing requires a unique person/bone for each binding.")
    profiles = []
    for binding, source in zip(bindings, config):
        unsupported = {"reference_text", "reference_token_indices", "competition_strength",
                       "opposite_strength", "attribute_strength"}.intersection(source)
        if unsupported:
            raise ValueError(f"Graph routing uses direct text blending; remove {', '.join(sorted(unsupported))}.")
        if source.get("query_mode", "sample") != "sample":
            raise ValueError("Graph routing uses sample queries.")
        if "cross_start_percent" in source or "cross_end_percent" in source:
            cross_start = source.get("cross_start_percent", 0.0)
            cross_end = source.get("cross_end_percent", 1.0)
            if (not isinstance(cross_start, (int, float)) or not math.isfinite(cross_start)
                    or not isinstance(cross_end, (int, float)) or not math.isfinite(cross_end)
                    or not 0 <= cross_start < cross_end <= 1):
                raise ValueError("Graph cross_start_percent and cross_end_percent must define a window within 0..1.")
            binding["cross_start_percent"], binding["cross_end_percent"] = cross_start, cross_end
            fade = source.get("cross_fade", 0.0)
            if not isinstance(fade, (int, float)) or not math.isfinite(fade) or not 0 <= fade <= 0.5:
                raise ValueError("Graph cross_fade must be between 0 and 0.5.")
            binding["cross_fade"] = fade
        elif "cross_fade" in source:
            raise ValueError("Graph cross_fade requires a cross-attention time window.")
        if not 0 <= binding["strength"] <= 1:
            raise ValueError("Graph strength is a blend fraction between 0 and 1.")
        radius = source.get("radius", 0.018)
        if not isinstance(radius, (int, float)) or not math.isfinite(radius) or not 0 < radius <= 0.15:
            raise ValueError("Graph radius must be finite and between 0 (exclusive) and 0.15.")
        binding["radius"] = radius
        joint_falloff = source.get("joint_falloff", 0.0)
        if not isinstance(joint_falloff, (int, float)) or not math.isfinite(joint_falloff) or not 0 <= joint_falloff <= 0.5:
            raise ValueError("Graph joint_falloff must be between 0 and 0.5.")
        binding["joint_falloff"] = joint_falloff
        if "focus_sigma" in source:
            sigma = source["focus_sigma"]
            position = source.get("focus_position", 0.5)
            if not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or not 0 < sigma <= 1:
                raise ValueError("Graph focus_sigma must be greater than 0 and at most 1.")
            if not isinstance(position, (int, float)) or not math.isfinite(position) or not 0 <= position <= 1:
                raise ValueError("Graph focus_position must be between 0 and 1.")
            gain = source.get("focus_gain", 1.0)
            if not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain <= 0:
                raise ValueError("Graph focus_gain must be finite and greater than 0.")
            binding["focus_sigma"], binding["focus_position"] = sigma, position
            binding["focus_gain"] = gain
            if "focus_side" in source:
                side = source["focus_side"]
                side_sigma = source.get("focus_side_sigma", 0.5)
                if not isinstance(side, (int, float)) or not math.isfinite(side) or not -1 <= side <= 1:
                    raise ValueError("Graph focus_side must be between -1 and 1.")
                if not isinstance(side_sigma, (int, float)) or not math.isfinite(side_sigma) or side_sigma <= 0:
                    raise ValueError("Graph focus_side_sigma must be finite and greater than 0.")
                binding["focus_side"], binding["focus_side_sigma"] = side, side_sigma
            elif "focus_side_sigma" in source:
                raise ValueError("Graph focus_side_sigma requires focus_side.")
        elif any(key in source for key in ("focus_position", "focus_gain", "focus_side", "focus_side_sigma")):
            raise ValueError("Graph focus options require focus_sigma.")
        route_samples = source.get("route_samples", 1)
        if type(route_samples) is not int or not 1 <= route_samples <= 8:
            raise ValueError("Graph route_samples must be an integer between 1 and 8.")
        binding["route_samples"] = route_samples
        self_strength = source.get("self_attention_strength", 0.12)
        if not isinstance(self_strength, (int, float)) or not math.isfinite(self_strength) or not 0 <= self_strength <= 1:
            raise ValueError("Graph self_attention_strength must be a blend fraction between 0 and 1.")
        binding["self_attention_strength"] = self_strength
        self_scope = source.get("self_attention_scope", "neighbors")
        if self_scope not in ("neighbors", "partition", "person", "target", "target_read", "target_person_read"):
            raise ValueError("Graph self_attention_scope must be neighbors, partition, person, target, target_read or target_person_read.")
        binding["self_attention_scope"] = self_scope
        isolate = source.get("isolate_self_attention", False)
        if type(isolate) is not bool:
            raise ValueError("Graph isolate_self_attention must be a boolean.")
        binding["isolate_self_attention"] = isolate
        person_background = source.get("person_read_background", True)
        if type(person_background) is not bool:
            raise ValueError("Graph person_read_background must be a boolean.")
        binding["person_read_background"] = person_background
        threshold = source.get("isolation_min_weight", 0.0)
        if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Graph isolation_min_weight must be between 0 and 1.")
        binding["isolation_min_weight"] = threshold
        overlap_mode = source.get("overlap_mode", "blend")
        if overlap_mode not in ("blend", "nearest"):
            raise ValueError("Graph overlap_mode must be blend or nearest.")
        binding["overlap_mode"] = overlap_mode
        if route_samples > 1 and overlap_mode != "nearest":
            raise ValueError("Graph route_samples greater than 1 requires nearest overlap mode.")
        token_bias = source.get("token_bias", 0.0)
        bias_indices = source.get("token_bias_indices", [])
        if not isinstance(token_bias, (int, float)) or not math.isfinite(token_bias):
            raise ValueError("Graph token_bias must be finite.")
        if (not isinstance(bias_indices, list)
                or any(type(index) is not int or index < 0 for index in bias_indices)
                or len(set(bias_indices)) != len(bias_indices)):
            raise ValueError("Graph token_bias_indices must contain unique nonnegative integers.")
        if token_bias != 0 and not bias_indices:
            raise ValueError("Graph token_bias requires token_bias_indices.")
        if bias_indices:
            binding["token_bias"], binding["token_bias_indices"] = token_bias, bias_indices.copy()
        if "reference_binding" in source:
            reference = source["reference_binding"]
            if (not isinstance(reference, list) or len(reference) != 2
                    or type(reference[0]) is not int or not isinstance(reference[1], str)
                    or tuple(reference) not in identities):
                raise ValueError("Graph reference_binding must identify an existing [person, bone].")
            delta_scale = source.get("delta_scale", 0.5)
            if not isinstance(delta_scale, (int, float)) or not math.isfinite(delta_scale) or delta_scale < 0:
                raise ValueError("Graph delta_scale must be finite and nonnegative.")
            binding["reference_binding"], binding["delta_scale"] = tuple(reference), delta_scale
            gain_limit = source.get("delta_gain_limit", 4.0)
            if not isinstance(gain_limit, (int, float)) or not math.isfinite(gain_limit) or gain_limit <= 0:
                raise ValueError("Graph delta_gain_limit must be finite and greater than 0.")
            binding["delta_gain_limit"] = gain_limit
            reference_attention = source.get("reference_attention", "independent")
            if reference_attention not in ("independent", "target"):
                raise ValueError("Graph reference_attention must be independent or target.")
            binding["reference_attention"] = reference_attention
            if "reference_delta_tokens" in source:
                selected = source["reference_delta_tokens"]
                if (reference_attention != "target" or not isinstance(selected, list) or not selected
                        or any(type(index) is not int or index < 0 for index in selected)
                        or len(set(selected)) != len(selected)):
                    raise ValueError("Graph reference_delta_tokens requires target reference and unique nonnegative indices.")
                binding["reference_delta_tokens"] = selected.copy()
            if "reference_baseline_tokens" in source:
                baseline_tokens = source["reference_baseline_tokens"]
                if (reference_attention != "target" or not isinstance(baseline_tokens, list) or not baseline_tokens
                        or any(type(index) is not int or index < 0 for index in baseline_tokens)
                        or len(set(baseline_tokens)) != len(baseline_tokens)):
                    raise ValueError("Graph reference_baseline_tokens requires target reference and unique nonnegative indices.")
                binding["reference_baseline_tokens"] = baseline_tokens.copy()
        elif "delta_scale" in source:
            raise ValueError("Graph delta_scale requires reference_binding.")
        elif "reference_attention" in source:
            raise ValueError("Graph reference_attention requires reference_binding.")
        elif "reference_delta_tokens" in source:
            raise ValueError("Graph reference_delta_tokens requires reference_binding.")
        elif "delta_gain_limit" in source:
            raise ValueError("Graph delta_gain_limit requires reference_binding.")
        elif "reference_baseline_tokens" in source:
            raise ValueError("Graph reference_baseline_tokens requires reference_binding.")
        if "reference_delta_normalization" in source:
            normalization = source["reference_delta_normalization"]
            if normalization not in ("all", "selected"):
                raise ValueError("Graph reference_delta_normalization must be all or selected.")
            if normalization == "selected" and "reference_delta_tokens" not in binding:
                raise ValueError("Graph selected reference_delta_normalization requires reference_delta_tokens.")
            binding["reference_delta_normalization"] = normalization
        binding["joints"] = BONES[binding["bone"]]
        profiles.append((binding.get("start_percent", 0), binding.get("end_percent", 1),
                         binding.get("min_grid_size", 0),
                         tuple(sorted(set(binding.get("block_types", ("input", "middle", "output"))))),
                         self_strength, self_scope, overlap_mode, route_samples))
    if len(set(profiles)) != 1:
        raise ValueError("Graph bindings must share their time window, minimum grid, block_types, self_attention_strength, self_attention_scope, overlap_mode and route_samples.")
    return bindings


def attach_person_geometry(bindings, pose, confidence_threshold, include_unbound=False):
    width, height = pose["canvas_width"], pose["canvas_height"]
    people = {}
    for binding in bindings:
        person = binding["person"]
        if person not in people:
            flat = pose["people"][person]["pose_keypoints_2d"]
            visible = {joint: (flat[joint * 3] / width, flat[joint * 3 + 1] / height)
                       for joint in range(18) if flat[joint * 3 + 2] > 0
                       and flat[joint * 3 + 2] >= confidence_threshold}
            people[person] = {
                "person_bounds": (min(point[0] for point in visible.values()), min(point[1] for point in visible.values()),
                                  max(point[0] for point in visible.values()), max(point[1] for point in visible.values())),
                "person_segments": tuple((*visible[a], *visible[b]) for a, b in (*BONES.values(), (8, 11))
                                         if a in visible and b in visible),
                "person_torso": tuple(visible[joint] for joint in (2, 5, 11, 8))
                                if all(joint in visible for joint in (2, 5, 11, 8)) else (),
            }
        binding.update(people[person])
    if include_unbound:
        obstacles = []
        for person, item in enumerate(pose["people"]):
            if person in people:
                continue
            flat = item.get("pose_keypoints_2d", [])
            if len(flat) != 54 or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in flat):
                continue
            visible = {joint: (flat[joint * 3] / width, flat[joint * 3 + 1] / height)
                       for joint in range(18) if flat[joint * 3 + 2] > 0
                       and flat[joint * 3 + 2] >= confidence_threshold}
            edges = [(a, b) for a, b in (*BONES.values(), (8, 11)) if a in visible and b in visible]
            connected = {joint for edge in edges for joint in edge}
            segments = (tuple((*visible[a], *visible[b]) for a, b in edges)
                        + tuple((*visible[joint], *visible[joint]) for joint in visible if joint not in connected))
            if not segments:
                continue
            obstacles.append({"person": person,
                              "person_bounds": (min(point[0] for point in visible.values()),
                                                min(point[1] for point in visible.values()),
                                                max(point[0] for point in visible.values()),
                                                max(point[1] for point in visible.values())),
                              "person_segments": segments,
                              "person_torso": tuple(visible[joint] for joint in (2, 5, 11, 8))
                              if all(joint in visible for joint in (2, 5, 11, 8)) else ()})
        for binding in bindings:
            binding["unbound_person_geometry"] = tuple(obstacles)


def _inside_polygon(px, py, polygon):
    inside = False
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1]):
        if (ay > py) != (by > py) and px < ax + (py - ay) * (bx - ax) / (by - ay):
            inside = not inside
    return inside


def build_person_regions(bindings, height, width):
    people = {}
    for binding in bindings:
        person = binding["person"]
        if person not in people:
            people[person] = {"bounds": binding["person_bounds"], "radius": binding["radius"],
                              "segments": binding["person_segments"], "torso": binding["person_torso"]}
        people[person]["radius"] = max(people[person]["radius"], binding["radius"])
    radius = max(binding["radius"] for binding in bindings)
    for person in bindings[0].get("unbound_person_geometry", ()):
        people[person["person"]] = {"bounds": person["person_bounds"], "radius": radius,
                                    "segments": person["person_segments"], "torso": person["person_torso"]}
    groups = {person: [] for person in sorted(people)}
    background = []
    for y in range(height):
        py = (y + 0.5) / height
        for x in range(width):
            px = (x + 0.5) / width
            candidates = []
            for person, data in people.items():
                x0, y0, x1, y1 = data["bounds"]
                radius = data["radius"]
                if x0 - radius <= px <= x1 + radius and y0 - radius <= py <= y1 + radius:
                    distance = min(_segment_projection(px, py, *segment)[0] for segment in data["segments"])
                    if distance <= radius or _inside_polygon(px, py, data["torso"]):
                        candidates.append((distance, person))
            index = y * width + x
            if not candidates:
                background.append(index)
                continue
            owner = min(candidates)[1]
            groups[owner].append(index)
    return list(groups.values()) + [background]


def _focus_factor(binding, position, px, py, ax, ay, bx, by, radius):
    offset = (position - binding["focus_position"]) / binding["focus_sigma"]
    weight = min(1.0, binding.get("focus_gain", 1.0) * math.exp(-0.5 * offset * offset))
    if "focus_side" in binding:
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        signed = ((dx / length) * (py - ay) - (dy / length) * (px - ax)) / radius if length else 0.0
        offset = (signed - binding["focus_side"]) / binding["focus_side_sigma"]
        weight *= math.exp(-0.5 * offset * offset)
    return weight


def build_routes(bindings, height, width):
    spatial = []
    owners = {}
    focused_ownership = any("focus_sigma" in binding for binding in bindings)
    for binding_index, binding in enumerate(bindings):
        ax, ay = binding["points"][0]
        bx, by = binding["points"][-1]
        samples = binding.get("route_samples", 1)
        if samples > 1:
            radius = binding["radius"]
            x0, x1 = max(0, math.floor((min(ax, bx) - radius) * width)), min(width, math.ceil((max(ax, bx) + radius) * width))
            y0, y1 = max(0, math.floor((min(ay, by) - radius) * height)), min(height, math.ceil((max(ay, by) + radius) * height))
            indices, weights = [], []
            for y in range(y0, y1):
                for x in range(x0, x1):
                    weight = 0.0
                    ownership_weight = 0.0
                    for sy in range(samples):
                        for sx in range(samples):
                            px = (x + (sx + 0.5) / samples) / width
                            py = (y + (sy + 0.5) / samples) / height
                            distance, position = _segment_projection(px, py, ax, ay, bx, by)
                            if distance > radius:
                                continue
                            taper = binding.get("joint_falloff", 0.0)
                            factor = max(0.0, min(1.0, position / taper, (1.0 - position) / taper)) if taper else 1.0
                            coverage = factor * math.exp(-0.5 * (distance / radius) ** 2)
                            ownership_weight += coverage
                            if "focus_sigma" in binding:
                                coverage *= _focus_factor(binding, position, px, py, ax, ay, bx, by, radius)
                            weight += coverage
                    weight /= samples * samples
                    ownership_weight /= samples * samples
                    if (ownership_weight if focused_ownership else weight) == 0:
                        continue
                    index = y * width + x
                    indices.append(index)
                    weights.append(weight)
                    rank = (-(ownership_weight if focused_ownership else weight), binding["person"], binding["bone"])
                    if index not in owners or rank < owners[index][0]:
                        owners[index] = (rank, binding_index)
            spatial.append((tuple(indices), tuple(weights)))
            continue
        radius = binding["radius"] + 0.5 / min(height, width)
        x0, x1 = max(0, math.floor((min(ax, bx) - radius) * width)), min(width, math.ceil((max(ax, bx) + radius) * width))
        y0, y1 = max(0, math.floor((min(ay, by) - radius) * height)), min(height, math.ceil((max(ay, by) + radius) * height))
        indices, weights = [], []
        for y in range(y0, y1):
            for x in range(x0, x1):
                distance, position = _segment_projection((x + 0.5) / width, (y + 0.5) / height, ax, ay, bx, by)
                if distance > radius:
                    continue
                index = y * width + x
                indices.append(index)
                weight = math.exp(-0.5 * (distance / radius) ** 2)
                falloff = binding.get("joint_falloff", 0.0)
                if falloff:
                    weight *= max(0.0, min(1.0, position / falloff, (1.0 - position) / falloff))
                if "focus_sigma" in binding:
                    weight *= _focus_factor(binding, position, (x + 0.5) / width, (y + 0.5) / height,
                                            ax, ay, bx, by, binding["radius"])
                weights.append(weight)
                rank = (distance, binding["person"], binding["bone"])
                if index not in owners or rank < owners[index][0]:
                    owners[index] = (rank, binding_index)
        spatial.append((tuple(indices), tuple(weights)))
    for binding_index, (indices, weights) in enumerate(spatial):
        if bindings[binding_index].get("overlap_mode", "blend") == "nearest":
            selected = [(index, weight) for index, weight in zip(indices, weights)
                        if owners[index][1] == binding_index]
        else:
            selected = [(index, weight) for index, weight in zip(indices, weights)
                        if bindings[owners[index][1]]["person"] == bindings[binding_index]["person"]]
        spatial[binding_index] = (tuple(index for index, _ in selected), tuple(weight for _, weight in selected))
    owned = [[] for _ in bindings]
    for index, (_, owner) in sorted(owners.items()):
        owned[owner].append(index)
    background = tuple(index for index in range(height * width) if index not in owners)
    allowed = []
    for binding in bindings:
        neighbors = [j for j, other in enumerate(bindings)
                     if binding["person"] == other["person"]
                     and set(binding["joints"]).intersection(other["joints"])]
        allowed.append(tuple(sorted(set(background).union(*(owned[j] for j in neighbors)))))
    return {"spatial": spatial, "owned": owned, "background": background, "allowed": allowed}


def _active_bindings(q, options):
    height, width = options["activations_shape"][-2:]
    if height * width != q.shape[1]:
        raise ValueError("Graph bone routing requires full-frame 2D spatial attention.")
    bindings = options["bone_prompt"]["bindings"]
    first = bindings[0]
    if min(height, width) < first.get("min_grid_size", 0):
        return [], height, width
    if "block_types" in first and options["block"][0] not in first["block_types"]:
        return [], height, width
    return [binding for binding in bindings if binding["strength"] > 0], height, width


def _routes(layout, bindings, height, width):
    # This cache is created by APPLY_MODEL and discarded after that U-Net call.
    cache = layout["route_cache"]
    if (height, width) not in cache:
        cache[height, width] = build_routes(bindings, height, width)
        if bindings[0].get("self_attention_scope") in ("person", "target_person_read"):
            cache[height, width]["person_groups"] = build_person_regions(bindings, height, width)
            cache[height, width]["person_ids"] = sorted(
                {binding["person"] for binding in bindings}
                | {person["person"] for person in bindings[0].get("unbound_person_geometry", ())})
    return cache[height, width]


def _rms(values):
    if values.dtype in (torch.float16, torch.bfloat16):
        values = values.float()
    return values.square().mean(-1, keepdim=True).sqrt()


def graph_cross_attention(q, k, v, options, attention):
    call_options = {"attn_precision": options.get("attn_precision"), "transformer_options": options}
    heads = options["n_heads"]
    layout = options["bone_prompt"]
    base_tokens = layout["base_tokens"]
    if k.shape[1] != base_tokens + layout["bone_tokens"]:
        raise ValueError("Graph bone context length changed; check other attention patches.")
    base = attention(q, k[:, :base_tokens], v[:, :base_tokens], heads, **call_options)
    bindings, height, width = _active_bindings(q, options)
    if not bindings:
        return base
    routes = _routes(layout, bindings, height, width)
    context_queries = {}
    requests = []
    for binding, (indices, _) in zip(bindings, routes["spatial"]):
        specs = [(binding["start"], binding["end"], binding.get("token_bias", 0),
                  tuple(binding.get("token_bias_indices", ())),
                  (binding["reference_start"], binding["reference_end"])
                  if binding.get("reference_attention") == "target" else None,
                  tuple(binding.get("reference_delta_tokens", ())),
                  binding.get("reference_delta_normalization") == "selected",
                  tuple(binding.get("reference_baseline_tokens", ())))]
        if "reference_start" in binding and binding.get("reference_attention") != "target":
            specs.append((binding["reference_start"], binding["reference_end"],
                          binding["reference_token_bias"], tuple(binding["reference_token_bias_indices"]), None, (), False, ()))
        jobs = []
        for start, end, bias, biased, reference, delta_tokens, selected_only, baseline_tokens in specs:
            key = (start, end, bias, tuple(sorted(biased)) if bias else (), reference, tuple(sorted(delta_tokens)),
                   selected_only, tuple(sorted(baseline_tokens)))
            queries = context_queries.setdefault(key, [])
            jobs.append((key, len(queries)))
            queries.extend(indices)
        requests.append(jobs)
    context_outputs = {}
    for key, indices in context_queries.items():
        if not indices:
            continue
        start, end, bias, biased, reference, delta_tokens, selected_only, baseline_tokens = key
        keys = k[:, base_tokens + start:base_tokens + end]
        values = v[:, base_tokens + start:base_tokens + end]
        if reference is not None:
            reference_values = v[:, base_tokens + reference[0]:base_tokens + reference[1]]
            if baseline_tokens:
                reference_values = reference_values[:, list(baseline_tokens)].mean(dim=1, keepdim=True)
            values = values - reference_values
            if selected_only:
                keys, values = keys[:, list(delta_tokens)], values[:, list(delta_tokens)]
            elif delta_tokens:
                token_mask = values.new_zeros((1, values.shape[1], 1))
                token_mask[:, list(delta_tokens)] = 1
                values = values * token_mask
        mask = None
        if bias != 0:
            mask = q.new_zeros((1, 1, keys.shape[1]))
            positions = [i for i, token in enumerate(delta_tokens) if token in biased] if selected_only else list(biased)
            mask[:, :, positions] = bias
        context_outputs[key] = attention(q[:, indices], keys, values, heads, mask=mask, **call_options)
    change = base.new_zeros(base.shape)
    overlap = base.new_zeros((base.shape[0], base.shape[1], 1))
    for binding, (indices, weights), jobs in zip(bindings, routes["spatial"], requests):
        if not indices:
            continue
        indices = list(indices)
        key, offset = jobs[0]
        local = context_outputs[key][:, offset:offset + len(indices)]
        if binding.get("reference_attention") == "target":
            delta = local
        elif "reference_start" in binding:
            key, offset = jobs[1]
            delta = local - context_outputs[key][:, offset:offset + len(indices)]
        if "reference_start" in binding:
            delta_rms = _rms(delta)
            limit = binding["delta_scale"] * _rms(base[:, indices])
            scale = (limit / delta_rms.clamp_min(1e-6)).clamp_max(binding.get("delta_gain_limit", 4.0))
            local = base[:, indices] + scale * delta
        coverage = (base.new_tensor(weights).reshape(1, -1, 1)
                    * binding.get("gate", 1.0) * binding.get("cross_gate", 1.0))
        change[:, indices] += coverage * binding["strength"] * (local - base[:, indices])
        overlap[:, indices] += coverage
    return base + change / overlap.clamp_min_(1)


def graph_self_attention(q, k, v, options, attention):
    call_options = {"attn_precision": options.get("attn_precision"), "transformer_options": options}
    heads = options["n_heads"]
    layout = options["bone_prompt"]
    bindings, height, width = _active_bindings(q, options)
    base = attention(q, k, v, heads, **call_options)
    if not bindings or bindings[0]["self_attention_strength"] == 0:
        return base
    routes = _routes(layout, bindings, height, width)
    output = base.clone()
    strength = bindings[0]["self_attention_strength"] * bindings[0].get("gate", 1.0)
    groups = zip(routes["owned"], routes["allowed"])
    if bindings[0].get("self_attention_scope", "neighbors") == "partition":
        groups = [(indices, indices) for indices in routes["owned"] + [routes["background"]]]
    elif bindings[0].get("self_attention_scope") == "person":
        groups = [(indices, indices) for indices in routes["person_groups"][:-1]]
    elif bindings[0].get("self_attention_scope") in ("target", "target_read", "target_person_read"):
        isolated = []
        for binding, owned, (indices, weights) in zip(bindings, routes["owned"], routes["spatial"]):
            if not binding.get("isolate_self_attention") or not owned:
                continue
            threshold = binding.get("isolation_min_weight", 0.0)
            if threshold:
                eligible = {index for index, weight in zip(indices, weights) if weight >= threshold}
                owned = [index for index in owned if index in eligible]
            if owned:
                isolated.append((binding, owned))
        if not isolated:
            return base
        excluded = {index for _, indices in isolated for index in indices}
        rest = [index for index in range(q.shape[1]) if index not in excluded]
        scope = bindings[0]["self_attention_scope"]
        groups = [(indices, indices) for _, indices in isolated] if scope == "target" else []
        if scope == "target_person_read":
            people = routes.get("person_ids", sorted({binding["person"] for binding in bindings}))
            person_regions = dict(zip(people, routes["person_groups"][:-1]))
            background = set(routes["person_groups"][-1])
            groups = []
            for binding, indices in isolated:
                allowed = set(person_regions[binding["person"]]) | set(indices)
                if binding.get("person_read_background", True):
                    allowed |= background
                groups.append((indices, sorted(allowed)))
        if rest:
            groups.append((rest, rest))
    for indices, allowed in groups:
        if not indices:
            continue
        indices, allowed = list(indices), list(allowed)
        local = attention(q[:, indices], k[:, allowed], v[:, allowed], heads, **call_options)
        output[:, indices] = (base[:, indices] + strength * (local - base[:, indices])).to(output.dtype)
    return output
