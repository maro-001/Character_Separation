import math

from .core import BONES, _segment_projection


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
        route_samples = source.get("route_samples", 1)
        if type(route_samples) is not int or not 1 <= route_samples <= 8:
            raise ValueError("Graph route_samples must be an integer between 1 and 8.")
        binding["route_samples"] = route_samples
        self_strength = source.get("self_attention_strength", 0.12)
        if not isinstance(self_strength, (int, float)) or not math.isfinite(self_strength) or not 0 <= self_strength <= 1:
            raise ValueError("Graph self_attention_strength must be a blend fraction between 0 and 1.")
        binding["self_attention_strength"] = self_strength
        self_scope = source.get("self_attention_scope", "neighbors")
        if self_scope not in ("neighbors", "partition"):
            raise ValueError("Graph self_attention_scope must be neighbors or partition.")
        binding["self_attention_scope"] = self_scope
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
            reference_attention = source.get("reference_attention", "independent")
            if reference_attention not in ("independent", "target"):
                raise ValueError("Graph reference_attention must be independent or target.")
            binding["reference_attention"] = reference_attention
        elif "delta_scale" in source:
            raise ValueError("Graph delta_scale requires reference_binding.")
        elif "reference_attention" in source:
            raise ValueError("Graph reference_attention requires reference_binding.")
        binding["joints"] = BONES[binding["bone"]]
        profiles.append((binding.get("start_percent", 0), binding.get("end_percent", 1),
                         binding.get("min_grid_size", 0),
                         tuple(sorted(set(binding.get("block_types", ("input", "middle", "output"))))),
                         self_strength, self_scope, overlap_mode, route_samples))
    if len(set(profiles)) != 1:
        raise ValueError("Graph bindings must share their time window, minimum grid, block_types, self_attention_strength, self_attention_scope, overlap_mode and route_samples.")
    return bindings


def build_routes(bindings, height, width):
    spatial = []
    owners = {}
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
                    for sy in range(samples):
                        for sx in range(samples):
                            px = (x + (sx + 0.5) / samples) / width
                            py = (y + (sy + 0.5) / samples) / height
                            distance, position = _segment_projection(px, py, ax, ay, bx, by)
                            if distance > radius:
                                continue
                            taper = binding.get("joint_falloff", 0.0)
                            factor = max(0.0, min(1.0, position / taper, (1.0 - position) / taper)) if taper else 1.0
                            weight += factor * math.exp(-0.5 * (distance / radius) ** 2)
                    weight /= samples * samples
                    if weight == 0:
                        continue
                    index = y * width + x
                    indices.append(index)
                    weights.append(weight)
                    rank = (-weight, binding["person"], binding["bone"])
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
    return cache[height, width]


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
                  if binding.get("reference_attention") == "target" else None)]
        if "reference_start" in binding and binding.get("reference_attention") != "target":
            specs.append((binding["reference_start"], binding["reference_end"],
                          binding["reference_token_bias"], tuple(binding["reference_token_bias_indices"]), None))
        jobs = []
        for start, end, bias, biased, reference in specs:
            key = (start, end, bias, tuple(sorted(biased)) if bias else (), reference)
            queries = context_queries.setdefault(key, [])
            jobs.append((key, len(queries)))
            queries.extend(indices)
        requests.append(jobs)
    context_outputs = {}
    for key, indices in context_queries.items():
        if not indices:
            continue
        start, end, bias, biased, reference = key
        mask = None
        if bias != 0:
            mask = q.new_zeros((1, 1, end - start))
            mask[:, :, list(biased)] = bias
        values = v[:, base_tokens + start:base_tokens + end]
        if reference is not None:
            values = values - v[:, base_tokens + reference[0]:base_tokens + reference[1]]
        context_outputs[key] = attention(q[:, indices], k[:, base_tokens + start:base_tokens + end],
                                         values, heads, mask=mask, **call_options)
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
            delta_rms = delta.square().mean(-1, keepdim=True).sqrt()
            limit = binding["delta_scale"] * base[:, indices].square().mean(-1, keepdim=True).sqrt()
            scale = (limit / delta_rms.clamp_min(1e-6)).clamp_max(4)
            local = base[:, indices] + scale * delta
        coverage = base.new_tensor(weights).reshape(1, -1, 1) * binding.get("gate", 1.0)
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
    for indices, allowed in groups:
        if not indices:
            continue
        indices, allowed = list(indices), list(allowed)
        local = attention(q[:, indices], k[:, allowed], v[:, allowed], heads, **call_options)
        output[:, indices] = (base[:, indices] + strength * (local - base[:, indices])).to(output.dtype)
    return output
