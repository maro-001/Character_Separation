def node(cls, inputs, title=None):
    return dict(class_type=cls, inputs=inputs, _meta=dict(title=title or cls))

def ui_workflow(prompt, definitions):
    positions = {'1': [0, 40], '10': [0, 270], '11': [0, 480], '12': [350, 40], '8': [350, 280], '9': [780, 450], '2': [780, 40], '3': [780, 260], '4': [1200, 680], '5': [1200, 40], '6': [1570, 40], '7': [1900, 40], '40': [0, 850], '41': [510, 850]}
    nodes = []
    links = []
    lid = 0
    for (id, p) in prompt.items():
        definition = definitions[p['class_type']]
        n = dict(id=int(id), type=p['class_type'], title=p['_meta']['title'], pos=positions.get(id, [0, 0]), size=[330, 220], flags={}, order=len(nodes), mode=0, inputs=[], outputs=[], properties={'Node name for S&R': p['class_type']}, widgets_values=[], widgets_values_named={})
        for kind in ['required', 'optional']:
            for (name, schema) in definition['input'].get(kind, {}).items():
                if name not in p['inputs']:
                    continue
                value = p['inputs'][name]
                typ = schema[0]
                if isinstance(value, list) and len(value) == 2 and (str(value[0]) in prompt):
                    n['inputs'].append(dict(name=name, type=typ, link=None))
                elif isinstance(typ, list) or typ in ['STRING', 'INT', 'FLOAT', 'BOOLEAN']:
                    n['widgets_values'].append(value)
                    n['widgets_values_named'][name] = value
                    if name == 'seed':
                        n['widgets_values'].append('fixed')
                        n['widgets_values_named']['control_after_generate'] = 'fixed'
                else:
                    n['inputs'].append(dict(name=name, type=typ, link=None))
        for (slot, typ) in enumerate(definition['output']):
            n['outputs'].append(dict(name=definition.get('output_name', definition['output'])[slot], type=typ, links=[]))
        if id == '40':
            n['size'] = [440, 650]
            n['properties']['pose_editor_json'] = p['inputs']['POSE_JSON']
            n['properties']['reference_pose_json'] = p['inputs']['POSE_JSON']
        if id in ['2', '3']:
            n['size'] = [350, 170]
        if id in ['7', '41']:
            n['size'] = [360, 380]
        nodes.append(n)
    lookup = {str(n['id']): n for n in nodes}
    for (id, p) in prompt.items():
        for (name, value) in p['inputs'].items():
            if not (isinstance(value, list) and len(value) == 2 and (str(value[0]) in lookup)):
                continue
            lid += 1
            target = lookup[id]
            source = lookup[str(value[0])]
            slot = next((i for (i, x) in enumerate(target['inputs']) if x['name'] == name))
            target['inputs'][slot]['link'] = lid
            source['outputs'][value[1]]['links'].append(lid)
            links.append([lid, int(value[0]), value[1], int(id), slot, source['outputs'][value[1]]['type']])
    return dict(last_node_id=max((n['id'] for n in nodes)), last_link_id=lid, nodes=nodes, links=links, groups=[], config={}, extra={'ds': {'scale': 0.55, 'offset': [80, 70]}}, version=0.4)
