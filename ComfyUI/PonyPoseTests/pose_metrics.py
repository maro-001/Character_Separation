BODY=list(range(18))
ARMS=[2,3,4,5,6,7]
import itertools, math, statistics
def points(person, width, height):
    values = person.get('pose_keypoints_2d') or []
    triplets = [values[i:i + 3] for i in range(0, len(values), 3)]
    pixel = max((max(x[:2]) for x in triplets), default=0) > 2
    return [(x[0] / width if pixel else x[0], x[1] / height if pixel else x[1], x[2]) for x in triplets]

def errors(target, detected):
    result = {}
    for i in BODY:
        if i >= len(target) or target[i][2] <= 0:
            continue
        if i >= len(detected) or detected[i][2] <= 0:
            result[i] = None
        else:
            result[i] = math.hypot(target[i][0] - detected[i][0], target[i][1] - detected[i][1]) / math.sqrt(2)
    return result

def score_pose(reference, estimate):
    frame = reference[0]
    target = [points(p, frame['canvas_width'], frame['canvas_height']) for p in frame['people']]
    estimated = estimate[0] if estimate else {'people': [], 'canvas_width': 1024, 'canvas_height': 1024}
    detected = [points(p, estimated['canvas_width'], estimated['canvas_height']) for p in estimated.get('people', [])][:8]
    candidates = []
    count = len(target)
    padded = detected + [[]] * max(0, count - len(detected))
    for assignment in itertools.permutations(range(len(padded)), count):
        err = [errors(t, padded[j]) for (t, j) in zip(target, assignment)]
        cost = sum((sum((1.0 if d is None else d for d in e.values())) for e in err))
        candidates.append((cost, assignment, err))
    (_, assignment, err) = min(candidates, key=lambda x: x[0])
    all_errors = [d for e in err for d in e.values()]
    arm_errors = [d for e in err for (joint, d) in e.items() if joint in ARMS]
    pck = lambda items: round(100 * sum((d is not None and d <= 0.05 for d in items)) / len(items), 2) if items else 0.0
    visible = [d for d in all_errors if d is not None]
    return dict(people_detected=len(detected), target_people=count, pck_body=pck(all_errors), pck_arms=pck(arm_errors), joints_matched=len(visible), joints_target=len(all_errors), mean_error_pixels=round(statistics.mean(visible) * math.sqrt(2) * 1024, 2) if visible else None, assignment=list(assignment))
