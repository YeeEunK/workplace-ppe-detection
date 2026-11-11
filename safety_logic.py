"""Geometry and conservative PPE status rules for the educational demo."""

def contains(point, polygon):
    x, y = point
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-9 and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def associate(detections, polygon):
    people = [dict(d, gear=[]) for d in detections if d['name'] == 'Person']
    unassigned = []
    for item in (d for d in detections if d['name'] != 'Person'):
        x1, y1, x2, y2 = item['box']
        center = ((x1+x2)/2, (y1+y2)/2)
        matches = [p for p in people if p['box'][0] <= center[0] <= p['box'][2] and p['box'][1] <= center[1] <= p['box'][3]]
        if matches:
            # Smallest containing person limits ambiguous overlaps, but is still heuristic.
            min(matches, key=lambda p: (p['box'][2]-p['box'][0])*(p['box'][3]-p['box'][1]))['gear'].append(item['name'])
        else:
            unassigned.append(item)
    for person in people:
        x1, y1, x2, y2 = person['box']
        person['foot'] = ((x1+x2)/2, y2)
        person['intrusion'] = bool(polygon) and contains(person['foot'], polygon)
        for key, positive, negative in [('helmet', 'helmet', 'no_helmet'), ('vest', 'vest', 'no_vest')]:
            present, absent = positive in person['gear'], negative in person['gear']
            person[key] = '충돌' if present and absent else '미착용' if absent else '착용' if present else '확인 불가'
    return people, unassigned


def overall(people, unassigned, zone_enabled):
    if any(p['intrusion'] for p in people):
        return 'DANGER'
    if any(p[k] in ('미착용', '충돌') for p in people for k in ('helmet', 'vest')) or any(d['name'].startswith('no_') for d in unassigned):
        return 'WARNING'
    if not people or not zone_enabled or unassigned or any(p[k] == '확인 불가' for p in people for k in ('helmet', 'vest')):
        return '확인 필요'
    return 'SAFE'
