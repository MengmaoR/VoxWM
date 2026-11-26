"""Spatial overlap of integer voxel windows in one world-seed namespace."""
from collections import Counter, defaultdict


def summarize(scenes, *, cross=False):
    """Compare scene boxes, optionally restricting pairs to Static × Dynamic.

    A record is (seed, center_xyz, category, clip, scene_id). All centers use
    the same local-voxel offset in the recorded 48³ window. Each supplied
    record represents one scene even when several records have the same center.
    """
    groups = defaultdict(list)
    for i, (seed, center, category, clip, scene_id) in enumerate(scenes):
        groups[seed].append((center, category, clip, scene_id, i))

    eligible = total = exact = center16 = across_clip = 0
    volume_sum = 0
    volume_lo = volume_hi = None
    affected = set()
    affected_seeds = set()
    pairs_by_category = Counter()
    overlaps = []

    for seed, group in groups.items():
        n = len(group)
        if cross:
            n_static = sum(row[1] == 'Static' for row in group)
            eligible += n_static * (n - n_static)
            if n_static in (0, n):
                continue
        else:
            eligible += n * (n - 1) // 2

        # By increasing X, a separation of 48 rules out all subsequent rows.
        group.sort(key=lambda row: row[0])
        for j, (a, ca, clip_a, id_a, i_a) in enumerate(group):
            for k in range(j + 1, n):
                b, cb, clip_b, id_b, i_b = group[k]
                dx = b[0] - a[0]
                if dx >= 48:
                    break
                if cross and (ca == 'Static') == (cb == 'Static'):
                    continue
                dy, dz = abs(a[1] - b[1]), abs(a[2] - b[2])
                if dy >= 48 or dz >= 48:
                    continue
                volume = (48 - dx) * (48 - dy) * (48 - dz)
                total += 1
                same = dx == dy == dz == 0
                exact += int(same)
                center16 += int(max(dx, dy, dz) < 16)
                across_clip += int(clip_a != clip_b)
                affected.update((i_a, i_b))
                affected_seeds.add(seed)
                pairs_by_category['/'.join(sorted((ca, cb)))] += 1
                volume_sum += volume
                volume_lo = volume if volume_lo is None else min(volume_lo, volume)
                volume_hi = volume if volume_hi is None else max(volume_hi, volume)
                overlaps.append(dict(seed=seed, centers=[list(a), list(b)], delta=[dx, dy, dz],
                                     ids=[id_a, id_b], clips=[clip_a, clip_b], categories=[ca, cb],
                                     volume=volume, center16_volume=max(0, 16-dx)*max(0, 16-dy)*max(0, 16-dz)))

    by_category = dict.fromkeys(sorted({s[2] for s in scenes}), 0)
    for i in affected:
        by_category[scenes[i][2]] += 1
    return dict(scenes=len(scenes), seeds=len(groups), eligible_pairs=eligible,
                overlap_pairs=total, affected_scenes=len(affected), affected_seeds=len(affected_seeds),
                affected_by_category=by_category, identical_center_pairs=exact,
                partial_pairs=total - exact, center16_overlap_pairs=center16,
                pair_volume_sum=volume_sum, pair_volume_mean=volume_sum / total if total else None,
                pair_volume_min=volume_lo, pair_volume_max=volume_hi,
                within_clip_pairs=total - across_clip, across_clip_pairs=across_clip,
                pairs_by_category=dict(sorted(pairs_by_category.items())), overlaps=overlaps)
