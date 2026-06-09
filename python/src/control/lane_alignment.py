#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lane-pair selection and post-turn alignment checks."""

import math
from statistics import pstdev


def select_lane_pair(xs, image_center, expected_width, reference_center=None,
                     turn_bias=None):
    """Select one adjacent lane pair without spanning an unrelated third line."""
    points = sorted(float(x) for x in xs)
    if len(points) < 2:
        return None

    reference = image_center if reference_center is None else reference_center
    min_width = max(30.0, expected_width * 0.3)
    candidates = []
    for left, right in zip(points, points[1:]):
        width = right - left
        if width < min_width:
            continue
        center = (left + right) / 2.0
        score = abs(center - reference) + 0.15 * abs(width - expected_width)
        if turn_bias == "right":
            score -= 0.12 * (center - image_center)
        elif turn_bias == "left":
            score += 0.12 * (center - image_center)
        candidates.append((score, left, right, center))

    if not candidates:
        return None
    _, left, right, center = min(candidates, key=lambda item: item[0])
    return left, right, center


def _lane_pair_candidates(xs, image_center, expected_width):
    points = sorted(float(x) for x in xs)
    min_width = max(30.0, expected_width * 0.3)
    max_width = expected_width * 2.2
    pairs = []
    for left, right in zip(points, points[1:]):
        width = right - left
        if width < min_width or width > max_width:
            continue
        center = (left + right) / 2.0
        straddles_center = left <= image_center <= right
        pairs.append({
            "left": left,
            "right": right,
            "center": center,
            "width": width,
            "straddles_center": straddles_center,
        })
    return pairs


def select_lane_path(candidates, image_width, expected_width=None,
                     reference_center=None, turn_bias=None,
                     max_center_jump=None):
    """Track one continuous lane pair from near rows to far rows.

    The first pair is chosen from rows near the car. Later rows must stay close
    to the previous lane center, so points from a third branch/path are ignored.
    Returns ``{y: (left, right, center)}``.
    """
    image_center = image_width / 2.0
    expected_width = expected_width or image_width * 0.20
    reference = reference_center if reference_center is not None else image_center
    max_center_jump = max_center_jump or expected_width * 0.35
    selected = {}
    locked = False

    for y, xs in reversed(candidates):
        pairs = _lane_pair_candidates(xs, image_center, expected_width)
        if not pairs:
            continue

        scored = []
        for pair in pairs:
            width_penalty = 0.15 * abs(pair["width"] - expected_width)
            center_penalty = abs(pair["center"] - reference)
            score = center_penalty + width_penalty
            if not locked and pair["straddles_center"]:
                score -= expected_width * 0.35
            if turn_bias == "right":
                score -= 0.08 * (pair["center"] - image_center)
            elif turn_bias == "left":
                score += 0.08 * (pair["center"] - image_center)
            scored.append((score, pair))

        _, best = min(scored, key=lambda item: item[0])
        if locked and abs(best["center"] - reference) > max_center_jump:
            continue

        selected[y] = (best["left"], best["right"], best["center"])
        reference = best["center"]
        locked = True

    return selected


def _track_side_points(points_by_row, max_jump):
    tracks = []
    for row_index, y, xs in points_by_row:
        used = set()
        for track in sorted(tracks, key=lambda item: item["last_index"],
                            reverse=True):
            if track["last_index"] == row_index:
                continue
            if row_index - track["last_index"] > 2:
                continue
            best = None
            for idx, x in enumerate(xs):
                if idx in used:
                    continue
                jump = abs(x - track["last_x"])
                if jump <= max_jump and (best is None or jump < best[0]):
                    best = (jump, idx, x)
            if best is None:
                continue
            _, idx, x = best
            track["points"].append((row_index, y, x))
            track["last_index"] = row_index
            track["last_x"] = x
            used.add(idx)

        for idx, x in enumerate(xs):
            if idx not in used:
                tracks.append({
                    "points": [(row_index, y, x)],
                    "last_index": row_index,
                    "last_x": x,
                })
    return tracks


def _select_side_points(candidates, image_center, side, continuity_jump):
    points_by_row = []
    for row_index, (y, xs) in enumerate(candidates):
        if side == "left":
            side_xs = sorted(float(x) for x in xs if float(x) < image_center)
        else:
            side_xs = sorted(float(x) for x in xs if float(x) > image_center)
        points_by_row.append((row_index, y, side_xs))

    tracks = _track_side_points(points_by_row, continuity_jump)
    long_tracks = [track for track in tracks if len(track["points"]) >= 3]
    selected = {}
    if long_tracks:
        def track_score(track):
            avg_center_dist = (
                sum(abs(x - image_center) for _, _, x in track["points"])
                / len(track["points"])
            )
            return avg_center_dist - len(track["points"]) * 5.0

        best_track = min(long_tracks, key=track_score)
        selected = {
            row_index: x
            for row_index, _, x in best_track["points"]
        }

    for row_index, y, side_xs in points_by_row:
        if row_index in selected or not side_xs:
            continue
        if side == "left":
            selected[row_index] = max(side_xs)
        else:
            selected[row_index] = min(side_xs)
    return selected


def select_center_side_lane_points(candidates, image_width, expected_width=None):
    """Select nearest left/right lane evidence around camera center.

    A side only enables continuity filtering when at least one candidate line
    has 3 or more scan-row points. Otherwise each visible single point is kept,
    because close lane borders can legitimately appear as only one point.
    """
    image_center = image_width / 2.0
    expected_width = expected_width or image_width * 0.20
    continuity_jump = expected_width * 0.45
    left = _select_side_points(
        candidates, image_center, "left", continuity_jump)
    right = _select_side_points(
        candidates, image_center, "right", continuity_jump)
    return left, right


def build_lane_row_info(candidates, image_width, expected_width=None,
                        reference_center=None, turn_bias=None):
    """Build per-scan-row control info while preserving single-line evidence.

    ``lane_center`` rows are selected lane pairs and can update the reference
    center. ``single_line`` rows are not used as a center reference; they only
    tell the controller that a nearer row still sees a line, which produces the
    small ``dist`` correction after turns.
    """
    image_center = image_width / 2.0
    expected_width = expected_width or image_width * 0.20
    left_points, right_points = select_center_side_lane_points(
        candidates, image_width, expected_width)

    row_info = []
    for row_index, (y, xs) in enumerate(candidates):
        raw_xs = sorted(float(x) for x in xs)
        left = left_points.get(row_index)
        right = right_points.get(row_index)
        if left is not None and right is not None:
            selected_xs = [left, right]
            mode = "lane_center"
            target = (left + right) / 2.0
            xs_out = selected_xs
        elif left is not None:
            mode = "single_line"
            target = left
            xs_out = [left]
        elif right is not None:
            mode = "single_line"
            target = right
            xs_out = [right]
        elif raw_xs:
            nearest_x = min(raw_xs, key=lambda x: abs(x - image_center))
            mode = "single_line"
            target = nearest_x
            xs_out = [nearest_x]
        else:
            mode = "none"
            target = None
            xs_out = []
        row_info.append((y, xs_out, target, mode))
    return row_info


def compute_lane_row_correction(row_info, image_center):
    """Compute notebook-style dist correction from filtered scan rows."""
    nearest_valid = next(
        (i for i in range(len(row_info) - 1, -1, -1) if row_info[i][1]),
        None,
    )
    nearest_dual = next(
        (i for i in range(len(row_info) - 1, -1, -1)
         if row_info[i][3] == "lane_center"),
        None,
    )

    if nearest_valid is None:
        return {
            "correction": 0,
            "mode": "lost",
            "nearest_valid": None,
            "nearest_dual": nearest_dual,
            "target_row": nearest_dual,
        }

    if nearest_dual is not None and nearest_valid == nearest_dual:
        correction = 0
        mode = "on_track"
    else:
        distance = (
            nearest_valid - nearest_dual
            if nearest_dual is not None
            else nearest_valid
        )
        nearest_x = row_info[nearest_valid][1][0]
        correction = distance if nearest_x < image_center else -distance
        mode = f"dist={distance}"

    return {
        "correction": correction,
        "mode": mode,
        "nearest_valid": nearest_valid,
        "nearest_dual": nearest_dual,
        "target_row": (
            nearest_dual if nearest_dual is not None else nearest_valid
        ),
    }


def compute_differential_speeds(
        base_speed, min_speed, correction, correction_gain=1,
        right_wheel_compensation=0, max_wheel_speed_delta=8,
        max_speed=40):
    """Map a signed lane correction to left/right wheel speeds."""
    speed_correction = correction * correction_gain
    if speed_correction >= 0:
        left_speed = base_speed + speed_correction
        right_speed = base_speed
    else:
        left_speed = base_speed
        right_speed = base_speed - speed_correction

    right_speed += right_wheel_compensation
    left_speed = max(min_speed, min(max_speed, left_speed))
    right_speed = max(min_speed, min(max_speed, right_speed))

    if abs(left_speed - right_speed) > max_wheel_speed_delta:
        if left_speed > right_speed:
            left_speed = min(right_speed + max_wheel_speed_delta, max_speed)
        else:
            right_speed = min(left_speed + max_wheel_speed_delta, max_speed)

    return left_speed, right_speed


def count_center_straddling_pairs(candidates, image_width, expected_width=None):
    """Count rows with a plausible left/right pair around the camera center."""
    expected_width = expected_width or image_width * 0.20
    left_points, right_points = select_center_side_lane_points(
        candidates, image_width, expected_width)
    count = 0
    centers = []
    for row_index, _ in enumerate(candidates):
        left = left_points.get(row_index)
        right = right_points.get(row_index)
        if left is None or right is None:
            continue
        count += 1
        centers.append((left + right) / 2.0)
    return count, centers


def _wide_runs_on_row(mask, y, min_run_width=3):
    row = mask[y] > 0
    runs = []
    start = None
    for idx, value in enumerate(row):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            width = idx - start
            if width >= min_run_width:
                runs.append({
                    "left": float(start),
                    "right": float(idx - 1),
                    "center": (start + idx - 1) / 2.0,
                    "width": float(width),
                })
            start = None
    if start is not None:
        width = len(row) - start
        if width >= min_run_width:
            runs.append({
                "left": float(start),
                "right": float(len(row) - 1),
                "center": (start + len(row) - 1) / 2.0,
                "width": float(width),
            })
    return runs


def analyze_center_lane_from_mask(mask, scan_ratios=None,
                                  center_tolerance_ratio=0.22,
                                  min_lane_width_ratio=0.20,
                                  max_lane_width_ratio=0.95,
                                  min_side_gap_ratio=0.03):
    """Find rows where yellow mask has left/right lane edges around center.

    This deliberately does not apply the normal max run width limit. After a
    turn, close lane borders can appear thick in the mask, but they are still
    useful evidence that the car is already between two lane lines.
    """
    if mask is None:
        return {"center_pair_rows": 0, "pairs": []}

    height, width = mask.shape[:2]
    if height <= 0 or width <= 0:
        return {"center_pair_rows": 0, "pairs": []}

    scan_ratios = scan_ratios or [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.78]
    image_center = width / 2.0
    min_lane_width = width * min_lane_width_ratio
    max_lane_width = width * max_lane_width_ratio
    center_tolerance = width * center_tolerance_ratio
    side_gap = width * min_side_gap_ratio
    pairs = []

    for ratio in scan_ratios:
        y = min(height - 1, max(0, int(height * ratio)))
        runs = _wide_runs_on_row(mask, y)
        left_runs = [
            run for run in runs
            if run["center"] < image_center - side_gap
        ]
        right_runs = [
            run for run in runs
            if run["center"] > image_center + side_gap
        ]
        if not left_runs or not right_runs:
            continue

        left = max(left_runs, key=lambda run: run["center"])
        right = min(right_runs, key=lambda run: run["center"])
        lane_width = right["center"] - left["center"]
        if lane_width < min_lane_width or lane_width > max_lane_width:
            continue

        midpoint = (left["center"] + right["center"]) / 2.0
        if abs(midpoint - image_center) > center_tolerance:
            continue

        pairs.append({
            "y": y,
            "left": left["center"],
            "right": right["center"],
            "center": midpoint,
            "width": lane_width,
        })

    return {
        "center_pair_rows": len(pairs),
        "pairs": pairs,
    }


def evaluate_lane_alignment(candidates, image_width, expected_width=None,
                            turn_bias=None, min_rows=4):
    """Return whether scan-line pairs describe a sufficiently straight lane."""
    image_center = image_width / 2.0
    expected_width = expected_width or image_width * 0.20
    row_info = build_lane_row_info(
        candidates,
        image_width=image_width,
        expected_width=expected_width,
        reference_center=image_center,
        turn_bias=turn_bias,
    )
    rows = [
        (float(y), xs[0], xs[1], target)
        for y, xs, target, mode in row_info
        if mode == "lane_center" and len(xs) >= 2
    ]

    result = {
        "aligned": False,
        "valid_rows": len(rows),
        "center_pair_rows": count_center_straddling_pairs(
            candidates,
            image_width=image_width,
            expected_width=expected_width,
        )[0],
        "mask_center_pair_rows": 0,
        "center_std": math.inf,
        "heading_delta": math.inf,
        "near_offset": math.inf,
        "width_cv": math.inf,
    }
    if len(rows) < min_rows:
        return result

    ys = [row[0] for row in rows]
    centers = [row[3] for row in rows]
    widths = [row[2] - row[1] for row in rows]
    mean_y = sum(ys) / len(ys)
    mean_center = sum(centers) / len(centers)
    variance_y = sum((y - mean_y) ** 2 for y in ys)
    slope = (
        sum((y - mean_y) * (center - mean_center)
            for y, center in zip(ys, centers)) / variance_y
        if variance_y > 0 else 0.0
    )
    heading_delta = abs(slope) * (max(ys) - min(ys))
    near_center = centers[ys.index(max(ys))]
    mean_width = sum(widths) / len(widths)
    width_cv = pstdev(widths) / mean_width if mean_width > 0 else math.inf
    center_std = pstdev(centers)
    near_offset = abs(near_center - image_center)

    result.update({
        "center_std": center_std,
        "heading_delta": heading_delta,
        "near_offset": near_offset,
        "width_cv": width_cv,
    })
    result["aligned"] = (
        center_std <= 65.0
        and heading_delta <= 80.0
        and near_offset <= 140.0
        and width_cv <= 0.40
    )
    return result
