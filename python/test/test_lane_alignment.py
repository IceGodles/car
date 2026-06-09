#!/usr/bin/env python3

import numpy as np

from src.control.lane_alignment import (
    analyze_center_lane_from_mask,
    build_lane_row_info,
    compute_lane_row_correction,
    count_center_straddling_pairs,
    evaluate_lane_alignment,
    select_center_side_lane_points,
    select_lane_pair,
    select_lane_path,
)


def test_select_lane_pair_does_not_span_three_lines():
    pair = select_lane_pair(
        [32, 572, 1078],
        image_center=640,
        expected_width=500,
        reference_center=700,
        turn_bias="right",
    )
    assert pair[:2] == (572.0, 1078.0)


def test_lane_path_locks_pair_around_camera_on_near_row():
    candidates = [
        (324, [80, 470, 850, 1190]),
        (396, [90, 490, 880]),
        (468, [100, 500, 900]),
    ]

    path = select_lane_path(
        candidates,
        image_width=1280,
        expected_width=400,
    )

    assert path[468][:2] == (500.0, 900.0)
    assert path[396][:2] == (490.0, 880.0)
    assert path[324][:2] == (470.0, 850.0)


def test_lane_path_ignores_far_branch_center_jump():
    candidates = [
        (324, [40, 440, 1180]),
        (396, [120, 500, 900]),
        (468, [140, 520, 920]),
    ]

    path = select_lane_path(
        candidates,
        image_width=1280,
        expected_width=400,
        max_center_jump=80,
    )

    assert path[468][:2] == (520.0, 920.0)
    assert path[396][:2] == (500.0, 900.0)
    assert 324 not in path


def test_row_info_keeps_near_single_line_for_dist_correction():
    candidates = [
        (324, [360, 920]),
        (360, [350, 930]),
        (396, [410]),
        (432, [405]),
        (468, [400]),
    ]

    row_info = build_lane_row_info(
        candidates,
        image_width=1280,
        expected_width=280,
    )

    nearest_valid = max(i for i, row in enumerate(row_info) if row[1])
    nearest_dual = max(i for i, row in enumerate(row_info)
                       if row[3] == "lane_center")

    assert row_info[nearest_dual][3] == "lane_center"
    assert row_info[nearest_valid][3] == "single_line"
    assert nearest_valid - nearest_dual == 3
    assert row_info[nearest_valid][1][0] < 640


def test_single_line_without_lane_pair_still_corrects():
    row_info = [
        (324, [], None, "none"),
        (360, [], None, "none"),
        (396, [], None, "none"),
        (432, [], None, "none"),
        (468, [], None, "none"),
        (504, [871.0], 871.0, "single_line"),
        (561, [], None, "none"),
    ]

    result = compute_lane_row_correction(row_info, image_center=550.0)

    assert result["mode"] == "dist=5"
    assert result["correction"] == -5
    assert result["target_row"] == 5


def test_center_side_points_ignore_far_third_line():
    candidates = [
        (324, [40, 430, 890]),
        (360, [60, 450, 910]),
        (396, [80, 470, 930]),
        (432, [100, 490, 950]),
    ]

    row_info = build_lane_row_info(
        candidates,
        image_width=1280,
        expected_width=420,
    )

    assert row_info[0][1] == [430.0, 890.0]
    assert row_info[1][1] == [450.0, 910.0]
    assert all(mode == "lane_center" for _, _, _, mode in row_info)


def test_single_points_are_kept_when_no_line_has_three_points():
    candidates = [
        (324, [80]),
        (360, []),
        (396, [900]),
    ]

    left, right = select_center_side_lane_points(
        candidates,
        image_width=1280,
        expected_width=420,
    )

    assert left == {0: 80.0}
    assert right == {2: 900.0}


def test_counts_center_straddling_pairs_for_post_turn_exit():
    candidates = [
        (324, [180, 675, 940]),
        (360, [126, 700, 944]),
        (396, [72, 728, 996]),
        (432, [26, 755]),
        (468, [914]),
    ]

    count, centers = count_center_straddling_pairs(
        candidates,
        image_width=1100,
        expected_width=360,
    )

    assert count == 4
    assert len(centers) == 4


def test_mask_center_pairs_accept_wide_lane_runs():
    mask = np.zeros((720, 1280), dtype=np.uint8)
    for ratio in [0.45, 0.50, 0.55, 0.60]:
        y = int(720 * ratio)
        # The right run is intentionally wider than visual_lane.max_run_width.
        mask[y, 120:380] = 255
        mask[y, 900:1220] = 255

    result = analyze_center_lane_from_mask(
        mask,
        scan_ratios=[0.45, 0.50, 0.55, 0.60],
    )

    assert result["center_pair_rows"] == 4
    assert len(result["pairs"]) == 4


def test_straight_lane_is_aligned():
    candidates = [
        (324, [400, 880]),
        (360, [390, 890]),
        (396, [380, 900]),
        (432, [370, 910]),
        (468, [360, 920]),
    ]
    result = evaluate_lane_alignment(candidates, image_width=1280)
    assert result["aligned"]
    assert result["valid_rows"] == 5


def test_diagonal_lane_needs_more_turn():
    candidates = [
        (324, [100, 600]),
        (360, [160, 660]),
        (396, [220, 720]),
        (432, [280, 780]),
        (468, [340, 840]),
    ]
    result = evaluate_lane_alignment(candidates, image_width=1280)
    assert not result["aligned"]
    assert result["heading_delta"] > 120


def test_moderate_right_turn_is_not_aligned_yet():
    candidates = [
        (324, [460, 964]),
        (360, [490, 1010]),
        (396, [520, 1044]),
        (432, [550, 1074]),
    ]
    result = evaluate_lane_alignment(
        candidates,
        image_width=1280,
        turn_bias="right",
    )
    assert not result["aligned"]
    assert result["heading_delta"] > 80 or result["near_offset"] > 140


def test_too_few_lane_pairs_is_not_aligned():
    candidates = [
        (324, [400, 880]),
        (360, [390]),
        (396, []),
        (432, [370, 910]),
    ]
    result = evaluate_lane_alignment(candidates, image_width=1280)
    assert not result["aligned"]
    assert result["valid_rows"] == 2
