#!/usr/bin/env python3

import numpy as np

from src.control.sign_detector import filter_signs_by_lane_mask


def _sign(sign_type, x, y):
    return {
        "type": sign_type,
        "x": x,
        "y": y,
        "bbox": [x - 20, y - 20, x + 20, y + 20],
        "score": 0.95,
    }


def test_lane_filter_keeps_target_without_yellow_crossing():
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[:, 300:310] = 255
    sign_result = {
        "signs": [_sign("left", 700, 350)],
        "stop": False,
        "stop_signs": [],
    }

    filtered = filter_signs_by_lane_mask(sign_result, mask)

    assert [sign["type"] for sign in filtered["signs"]] == ["left"]
    assert not filtered["lane_filter"]["decisions"][0]["crossed"]


def test_lane_filter_rejects_target_across_yellow_lane_line():
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[:, 790:800] = 255
    sign_result = {
        "signs": [_sign("right", 980, 320)],
        "stop": False,
        "stop_signs": [],
    }

    filtered = filter_signs_by_lane_mask(sign_result, mask)

    assert filtered["signs"] == []
    assert filtered["lane_filter"]["decisions"][0]["crossed"]
    assert filtered["lane_filter"]["decisions"][0]["crossing"] is not None


def test_lane_filter_ignores_isolated_yellow_noise():
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[500, 700] = 255
    sign_result = {
        "signs": [_sign("left", 800, 300)],
        "stop": False,
        "stop_signs": [],
    }

    filtered = filter_signs_by_lane_mask(sign_result, mask)

    assert [sign["type"] for sign in filtered["signs"]] == ["left"]


def test_lane_filter_applies_to_stop_signs():
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[:, 480:490] = 255
    sign_result = {
        "signs": [],
        "stop": True,
        "stop_signs": [_sign("stop", 300, 320)],
    }

    filtered = filter_signs_by_lane_mask(sign_result, mask)

    assert filtered["stop_signs"] == []
    assert filtered["stop"] is False
