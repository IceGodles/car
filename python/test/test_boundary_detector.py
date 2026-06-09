#!/usr/bin/env python3

import numpy as np

from src.control.boundary_detector import analyze_horizontal_boundary_mask


def test_wide_line_triggers_early_stop():
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[470:480, 120:1160] = 255
    result = analyze_horizontal_boundary_mask(mask)
    assert result["hit"]
    assert result["coverage"] > 0.8
    assert 0.65 <= result["y_ratio"] <= 0.67


def test_wide_line_above_stop_zone_only_warns():
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[420:430, 120:1160] = 255
    result = analyze_horizontal_boundary_mask(mask)
    assert not result["hit"]
    assert result["warning"]
    assert result["y_ratio"] < 0.63


def test_longitudinal_lane_does_not_trigger_stop():
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[400:720, 280:360] = 255
    mask[400:720, 920:1000] = 255
    result = analyze_horizontal_boundary_mask(mask)
    assert not result["hit"]
    assert not result["warning"]


def test_medium_horizontal_line_only_warns():
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[440:450, 300:940] = 255
    result = analyze_horizontal_boundary_mask(mask)
    assert not result["hit"]
    assert result["warning"]


def test_short_horizontal_line_does_not_warn_below_threshold():
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[440:450, 420:780] = 255
    result = analyze_horizontal_boundary_mask(mask)
    assert not result["hit"]
    assert not result["warning"]
