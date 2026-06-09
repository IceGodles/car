#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detect a wide horizontal yellow boundary before the car reaches it."""

import numpy as np


def analyze_horizontal_boundary_mask(mask, min_y_ratio=0.56,
                                     stop_y_ratio=0.63,
                                     stop_coverage=0.60,
                                     warning_coverage=0.40):
    height, width = mask.shape
    start_y = max(0, min(height - 1, int(height * min_y_ratio)))
    row_coverage = np.count_nonzero(mask, axis=1) / float(width)
    best_y = start_y + int(np.argmax(row_coverage[start_y:]))
    coverage = float(row_coverage[best_y])
    y_ratio = best_y / float(height)
    return {
        "hit": coverage >= stop_coverage and y_ratio >= stop_y_ratio,
        "warning": coverage >= warning_coverage,
        "coverage": coverage,
        "y": best_y,
        "y_ratio": y_ratio,
    }


def detect_horizontal_boundary(frame, min_y_ratio=0.56,
                               stop_y_ratio=0.78,
                               stop_coverage=0.60,
                               warning_coverage=0.40):
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([12, 70, 80], dtype=np.uint8),
        np.array([45, 255, 255], dtype=np.uint8),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    result = analyze_horizontal_boundary_mask(
        mask,
        min_y_ratio=min_y_ratio,
        stop_y_ratio=stop_y_ratio,
        stop_coverage=stop_coverage,
        warning_coverage=warning_coverage,
    )
    result["mask"] = mask
    return result
