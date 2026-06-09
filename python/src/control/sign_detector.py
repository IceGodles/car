#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO低频标志检测封装
- 每N帧运行一次YOLO（默认6帧≈5fps）
- 缓存检测结果供巡迹主线程使用
"""
import time

import numpy as np


def _path_crosses_yellow(mask, start, end, radius=2, min_run=4):
    """
    Return whether the segment from start to end crosses a yellow mask.

    A short consecutive run is required so isolated HSV noise does not reject
    an otherwise valid target.
    """
    if mask is None or mask.ndim != 2 or mask.size == 0:
        return False, None

    height, width = mask.shape
    x0, y0 = start
    x1, y1 = end
    sample_count = max(abs(int(x1) - int(x0)), abs(int(y1) - int(y0))) + 1
    if sample_count <= 1:
        return False, None

    xs = np.linspace(x0, x1, sample_count).astype(np.int32)
    ys = np.linspace(y0, y1, sample_count).astype(np.int32)
    consecutive = 0
    first_crossing = None

    for x, y in zip(xs, ys):
        if x < 0 or x >= width or y < 0 or y >= height:
            consecutive = 0
            first_crossing = None
            continue
        x_start = max(0, x - radius)
        x_end = min(width, x + radius + 1)
        y_start = max(0, y - radius)
        y_end = min(height, y + radius + 1)
        if np.any(mask[y_start:y_end, x_start:x_end] > 0):
            if consecutive == 0:
                first_crossing = (int(x), int(y))
            consecutive += 1
            if consecutive >= min_run:
                return True, first_crossing
        else:
            consecutive = 0
            first_crossing = None

    return False, None


def filter_signs_by_lane_mask(sign_result, yellow_mask,
                              path_radius=2, min_crossing_run=4):
    """Reject targets whose camera-to-target path crosses a yellow lane line."""
    if not sign_result or yellow_mask is None:
        return sign_result

    if yellow_mask.ndim != 2 or yellow_mask.size == 0:
        return sign_result

    height, width = yellow_mask.shape
    start = (width // 2, height - 1)
    decisions = []

    def keep(sign):
        x = sign.get("x")
        y = sign.get("y")
        if x is None or y is None:
            decisions.append({
                "type": sign.get("type"),
                "bbox": sign.get("bbox"),
                "target": None,
                "crossed": True,
                "crossing": None,
            })
            return False
        target = (int(round(x)), int(round(y)))
        crossed, crossing = _path_crosses_yellow(
            yellow_mask,
            start,
            target,
            radius=path_radius,
            min_run=min_crossing_run,
        )
        decisions.append({
            "type": sign.get("type"),
            "bbox": sign.get("bbox"),
            "target": target,
            "crossed": crossed,
            "crossing": crossing,
        })
        return not crossed

    filtered_signs = [s for s in sign_result.get("signs", []) if keep(s)]
    filtered_stop_signs = [s for s in sign_result.get("stop_signs", []) if keep(s)]

    filtered = dict(sign_result)
    filtered["signs"] = filtered_signs
    filtered["stop_signs"] = filtered_stop_signs
    filtered["stop"] = bool(filtered_stop_signs)
    filtered["lane_filter"] = {
        "enabled": True,
        "origin": start,
        "path_radius": path_radius,
        "min_crossing_run": min_crossing_run,
        "signs_before": len(sign_result.get("signs", [])),
        "signs_after": len(filtered_signs),
        "stop_before": len(sign_result.get("stop_signs", [])),
        "stop_after": len(filtered_stop_signs),
        "decisions": decisions,
    }
    return filtered


class SignDetector:
    def __init__(self, yolo_model, config=None):
        config = config or {}
        self.det = yolo_model
        self.detect_interval = config.get("detect_interval", 6)  # 每N帧检测一次
        self.roi_keep_ratio = config.get("roi_keep_ratio", 0.6)  # 下半部分ROI
        self.conf_threshold = config.get("conf_threshold", 0.85)
        # 标志牌在画面中的触发区域（底部区域）
        self.sign_trigger_y_min = config.get("sign_trigger_y_min", 0.45)
        self.sign_trigger_x_range = config.get("sign_trigger_x_range", (0.15, 0.85))

        self._frame_count = 0
        self._last_result = {
            "signs": [],
            "stop": False,
            "stop_signs": [],
            "all_bboxes": [],
            "fresh": False,
        }
        self._last_detect_time = 0.0

    def _bottom_roi(self, img_bgr):
        height = img_bgr.shape[0]
        roi_top = int(height * (1.0 - self.roi_keep_ratio))
        return img_bgr[roi_top:].copy(), roi_top

    def _map_bboxes(self, bboxes, y_offset):
        mapped = []
        for x1, y1, x2, y2, cate, score in bboxes:
            mapped.append([x1, y1 + y_offset, x2, y2 + y_offset, cate, score])
        return mapped

    def detect(self, img_bgr, force=False):
        """
        检测标志牌，按间隔跳帧
        返回: {"signs": [...], "stop": bool, "all_bboxes": [...]}
        """
        self._frame_count += 1
        if not force and self._frame_count % self.detect_interval != 0:
            cached = dict(self._last_result)
            cached["fresh"] = False
            return cached

        roi_bgr, roi_top = self._bottom_roi(img_bgr)
        raw_bboxes = self.det.infer(roi_bgr)
        bboxes = self._map_bboxes(raw_bboxes, roi_top)

        height, width = img_bgr.shape[:2]
        signs = []
        stop_signs = []
        stop_detected = False

        for x1, y1, x2, y2, cate, score in bboxes:
            if float(score) < self.conf_threshold:
                continue
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            # 检查是否在触发区域内（画面下半部分）
            in_trigger_y = cy >= height * self.sign_trigger_y_min
            in_trigger_x = (width * self.sign_trigger_x_range[0]
                            <= cx <= width * self.sign_trigger_x_range[1])

            if in_trigger_y and in_trigger_x:
                if cate == "stop":
                    stop_signs.append({
                        "type": cate,
                        "x": cx,
                        "y": cy,
                        "bbox": [x1, y1, x2, y2],
                        "score": float(score),
                    })
                    stop_detected = True
                elif cate in ("left", "right", "turnaround"):
                    signs.append({
                        "type": cate,
                        "x": cx,
                        "y": cy,
                        "bbox": [x1, y1, x2, y2],
                        "score": float(score),
                    })

        self._last_result = {
            "signs": signs,
            "stop": stop_detected,
            "stop_signs": stop_signs,
            "all_bboxes": bboxes,
            "fresh": True,
        }
        self._last_detect_time = time.time()
        return self._last_result
