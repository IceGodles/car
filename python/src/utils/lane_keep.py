#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


@dataclass
class LaneKeepConfig:
    yellow_lower: Tuple[int, int, int] = (15, 45, 80)
    yellow_upper: Tuple[int, int, int] = (45, 255, 255)
    road_lower: Tuple[int, int, int] = (0, 0, 0)
    road_upper: Tuple[int, int, int] = (179, 120, 220)
    roi_top_ratio: float = 0.12
    sample_step: int = 8
    min_row_pixels: int = 60
    polynomial_degree: int = 2
    morph_kernel: int = 5
    lookahead_ratio: float = 0.72
    center_deadband_px: int = 24
    max_lost_frames: int = 8
    yellow_prefer_min_area: int = 3000
    road_max_area_ratio: float = 0.72
    kp: float = 0.0012
    ki: float = 0.0
    kd: float = 0.0005
    output_limit: float = 0.65


class LaneKeepPID:
    def __init__(self, kp=0.0012, ki=0.0, kd=0.0005, output_limit=0.65):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral = 0.0
        self.last_error = None

    def reset(self):
        self.integral = 0.0
        self.last_error = None

    def update(self, error):
        self.integral += error
        derivative = 0.0 if self.last_error is None else error - self.last_error
        self.last_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(-self.output_limit, min(self.output_limit, output))


class LaneKeepFollower:
    def __init__(self, config=None):
        self.cfg = config or LaneKeepConfig()
        self.pid = LaneKeepPID(
            kp=self.cfg.kp,
            ki=self.cfg.ki,
            kd=self.cfg.kd,
            output_limit=self.cfg.output_limit,
        )
        self.last_target_x = None
        self.last_centerline = None
        self.lost_frames = 0

    def _build_mask(self, hsv, lower, upper):
        mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        if self.cfg.morph_kernel > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.cfg.morph_kernel, self.cfg.morph_kernel),
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    @staticmethod
    def _largest_contour_mask(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        contour = max(contours, key=cv2.contourArea)
        output = np.zeros_like(mask)
        cv2.drawContours(output, [contour], -1, 255, thickness=cv2.FILLED)
        return output

    def _fit_centerline_from_mask(self, mask):
        height, width = mask.shape[:2]
        y_start = int(height * self.cfg.roi_top_ratio)
        y_end = int(height * 0.94)
        points: List[Point] = []

        for y in range(y_start, y_end, self.cfg.sample_step):
            xs = np.flatnonzero(mask[y] > 0)
            if xs.size < self.cfg.min_row_pixels:
                continue
            center_x = float((xs[0] + xs[-1]) / 2.0)
            points.append((center_x, float(y)))

        if len(points) < 3:
            return points, None

        ys = np.array([p[1] for p in points], dtype=np.float32)
        xs = np.array([p[0] for p in points], dtype=np.float32)
        degree = min(self.cfg.polynomial_degree, len(points) - 1)
        coeffs = np.polyfit(ys, xs, degree)
        fitted_ys = np.linspace(ys.min(), ys.max(), num=120)
        fitted_xs = np.polyval(coeffs, fitted_ys)

        fitted = []
        for x, y in zip(fitted_xs, fitted_ys):
            xi = int(round(float(np.clip(x, 0, width - 1))))
            yi = int(round(float(np.clip(y, 0, height - 1))))
            fitted.append([xi, yi])
        return points, np.array(fitted, dtype=np.int32)

    @staticmethod
    def _x_at_y(centerline, y_value):
        if centerline is None or len(centerline) == 0:
            return None
        points = centerline.astype(np.float32)
        ys = points[:, 1]
        idx = int(np.searchsorted(ys, y_value, side="left"))
        if idx <= 0:
            return float(points[0, 0])
        if idx >= len(points):
            return float(points[-1, 0])
        p0 = points[idx - 1]
        p1 = points[idx]
        if p1[1] == p0[1]:
            return float(p1[0])
        ratio = (y_value - p0[1]) / (p1[1] - p0[1])
        return float(p0[0] + ratio * (p1[0] - p0[0]))

    def infer(self, frame):
        height, width = frame.shape[:2]
        image_center = width / 2.0
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_mask = self._build_mask(hsv, self.cfg.yellow_lower, self.cfg.yellow_upper)
        road_mask = self._build_mask(hsv, self.cfg.road_lower, self.cfg.road_upper)

        roi_top = int(height * self.cfg.roi_top_ratio)
        yellow_mask[:roi_top, :] = 0
        road_mask[:roi_top, :] = 0

        road_mask = self._largest_contour_mask(road_mask)
        road_area = int(cv2.countNonZero(road_mask))
        yellow_area = int(cv2.countNonZero(yellow_mask))
        roi_area = int((height - roi_top) * width)
        road_too_large = road_area > roi_area * self.cfg.road_max_area_ratio
        if yellow_area >= self.cfg.yellow_prefer_min_area:
            mask = yellow_mask
            mode = "yellow"
        elif road_area >= 2000 and not road_too_large:
            mask = road_mask
            mode = "road"
        else:
            mask = yellow_mask if yellow_area > 0 else road_mask
            mode = "yellow_fallback" if yellow_area > 0 else "road_fallback"

        center_points, centerline = self._fit_centerline_from_mask(mask)
        lookahead_y = int(height * self.cfg.lookahead_ratio)
        target_x = self._x_at_y(centerline, lookahead_y)

        if target_x is not None:
            self.last_target_x = target_x
            self.last_centerline = centerline
            self.lost_frames = 0
        elif self.last_target_x is not None and self.lost_frames < self.cfg.max_lost_frames:
            target_x = self.last_target_x
            centerline = self.last_centerline
            self.lost_frames += 1
            mode = "history"
        else:
            self.pid.reset()
            self.lost_frames += 1
            return {
                "ok": False,
                "mode": "lost",
                "mask": mask,
                "yellow_mask": yellow_mask,
                "road_mask": road_mask,
                "center_points": center_points,
                "centerline": centerline,
                "lookahead_y": lookahead_y,
                "target_x": None,
                "error": 0.0,
                "pid": 0.0,
                "road_area": road_area,
                "yellow_area": yellow_area,
            }

        error = target_x - image_center
        pid_output = 0.0 if abs(error) <= self.cfg.center_deadband_px else self.pid.update(error)
        return {
            "ok": True,
            "mode": mode,
            "mask": mask,
            "yellow_mask": yellow_mask,
            "road_mask": road_mask,
            "center_points": center_points,
            "centerline": centerline,
            "lookahead_y": lookahead_y,
            "target_x": target_x,
            "error": error,
            "pid": pid_output,
            "road_area": road_area,
            "yellow_area": yellow_area,
        }

    def draw_debug(self, frame, result):
        vis = frame.copy()
        height, width = vis.shape[:2]
        center_x = width // 2
        roi_top = int(height * self.cfg.roi_top_ratio)
        lookahead_y = int(result.get("lookahead_y", height * self.cfg.lookahead_ratio))

        cv2.line(vis, (center_x, roi_top), (center_x, height), (255, 255, 255), 2)
        cv2.line(vis, (0, lookahead_y), (width, lookahead_y), (255, 0, 255), 2)

        centerline = result.get("centerline")
        if centerline is not None and len(centerline) >= 2:
            cv2.polylines(vis, [centerline], False, (0, 0, 255), 4, lineType=cv2.LINE_AA)
            step = max(1, len(centerline) // 15)
            for point in centerline[::step]:
                cv2.circle(vis, tuple(int(v) for v in point), 4, (0, 255, 255), -1)

        for x, y in result.get("center_points", [])[::4]:
            cv2.circle(vis, (int(x), int(y)), 3, (255, 255, 0), -1)

        target_x = result.get("target_x")
        if target_x is not None:
            cv2.circle(vis, (int(target_x), lookahead_y), 12, (0, 0, 255), -1)
            cv2.line(vis, (center_x, height), (int(target_x), lookahead_y), (0, 0, 255), 3)

        text = "ok={} mode={} error={:.1f} pid={:.3f} road={} yellow={}".format(
            result.get("ok"),
            result.get("mode"),
            result.get("error", 0.0),
            result.get("pid", 0.0),
            result.get("road_area", 0),
            result.get("yellow_area", 0),
        )
        cv2.putText(vis, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return vis
