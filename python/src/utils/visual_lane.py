#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版视觉巡迹模块
- 7条扫描线 + 加权投票（近处权重高）
- 自适应HSV阈值（根据亮度动态调整）
- PID低通滤波 + 速度自适应
- 弯道预测（基于扫描线趋势）
"""
import cv2
import numpy as np


class PIDController:
    """带低通滤波的PID控制器"""
    def __init__(self, kp=0.0012, ki=0.0, kd=0.0005, output_limit=0.6,
                 smooth_alpha=0.3):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.smooth_alpha = smooth_alpha
        self.integral = 0.0
        self.last_error = None
        self.smoothed_output = 0.0
        # 弯道模式下放大Kp
        self.curve_boost = 1.5

    def reset(self):
        self.integral = 0.0
        self.last_error = None
        self.smoothed_output = 0.0

    def update(self, error, is_curve=False):
        self.integral += error
        derivative = 0.0 if self.last_error is None else error - self.last_error
        self.last_error = error
        kp = self.kp * (self.curve_boost if is_curve else 1.0)
        output = kp * error + self.ki * self.integral + self.kd * derivative
        output = max(-self.output_limit, min(self.output_limit, output))
        # 低通滤波平滑输出，消除抖动
        self.smoothed_output = (self.smooth_alpha * output
                                + (1 - self.smooth_alpha) * self.smoothed_output)
        # 死区处理
        if abs(self.smoothed_output) < 0.03:
            return 0.0
        return self.smoothed_output


class YellowLaneFollower:
    def __init__(self, config=None):
        config = config or {}
        self.lower_yellow = np.array(config.get("lower_yellow", [12, 70, 80]),
                                     dtype=np.uint8)
        self.upper_yellow = np.array(config.get("upper_yellow", [45, 255, 255]),
                                     dtype=np.uint8)
        # 基础HSV阈值（用于自适应调整）
        self.base_lower_yellow = self.lower_yellow.copy()
        self.base_upper_yellow = self.upper_yellow.copy()
        self.adaptive_hsv = config.get("adaptive_hsv", True)

        self.roi_top_ratio = config.get("roi_top_ratio", 0.35)
        # 7条扫描线，覆盖更多视野
        self.scan_ratios = config.get("scan_ratios",
                                       [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.78])
        # 加权投票：近处权重高（数组末尾=近处）
        self.scan_weights = config.get("scan_weights",
                                        [0.05, 0.10, 0.15, 0.25, 0.20, 0.15, 0.10])
        self.min_run_width = config.get("min_run_width", 4)
        self.max_run_width = config.get("max_run_width", 180)
        self.lane_width_ratio = config.get("lane_width_ratio", 0.20)
        self.single_line_gain = config.get("single_line_gain", 0.25)
        self.center_deadband_px = config.get("center_deadband_px", 20)
        self.max_lost_frames = config.get("max_lost_frames", 10)
        self.pid = PIDController(
            kp=config.get("kp", 0.0012),
            ki=config.get("ki", 0.0),
            kd=config.get("kd", 0.0005),
            output_limit=config.get("output_limit", 0.6),
            smooth_alpha=config.get("smooth_alpha", 0.3),
        )
        self.last_target_x = None
        self.last_reference_side = None
        self.lost_frames = 0
        self._error_history = []  # 用于弯道预测

    def _adaptive_hsv(self, frame):
        """根据图像亮度动态调整HSV阈值"""
        if not self.adaptive_hsv:
            return self.lower_yellow, self.upper_yellow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        # 亮度高时收窄饱和度下限（减少白色反光干扰）
        # 亮度低时放宽亮度下限（暗处也能检测黄色）
        s_min = int(np.clip(70 - (brightness - 128) * 0.3, 40, 100))
        v_min = int(np.clip(80 - (brightness - 128) * 0.5, 40, 120))
        lower = np.array([12, s_min, v_min], dtype=np.uint8)
        upper = np.array([45, 255, 255], dtype=np.uint8)
        return lower, upper

    def segment_yellow(self, frame):
        lower, upper = self._adaptive_hsv(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        height, width = mask.shape
        roi_top = int(height * self.roi_top_ratio)
        mask[:roi_top, :] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def _runs_on_row(self, mask, y):
        row = mask[y] > 0
        runs = []
        start = None
        for idx, value in enumerate(row):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                width = idx - start
                if self.min_run_width <= width <= self.max_run_width:
                    runs.append((start + idx - 1) / 2.0)
                start = None
        if start is not None:
            width = len(row) - start
            if self.min_run_width <= width <= self.max_run_width:
                runs.append((start + len(row) - 1) / 2.0)
        return runs

    def _collect_candidates(self, mask):
        height, _ = mask.shape
        candidates = []
        for ratio in self.scan_ratios:
            y = min(height - 1, max(0, int(height * ratio)))
            xs = self._runs_on_row(mask, y)
            candidates.append((y, xs))
        return candidates

    def _choose_target_from_row(self, xs, image_center, lane_width_px):
        """选择目标点 — 核心逻辑：
        - 有两个以上分散点 → 取最左最右的中点（车道中心）
        - 只有一个点 → 向中心偏移（回到车道中间）
        - 无点 → 返回None
        """
        if not xs:
            return None, None

        xs = sorted(xs)

        if len(xs) >= 2:
            leftmost = xs[0]
            rightmost = xs[-1]
            spread = rightmost - leftmost

            # 两点分散足够大 → 这是左右两条车道线，中点就是车道中心
            if spread > lane_width_px * 0.3:
                midpoint = (leftmost + rightmost) / 2.0
                # 异常点过滤：如果中点太偏，可能是误检
                if abs(midpoint - image_center) < image_center * 0.8:
                    return midpoint, "lane_center"

            # 多个点聚集在一侧 → 弯道特征，用加权中心
            center_of_mass = float(np.mean(xs))
            return center_of_mass, "cluster"

        # 只有一个点 → 向图像中心偏移，试图回到车道中间
        single = xs[0]
        side = "left" if single < image_center else "right"
        # 偏移量：离中心越远，回正力度越大
        offset_ratio = self.single_line_gain
        target = image_center + (single - image_center) * offset_ratio
        return target, f"single_{side}"

    def _detect_curve(self):
        """基于误差历史趋势判断是否进入弯道"""
        if len(self._error_history) < 3:
            return False
        recent = self._error_history[-5:]
        # 如果最近几帧误差持续增大且方向一致，判定为弯道
        if len(recent) >= 3:
            signs = [1 if e > 0 else -1 for e in recent]
            if len(set(signs)) == 1 and abs(recent[-1]) > 15:
                # 误差持续同向增大
                diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
                if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
                    return True
        return False

    def _predict_curve_speed(self, is_curve, error):
        """根据弯道强度自适应速度"""
        if is_curve:
            # 弯道越急，速度越低
            curve_intensity = min(abs(error) / 100.0, 1.0)
            return max(0.5, 1.0 - curve_intensity * 0.4)
        return 1.0

    def infer(self, frame):
        height, width = frame.shape[:2]
        image_center = width / 2.0
        lane_width_px = width * self.lane_width_ratio

        mask = self.segment_yellow(frame)
        candidates = self._collect_candidates(mask)

        # === 两遍处理：先算可靠中线，再用中线修正单点 ===

        # 第1遍：收集所有行的检测结果
        row_results = []
        for i, (y, xs) in enumerate(candidates):
            target_x, mode = self._choose_target_from_row(xs, image_center, lane_width_px)
            row_results.append((y, xs, target_x, mode, i))

        # 第2遍：用多点行的中线修正单点行
        # 可靠中线的加权平均
        lane_center = None
        lc_weights = []
        lc_values = []
        for i, (y, xs) in enumerate(candidates):
            if len(xs) >= 2:
                leftmost, rightmost = xs[0], xs[-1]
                spread = rightmost - leftmost
                if spread > lane_width_px * 0.3:
                    mid = (leftmost + rightmost) / 2.0
                    lc_values.append(mid)
                    lc_weights.append(self.scan_weights[i] if i < len(self.scan_weights) else 0.1)
        if lc_values:
            lane_center = float(np.average(lc_values, weights=lc_weights))

        # 第2遍：找可靠的双点线 + 插值预测路径
        reliable_rows = []  # (y, midpoint, weight)
        for i, (y, xs) in enumerate(candidates):
            if len(xs) >= 2:
                leftmost, rightmost = xs[0], xs[-1]
                spread = rightmost - leftmost
                if spread > lane_width_px * 0.3:
                    mid = (leftmost + rightmost) / 2.0
                    w = self.scan_weights[i] if i < len(self.scan_weights) else 0.1
                    reliable_rows.append((y, mid, w))

        if len(reliable_rows) >= 2:
            # 有多条可靠线 → 插值预测路径
            rys = np.array([r[0] for r in reliable_rows])
            rxs = np.array([r[1] for r in reliable_rows])
            rws = np.array([r[2] for r in reliable_rows])

            # 按y排序（远→近），拟合二次曲线
            order = np.argsort(rys)
            rys_sorted = rys[order]
            rxs_sorted = rxs[order]
            degree = min(2, len(rys_sorted) - 1)
            coeffs = np.polyfit(rys_sorted, rxs_sorted, degree)

            # 找离车最近的可靠线y值（最大y），向前看一个步长
            nearest_y = rys_sorted[-1]
            step = abs(rys_sorted[-1] - rys_sorted[-2]) if len(rys_sorted) > 2 else 24
            lookahead_y = nearest_y - step  # 往前一行
            lookahead_y = max(0, min(height - 1, int(lookahead_y)))

            # 插值得到lookahead_x
            lookahead_x = float(np.polyval(coeffs, lookahead_y))

            # 如果还有更远的可靠线，也加入权重
            all_targets = []
            all_weights = []
            for ry, rx, rw in reliable_rows:
                all_targets.append(rx)
                all_weights.append(rw)
            # lookahead点权重加倍
            all_targets.append(lookahead_x)
            all_weights.append(max(all_weights) * 1.5 if all_weights else 0.15)

            target_x = float(np.average(all_targets, weights=all_weights))
            mode = "interpolated"

        elif len(reliable_rows) == 1:
            # 只有一条可靠线 → 用它作为目标
            target_x = reliable_rows[0][1]
            mode = "single_reliable"

        else:
            # 没有可靠双点线 → 用单点修正
            target_x = lane_center if lane_center is not None else image_center
            mode = "fallback"

        # target_x 已在上面设置
        # 记录lookahead_y（用于绘制和控制）
        if 'lookahead_y' not in dir():
            lookahead_y = int(height * 0.72)

        if target_x is not None:
            self.last_target_x = target_x
            self.lost_frames = 0
        elif (self.last_target_x is not None
              and self.lost_frames < self.max_lost_frames):
            target_x = self.last_target_x
            self.lost_frames += 1
        else:
            self.pid.reset()
            self.lost_frames += 1
            self._error_history.clear()
            return {
                "ok": False,
                "mask": mask,
                "target_x": None,
                "error": 0.0,
                "pid": 0.0,
                "speed_factor": 1.0,
                "mode": "lost",
                "candidates": candidates,
                "is_curve": False,
                "lookahead_y": int(height * 0.72),
            }

        error = target_x - image_center
        self._error_history.append(error)
        if len(self._error_history) > 20:
            self._error_history.pop(0)

        is_curve = self._detect_curve()
        pid_output = 0.0
        if abs(error) > self.center_deadband_px:
            pid_output = self.pid.update(error, is_curve=is_curve)
        else:
            self.pid.update(0, is_curve=False)

        speed_factor = self._predict_curve_speed(is_curve, error)

        return {
            "ok": True,
            "mask": mask,
            "target_x": target_x,
            "error": error,
            "pid": pid_output,
            "speed_factor": speed_factor,
            "mode": mode if 'mode' in dir() else "unknown",
            "candidates": candidates,
            "is_curve": is_curve,
            "lookahead_y": lookahead_y,
        }

    def draw_debug(self, frame, result):
        vis = frame.copy()
        height, width = vis.shape[:2]
        center_x = width // 2
        cv2.line(vis, (center_x, int(height * self.roi_top_ratio)),
                 (center_x, height), (255, 255, 255), 2)

        for y, xs in result.get("candidates", []):
            cv2.line(vis, (0, y), (width, y), (80, 80, 80), 1)
            for x in xs:
                cv2.circle(vis, (int(x), y), 7, (0, 255, 255), -1)

        target_x = result.get("target_x")
        lk_y = result.get("lookahead_y", int(height * 0.72))
        if target_x is not None:
            cv2.circle(vis, (int(target_x), lk_y), 12, (0, 0, 255), -1)
            cv2.line(vis, (center_x, height),
                     (int(target_x), lk_y), (0, 0, 255), 3)

        text = "ok={} mode={} err={:.1f} pid={:.3f} curve={} spd={:.2f}".format(
            result.get("ok"), result.get("mode"), result.get("error", 0.0),
            result.get("pid", 0.0), result.get("is_curve", False),
            result.get("speed_factor", 1.0),
        )
        cv2.putText(vis, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)
        return vis
