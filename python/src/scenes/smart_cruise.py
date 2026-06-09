#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能巡航场景 — 独立进程运行
算法完全对齐 algorithm_analysis.ipynb Step3-5
"""
import math
import os
import time

import cv2
import numpy as np

from src.actions import (Advance, SetServo, SpinAntiClockwise,
                         SpinClockwise, Stop, TurnLeft, TurnRight)
from src.actions.complex_actions import TurnLeftInPlace, TurnRightInPlace, TurnAround
from src.control.boundary_detector import detect_horizontal_boundary
from src.control.lane_alignment import (
    analyze_center_lane_from_mask,
    build_lane_row_info,
    compute_differential_speeds,
    compute_lane_row_correction,
    evaluate_lane_alignment,
)
from src.control.sign_detector import filter_signs_by_lane_mask
from src.scenes.base_scene import BaseScene
from src.utils import log
from src.utils.constant import CAMERA_SERVO_ANGLE
from src.utils.visual_lane import YellowLaneFollower
from src.control.state_machine import DrivingStateMachine


class SmartCruise(BaseScene):
    def __init__(self, memory_name, camera_info, msg_queue):
        super().__init__(memory_name, camera_info, msg_queue)
        self.follower = None
        self.sm = None
        self.det = None
        self._lane_reference_center = None
        self._post_turn_align = False
        self._alignment_turn = None
        self._last_turn_direction = None
        self._alignment_steps = 0
        self._alignment_streak = 0
        self._alignment_max_steps = 3
        self._alignment_step_seconds = 0.18
        self._alignment_settle_seconds = 0.05
        self._boundary_escape = False
        self._boundary_warning_mode = False
        self._boundary_escape_direction = "right"
        self._boundary_escape_steps = 0
        self._boundary_escape_streak = 0
        self._boundary_escape_max_steps = 14
        self._boundary_escape_speed = 42
        self._boundary_escape_step_seconds = 0.10
        self._boundary_escape_settle_seconds = 0.20
        self._right_wheel_compensation = 1
        self._lane_correction_gain = 2
        self._max_wheel_speed_delta = 16

    def init_state(self):
        log.info(f'start init {self.__class__.__name__}')

        # 巡迹模块
        self.follower = YellowLaneFollower({
            "adaptive_hsv": True,
            "scan_ratios": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.78],
            "scan_weights": [0.05, 0.10, 0.15, 0.25, 0.20, 0.15, 0.10],
            "single_line_gain": 0.25,
            "center_deadband_px": 20,
        })
        log.info(f'  巡迹模块 OK')

        # 状态机（仅用于 PAUSE / TURN / STOP 状态管理）
        self.sm = DrivingStateMachine()
        log.info(f'  状态机 OK')

        # YOLO
        yolo_candidates = [
            os.path.join(os.getcwd(), 'weights', 'best.om'),
            os.path.join(os.getcwd(), 'weights', 'yolo.om'),
        ]
        for yolo_path in yolo_candidates:
            if not os.path.exists(yolo_path):
                continue
            try:
                from src.models import YoloV5
                from src.control.sign_detector import SignDetector
                model = YoloV5(yolo_path)
                self.det = SignDetector(model, {
                    "detect_interval": 3, "roi_keep_ratio": 0.6, "conf_threshold": 0.85,
                })
                log.info(f'  YOLO OK ({os.path.basename(yolo_path)})')
                break
            except Exception as e:
                log.warning(f'  YOLO fail: {e}')

        self.ctrl.execute(SetServo(servo=CAMERA_SERVO_ANGLE))
        log.info(f'{self.__class__.__name__} init succ.')
        return False

    def _compute_wheels(self, state_output, lane_result, h_frame, w_frame):
        """
        完全对齐 notebook Step4-5 的 图像→速度 逻辑
        state_output: 状态机输出（仅用 state / approach_target_x/y）
        lane_result:  YellowLaneFollower.infer() 输出（candidates）
        """
        state = state_output["state"]
        rr, fr, fl, rl = 0, 0, 0, 0
        action = "Stop"

        if state == "STOP":
            return rr, fr, fl, rl, action

        if state == "PAUSE":
            action = "Pause"
            return rr, fr, fl, rl, action

        if state == "TURN":
            action = "Turning..."
            return rr, fr, fl, rl, action

        # ====== 以下完全对齐 notebook Step4-5 ======
        image_center = w_frame / 2.0
        lane_width_px = w_frame * 0.20
        h, w = h_frame, w_frame

        # ---------- Step 2 变量：来自状态机 ----------
        approach_mode = state_output.get("is_approaching", False)
        approach_target_x = state_output.get("approach_target_x")
        approach_target_y = state_output.get("approach_target_y")
        approach_sign = state_output.get("pending_sign")
        trigger_turn = False  # 由边界检测设置

        # ---------- Step 3 变量：来自 YellowLaneFollower ----------
        candidates = lane_result.get("candidates", []) if lane_result and lane_result.get("ok") else []

        # 对齐 notebook: 按行分类 row_info = [(y, xs, tx, md), ...]
        turn_bias = approach_sign if approach_sign in ("left", "right") else None
        row_info = build_lane_row_info(
            candidates,
            image_width=w_frame,
            expected_width=lane_width_px,
            reference_center=self._lane_reference_center,
            turn_bias=turn_bias,
        )

        # ---------- Step 4: 边界检测 ----------
        # 由 loop 在外部计算 yellow_ratio，这里接收 boundary_hit
        boundary_hit = state_output.get("_boundary_hit", False)

        # ---------- Step 4: 行为决策 (对齐 notebook cell4) ----------
        target_x = image_center
        lookahead_y = int(h * 0.72)
        error = 0.0
        correction = 0
        curve_slow = False

        if boundary_hit:
            if approach_mode:
                # notebook: BOUNDARY→TURN
                correction = 0
                trigger_turn = True
                curve_slow = False
                action = 'BOUNDARY→TURN'
            else:
                # notebook: BOUNDARY→STOP
                correction = 0
                curve_slow = False
                action = 'BOUNDARY→STOP'

        elif approach_mode and approach_target_x is not None:
            # notebook: APPROACH→sign (对齐 cell4 angle→correction)
            target_x = approach_target_x
            lookahead_y = int(approach_target_y) if approach_target_y else int(h * 0.72)
            dx = approach_target_x - image_center
            dy = h - (approach_target_y or h / 2)
            angle_deg = math.degrees(math.atan2(dx, dy))
            abs_a = abs(angle_deg)
            if abs_a < 3:
                correction = 0
            elif abs_a < 8:
                correction = 1
            elif abs_a < 15:
                correction = 2
            elif abs_a < 25:
                correction = 3
            else:
                correction = 4
            if angle_deg < 0:
                correction = -correction
            error = target_x - image_center
            curve_slow = False
            action = f'APPROACH→{approach_sign}'

        elif candidates:
            # notebook: 巡迹模式 (对齐 cell4 else 分支)
            # 找最近有检测的线
            nv = None
            for i in range(len(row_info) - 1, -1, -1):
                if row_info[i][1]:
                    nv = i
                    break
            # 找最近双点线 (lane_center)
            nd = None
            for i in range(len(row_info) - 1, -1, -1):
                if row_info[i][3] == 'lane_center':
                    nd = i
                    break

            # has_cluster (对齐 notebook: md == 'cluster')
            has_cluster = any(md == 'cluster' for _, _, _, md in row_info)
            curve_slow = has_cluster

            if nv is not None and nd is not None:
                if nv == nd:
                    correction = 0
                    mode = 'on_track'
                else:
                    d = nv - nd
                    nearest_x = row_info[nv][1][0]
                    # 对齐 replay/notebook: 点在左→正修正, 点在右→负修正
                    correction = d if nearest_x < image_center else -d
                    mode = f'dist={d}'
            elif nv is not None:
                d = nv
                nearest_x = row_info[nv][1][0]
                correction = d if nearest_x < image_center else -d
                mode = f'dist={d}'
            else:
                correction = 0
                mode = 'lost'

            row_control = compute_lane_row_correction(row_info, image_center)
            correction = row_control["correction"]
            mode = row_control["mode"]
            target_row = row_control["target_row"]
            target_x = row_info[target_row][2] if target_row is not None else image_center
            lookahead_y = row_info[target_row][0] if target_row is not None else int(h * 0.72)
            error = target_x - image_center
            if nd is not None:
                self._lane_reference_center = target_x

        else:
            # 无候选 → 低速前进 (保留原 recovery 行为)
            base_spd = 12
            rr, fr, fl, rl = -base_spd, -base_spd, base_spd, base_spd
            action = "Recover"
            return rr, fr, fl, rl, action

        # ---------- Step 5: 速度+差速 (对齐 notebook cell5) ----------
        if approach_mode:
            base_speed = 18
            min_speed = 16
        else:
            base_speed = 23 if (trigger_turn or curve_slow) else 25
            min_speed = 22

        if trigger_turn:
            left_spd = 0
            right_spd = 0
        else:
            left_spd, right_spd = compute_differential_speeds(
                base_speed=base_speed,
                min_speed=min_speed,
                correction=correction,
                correction_gain=self._lane_correction_gain,
                right_wheel_compensation=self._right_wheel_compensation,
                max_wheel_speed_delta=self._max_wheel_speed_delta,
            )

        # ESP32 映射: 右轮负=正转, 左轮正=正转
        rr, fr, fl, rl = -right_spd, -right_spd, left_spd, left_spd

        if 'mode' in locals():
            action = f'{mode} Diff({correction:+d})'
        elif not action or action == "Stop":
            action = f'Diff({correction:+d})'

        return rr, fr, fl, rl, action

    def _start_post_turn_alignment(self, sign):
        self._post_turn_align = sign in ("left", "right")
        self._alignment_turn = sign if self._post_turn_align else None
        if sign in ("left", "right"):
            self._last_turn_direction = sign
        self._alignment_steps = 0
        self._alignment_streak = 0
        self._lane_reference_center = None
        self.sm.pending_sign = None
        self.sm.approach_target_x = None
        self.sm.approach_target_y = None

    def _finish_post_turn_alignment(self):
        self._post_turn_align = False
        self._alignment_turn = None
        self._alignment_steps = 0
        self._alignment_streak = 0

    def _save_correction_debug(self, frame, debug_dir, prefix, reason,
                               metrics=None, boundary_result=None,
                               action_desc=None, mask_pairs=None):
        if frame is None or not debug_dir:
            return
        out_dir = os.path.join(debug_dir, "correction")
        os.makedirs(out_dir, exist_ok=True)
        img = frame.copy()
        lines = [
            f"{prefix}: {reason}",
        ]
        if action_desc:
            lines.append(action_desc)
        if metrics:
            lines.append(
                "rows={valid_rows} center_pairs={center_pair_rows} mask_pairs={mask_center_pair_rows} "
                "center_std={center_std:.1f} "
                "heading={heading_delta:.1f} offset={near_offset:.1f} "
                "width_cv={width_cv:.2f}".format(**metrics)
            )
        if mask_pairs:
            for pair in mask_pairs:
                y = int(pair["y"])
                left = int(pair["left"])
                right = int(pair["right"])
                center = int(pair["center"])
                cv2.circle(img, (left, y), 5, (255, 255, 0), -1)
                cv2.circle(img, (right, y), 5, (255, 255, 0), -1)
                cv2.circle(img, (center, y), 5, (0, 255, 0), -1)
                cv2.line(img, (left, y), (right, y), (0, 255, 0), 2)
        if boundary_result:
            lines.append(
                "boundary hit={} cov={:.2f} y={:.2f}".format(
                    boundary_result.get("hit"),
                    boundary_result.get("coverage", 0.0),
                    boundary_result.get("y_ratio", 0.0),
                )
            )
            y = int(boundary_result.get("y", 0))
            cv2.line(img, (0, y), (img.shape[1], y), (0, 165, 255), 2)

        y0 = 28
        for i, line in enumerate(lines):
            cv2.putText(img, line, (10, y0 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
        filename = "{}_{}_{}.jpg".format(
            int(time.time() * 1000),
            prefix,
            reason.replace(" ", "_").replace("/", "_"),
        )
        cv2.imwrite(os.path.join(out_dir, filename), img)

    def _execute_alignment_step(self):
        if self._alignment_turn == "right":
            spin_action = SpinAntiClockwise
        else:
            spin_action = SpinClockwise
        self.ctrl.execute(spin_action(speed=42))
        time.sleep(0.05)
        self.ctrl.execute(spin_action(speed=34))
        time.sleep(self._alignment_step_seconds)
        self.ctrl.execute(Stop())
        time.sleep(self._alignment_settle_seconds)
        self._alignment_steps += 1

    def _handle_post_turn_alignment(self, lane_result, image_width,
                                    frame=None, debug_dir=None):
        metrics = evaluate_lane_alignment(
            lane_result.get("candidates", []) if lane_result else [],
            image_width=image_width,
            turn_bias=self._alignment_turn,
        )
        mask_alignment = {"center_pair_rows": 0, "pairs": []}
        if frame is not None and self.follower is not None:
            mask = self.follower.segment_yellow(frame)
            mask_alignment = analyze_center_lane_from_mask(
                mask,
                scan_ratios=getattr(self.follower, "scan_ratios", None),
            )
            metrics["mask_center_pair_rows"] = mask_alignment["center_pair_rows"]
        if metrics["aligned"]:
            self._alignment_streak += 1
            if self._alignment_streak >= 2:
                log.info(f'POST_TURN_ALIGN success: {metrics}')
                self._save_correction_debug(
                    frame, debug_dir, "POST_TURN_ALIGN", "success",
                    metrics=metrics,
                    mask_pairs=mask_alignment["pairs"],
                )
                self._finish_post_turn_alignment()
            return True

        self._alignment_streak = 0
        if metrics["center_pair_rows"] >= 3:
            log.info(f'POST_TURN_ALIGN center pairs ready: {metrics}')
            self._save_correction_debug(
                frame, debug_dir, "POST_TURN_ALIGN",
                "three_center_pairs_use_diff_control",
                metrics=metrics,
                mask_pairs=mask_alignment["pairs"],
                action_desc="no more spin; continue with cruise angle/diff correction",
            )
            self._finish_post_turn_alignment()
            return True

        if metrics["mask_center_pair_rows"] >= 3:
            log.info(f'POST_TURN_ALIGN mask center pairs ready: {metrics}')
            self._save_correction_debug(
                frame, debug_dir, "POST_TURN_ALIGN",
                "three_mask_pairs_use_diff_control",
                metrics=metrics,
                mask_pairs=mask_alignment["pairs"],
                action_desc="no more spin; mask shows centered left/right lanes",
            )
            self._finish_post_turn_alignment()
            return True

        usable_after_turn = (
            metrics["valid_rows"] >= 3
            and metrics["center_std"] <= 85.0
            and metrics["heading_delta"] <= 105.0
            and metrics["near_offset"] <= 230.0
            and metrics["width_cv"] <= 0.50
        )
        if usable_after_turn:
            log.info(f'POST_TURN_ALIGN usable: {metrics}')
            self._save_correction_debug(
                frame, debug_dir, "POST_TURN_ALIGN", "usable_no_more_turn",
                metrics=metrics,
                mask_pairs=mask_alignment["pairs"],
            )
            self._finish_post_turn_alignment()
            return True

        if self._alignment_steps > 0 and metrics["valid_rows"] < 2:
            log.warning(f'POST_TURN_ALIGN lost lane, stop correcting: {metrics}')
            self.ctrl.execute(Stop())
            self._save_correction_debug(
                frame, debug_dir, "POST_TURN_ALIGN", "lost_lane_stop",
                metrics=metrics,
                mask_pairs=mask_alignment["pairs"],
            )
            self._finish_post_turn_alignment()
            return True

        if self._alignment_steps >= self._alignment_max_steps:
            log.warning(f'POST_TURN_ALIGN limit reached: {metrics}')
            self.ctrl.execute(Stop())
            self._save_correction_debug(
                frame, debug_dir, "POST_TURN_ALIGN", "step_limit_stop",
                metrics=metrics,
                mask_pairs=mask_alignment["pairs"],
            )
            self._finish_post_turn_alignment()
            return True

        log.info(
            f'POST_TURN_ALIGN step={self._alignment_steps + 1} '
            f'turn={self._alignment_turn} metrics={metrics}'
        )
        self._save_correction_debug(
            frame, debug_dir, "POST_TURN_ALIGN",
            f"step_{self._alignment_steps + 1}",
            metrics=metrics,
            mask_pairs=mask_alignment["pairs"],
            action_desc=(
                "turn={} speed=42 for 0.05s, speed=34 for {:.2f}s, settle={:.2f}s"
                .format(self._alignment_turn,
                        self._alignment_step_seconds,
                        self._alignment_settle_seconds)
            ),
        )
        self._execute_alignment_step()
        return True

    def _stop_for_boundary(self):
        for _ in range(5):
            self.ctrl.send_raw_wheels(0, 0, 0, 0)

    def _bottom_yellow_ratio(self, frame, roi_ratio=0.30):
        h, _ = frame.shape[:2]
        bottom_roi = frame[int(h * (1.0 - roi_ratio)):, :]
        hsv_roi = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv_roi,
                                  np.array([12, 70, 80], dtype=np.uint8),
                                  np.array([45, 255, 255], dtype=np.uint8))
        yellow_count = cv2.countNonZero(yellow_mask)
        roi_area = yellow_mask.shape[0] * yellow_mask.shape[1]
        return yellow_count / roi_area if roi_area > 0 else 0.0

    def _save_boundary_warning_debug(self, frame, debug_dir, frame_count,
                                     boundary_result, yellow_ratio,
                                     stopped=False):
        if frame is None or not debug_dir:
            return
        img = frame.copy()
        label = (
            "BOUNDARY_WARNING{} cov={:.2f} y={:.2f} bottom30={:.2f}"
            .format(
                " STOP" if stopped else "",
                boundary_result.get("coverage", 0.0),
                boundary_result.get("y_ratio", 0.0),
                yellow_ratio,
            )
        )
        cv2.putText(img, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        y = int(boundary_result.get("y", 0))
        cv2.line(img, (0, y), (img.shape[1], y), (0, 0, 255), 2)
        cv2.imwrite(
            os.path.join(debug_dir, f"f{frame_count:05d}_BOUNDARY_WARNING.jpg"),
            img,
        )

    def _save_yolo_debug(self, frame, debug_dir, frame_count, sign_result):
        if frame is None or not debug_dir or not sign_result:
            return
        if not sign_result.get("fresh"):
            return

        target_labels = ("left", "right", "turnaround", "stop")
        bboxes = [
            bbox for bbox in sign_result.get("all_bboxes", [])
            if len(bbox) >= 6 and bbox[4] in target_labels
        ]
        accepted = sign_result.get("signs", []) + sign_result.get("stop_signs", [])
        if not bboxes and not accepted:
            return

        out_dir = os.path.join(debug_dir, "yolo")
        os.makedirs(out_dir, exist_ok=True)
        img = frame.copy()

        accepted_boxes = {
            tuple(int(v) for v in sign.get("bbox", []))
            for sign in accepted
            if sign.get("bbox")
        }
        lane_filter = sign_result.get("lane_filter", {})
        decisions = lane_filter.get("decisions", [])
        decisions_by_box = {
            tuple(int(v) for v in decision.get("bbox", [])): decision
            for decision in decisions
            if decision.get("bbox")
        }

        for x1, y1, x2, y2, cate, score in bboxes:
            box = (int(x1), int(y1), int(x2), int(y2))
            used = box in accepted_boxes
            decision = decisions_by_box.get(box)
            crossed = bool(decision and decision.get("crossed"))
            color = (0, 255, 0) if used else (
                (0, 0, 255) if crossed else (140, 140, 140)
            )
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = "{} {:.2f} {}".format(
                cate,
                float(score),
                "USED" if used else ("CROSS" if crossed else "SKIP"),
            )
            text_y = max(20, box[1] - 8)
            cv2.putText(img, label, (box[0], text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        origin = lane_filter.get(
            "origin",
            (img.shape[1] // 2, img.shape[0] - 1),
        )
        for decision in decisions:
            target = decision.get("target")
            if not target:
                continue
            crossed = decision.get("crossed", False)
            color = (0, 0, 255) if crossed else (0, 255, 0)
            cv2.line(img, tuple(origin), tuple(target), color, 2)
            crossing = decision.get("crossing")
            if crossing:
                cv2.circle(img, tuple(crossing), 7, (0, 165, 255), -1)

        summary = "YOLO signs={} stop={}".format(
            len(sign_result.get("signs", [])),
            len(sign_result.get("stop_signs", [])),
        )
        if lane_filter.get("enabled"):
            summary += " lane signs {}->{} stop {}->{}".format(
                lane_filter["signs_before"], lane_filter["signs_after"],
                lane_filter["stop_before"], lane_filter["stop_after"],
            )
        cv2.putText(img, summary, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        cv2.imwrite(
            os.path.join(debug_dir, "yolo", f"f{frame_count:05d}_YOLO.jpg"),
            img,
        )

    def _start_boundary_escape(self, boundary_result):
        if not self._boundary_escape:
            log.warning(
                "BOUNDARY_ESCAPE start: coverage=%.1f%% y=%.1f%%",
                boundary_result["coverage"] * 100,
                boundary_result["y_ratio"] * 100,
            )
        self._boundary_escape = True
        self._boundary_escape_direction = (
            self._last_turn_direction
            if self._last_turn_direction in ("left", "right")
            else "right"
        )
        self._boundary_escape_steps = 0
        self._boundary_escape_streak = 0
        self.sm.pending_sign = None
        self.sm.approach_target_x = None
        self.sm.approach_target_y = None

    def _finish_boundary_escape(self, reason, metrics=None):
        log.info(f'BOUNDARY_ESCAPE finish: {reason} metrics={metrics}')
        self._boundary_escape = False
        self._boundary_escape_steps = 0
        self._boundary_escape_streak = 0

    def _execute_boundary_escape_step(self):
        if self._boundary_escape_direction == "right":
            spin_action = SpinAntiClockwise
        else:
            spin_action = SpinClockwise
        self.ctrl.execute(spin_action(speed=self._boundary_escape_speed))
        time.sleep(self._boundary_escape_step_seconds)
        self.ctrl.execute(Stop())
        time.sleep(self._boundary_escape_settle_seconds)
        self._boundary_escape_steps += 1

    def _handle_boundary_escape(self, lane_result, image_width, boundary_result,
                                frame=None, debug_dir=None):
        metrics = evaluate_lane_alignment(
            lane_result.get("candidates", []) if lane_result else [],
            image_width=image_width,
            turn_bias=self._boundary_escape_direction,
        )
        if not boundary_result["hit"] and metrics["aligned"]:
            self._boundary_escape_streak += 1
            if self._boundary_escape_streak >= 2:
                self._save_correction_debug(
                    frame, debug_dir, "BOUNDARY_ESCAPE", "lane_aligned_finish",
                    metrics=metrics,
                    boundary_result=boundary_result,
                )
                self._finish_boundary_escape("lane_aligned", metrics)
            return True

        self._boundary_escape_streak = 0
        if self._boundary_escape_steps >= self._boundary_escape_max_steps:
            self._finish_boundary_escape("step_limit", metrics)
            self.ctrl.execute(Stop())
            self._save_correction_debug(
                frame, debug_dir, "BOUNDARY_ESCAPE", "step_limit_stop",
                metrics=metrics,
                boundary_result=boundary_result,
            )
            return True

        log.info(
            f'BOUNDARY_ESCAPE step={self._boundary_escape_steps + 1} '
            f'direction={self._boundary_escape_direction} '
            f'boundary_hit={boundary_result["hit"]} metrics={metrics}'
        )
        self._save_correction_debug(
            frame, debug_dir, "BOUNDARY_ESCAPE",
            f"step_{self._boundary_escape_steps + 1}",
            metrics=metrics,
            boundary_result=boundary_result,
            action_desc=(
                "direction={} speed={} for {:.2f}s, settle={:.2f}s"
                .format(self._boundary_escape_direction,
                        self._boundary_escape_speed,
                        self._boundary_escape_step_seconds,
                        self._boundary_escape_settle_seconds)
            ),
        )
        self._execute_boundary_escape_step()
        return True

    def loop(self):
        ret = self.init_state()
        if ret:
            log.error(f'{self.__class__.__name__} init failed.')
            return

        frame = np.ndarray((self.height, self.width, 3),
                           dtype=np.uint8, buffer=self.broadcaster.buf)
        log.info(f'{self.__class__.__name__} loop start')

        frame_count = 0
        stop_file = os.path.join(os.getcwd(), ".smart_stop")
        boundary_threshold = 0.30
        self._turn_executed = False
        last_cmd_time = time.time()
        CMD_TIMEOUT = 0.35
        debug_dir = os.path.join(os.getcwd(), "capture", "smart_debug")
        os.makedirs(debug_dir, exist_ok=True)

        # 等待摄像头数据正常
        log.info("等待摄像头数据...")
        valid_frame = False
        for wait_i in range(100):
            test_frame = frame.copy()
            if test_frame is not None and np.mean(test_frame) > 10:
                valid_frame = True
                log.info(f"摄像头就绪 (第{wait_i}帧)")
                break
            time.sleep(0.05)
        if not valid_frame:
            log.error("摄像头数据异常，退出")
            return

        try:
            while True:
                if self.stop_sign.value:
                    break
                if self.pause_sign.value:
                    continue
                if os.path.exists(stop_file):
                    log.info("收到停止信号")
                    break

                loop_start = time.time()
                img_bgr = frame.copy()
                h_f, w_f = img_bgr.shape[:2]

                boundary_result = detect_horizontal_boundary(img_bgr)
                state_snapshot = self.sm._build_output()
                warning_allowed = state_snapshot.get("state") not in (
                    "TURN", "PAUSE", "STOP")
                if not warning_allowed:
                    self._boundary_warning_mode = False
                if boundary_result["warning"] and warning_allowed:
                    self._boundary_warning_mode = True

                if self._boundary_warning_mode and warning_allowed:
                    yellow_ratio = self._bottom_yellow_ratio(
                        img_bgr, roi_ratio=0.30)
                    boundary_stop = (
                        yellow_ratio >= boundary_threshold
                        or boundary_result["hit"]
                    )
                    state_output = state_snapshot
                    if boundary_stop:
                        self._stop_for_boundary()
                        self._save_boundary_warning_debug(
                            img_bgr, debug_dir, frame_count,
                            boundary_result, yellow_ratio,
                            stopped=True,
                        )
                        pending_sign = state_output.get("pending_sign")
                        self._boundary_warning_mode = False
                        if pending_sign in ("left", "right", "turnaround"):
                            self.sm.on_boundary_detected()
                            log.info(
                                "Boundary warning stop: bottom30=%.1f%% sign=%s",
                                yellow_ratio * 100,
                                pending_sign,
                            )
                        else:
                            self._start_boundary_escape({
                                "coverage": max(
                                    yellow_ratio,
                                    boundary_result["coverage"],
                                ),
                                "y_ratio": boundary_result["y_ratio"],
                                "hit": True,
                            })
                            log.info(
                                "Boundary warning escape: bottom30=%.1f%%",
                                yellow_ratio * 100,
                            )
                    else:
                        slow_speed = 18
                        self.ctrl.send_raw_wheels(
                            -slow_speed, -slow_speed,
                            slow_speed, slow_speed,
                        )
                        self._save_boundary_warning_debug(
                            img_bgr, debug_dir, frame_count,
                            boundary_result, yellow_ratio,
                            stopped=False,
                        )
                        log.info(
                            "#%d BOUNDARY_WARNING bottom30=%.1f%% cov=%.1f%% y=%.1f%%",
                            frame_count,
                            yellow_ratio * 100,
                            boundary_result["coverage"] * 100,
                            boundary_result["y_ratio"] * 100,
                        )
                    frame_count += 1
                    last_cmd_time = time.time()
                    continue

                # 1. 巡迹感知 (YellowLaneFollower.infer)
                lane_result = self.follower.infer(img_bgr)

                # 2. YOLO检测
                sign_result = None
                if self.det and not self._post_turn_align and not self._boundary_escape:
                    sign_result = self.det.detect(img_bgr)
                    sign_result = filter_signs_by_lane_mask(
                        sign_result,
                        lane_result.get("mask") if lane_result else None,
                    )
                    self._save_yolo_debug(
                        img_bgr,
                        debug_dir,
                        frame_count,
                        sign_result,
                    )
                    if sign_result and sign_result.get("signs"):
                        for s in sign_result["signs"]:
                            log.info(f'YOLO: {s["type"]} score={s["score"]:.3f} pos=({s["x"]:.0f},{s["y"]:.0f})')
                    if sign_result and sign_result.get("lane_filter", {}).get("enabled"):
                        lane_filter = sign_result["lane_filter"]
                        crossed = sum(
                            1 for decision in lane_filter["decisions"]
                            if decision.get("crossed")
                        )
                        log.info(
                            "YOLO lane filter: crossed=%d signs=%d->%d stop=%d->%d",
                            crossed,
                            lane_filter["signs_before"],
                            lane_filter["signs_after"],
                            lane_filter["stop_before"],
                            lane_filter["stop_after"],
                        )

                # 3. 状态机更新（管理 PAUSE/TURN/STOP 转移）
                state_output = self.sm.update(lane_result, sign_result)

                # 4. 边界检测 (对齐 notebook cell4 顶部)
                boundary_hit = False
                if state_output.get("is_approaching"):
                    if boundary_result["hit"]:
                        boundary_hit = True
                        self._stop_for_boundary()
                        self.sm.on_boundary_detected()
                        state_output = self.sm._build_output()
                        log.info(
                            "Early boundary hit: coverage=%.1f%% y=%.1f%%",
                            boundary_result["coverage"] * 100,
                            boundary_result["y_ratio"] * 100,
                        )
                    elif boundary_result["warning"]:
                        log.info(
                            "Boundary warning: coverage=%.1f%% y=%.1f%%",
                            boundary_result["coverage"] * 100,
                            boundary_result["y_ratio"] * 100,
                        )
                elif self._boundary_escape:
                    self._handle_boundary_escape(
                        lane_result, w_f, boundary_result,
                        frame=img_bgr,
                        debug_dir=debug_dir,
                    )
                    last_cmd_time = time.time()
                    continue
                elif state_output.get("is_cruising"):
                    bottom_roi = img_bgr[int(h_f * 0.70):, :]
                    hsv_roi = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2HSV)
                    yellow_mask = cv2.inRange(hsv_roi,
                                              np.array([12, 70, 80], dtype=np.uint8),
                                              np.array([45, 255, 255], dtype=np.uint8))
                    yellow_count = cv2.countNonZero(yellow_mask)
                    roi_area = yellow_mask.shape[0] * yellow_mask.shape[1]
                    yellow_ratio = yellow_count / roi_area if roi_area > 0 else 0
                    if yellow_ratio >= boundary_threshold or boundary_result["hit"]:
                        boundary_hit = True
                        self._stop_for_boundary()
                        pending_sign = state_output.get("pending_sign")
                        if pending_sign in ("left", "right", "turnaround"):
                            self.sm.on_boundary_detected()
                            state_output = self.sm._build_output()
                            log.info(f'Boundary hit: {yellow_ratio:.1%}')
                        else:
                            self._start_boundary_escape({
                                "coverage": max(
                                    yellow_ratio,
                                    boundary_result["coverage"],
                                ),
                                "y_ratio": boundary_result["y_ratio"],
                                "hit": True,
                            })
                            last_cmd_time = time.time()
                            continue

                state_output["_boundary_hit"] = boundary_hit

                # 5. 状态处理
                state_name = state_output["state"]

                # PAUSE: 高频发停止指令
                if state_name == "PAUSE":
                    for _ in range(5):
                        self.ctrl.send_raw_wheels(0, 0, 0, 0)
                    time.sleep(0.3)
                    last_cmd_time = time.time()
                    continue

                # TURN: 执行旋转
                if state_name == "TURN":
                    if not self._turn_executed:
                        sign = state_output.get("pending_sign")
                        self._alignment_turn = sign
                        if sign == "left":
                            self.ctrl.execute(TurnLeftInPlace())
                            log.info("执行: TurnLeftInPlace")
                        elif sign == "right":
                            self.ctrl.execute(TurnRightInPlace())
                            log.info("执行: TurnRightInPlace")
                        elif sign == "turnaround":
                            self.ctrl.execute(TurnAround())
                            log.info("执行: TurnAround")
                        self._turn_executed = True
                    continue

                if self._turn_executed:
                    self._start_post_turn_alignment(self._alignment_turn)
                self._turn_executed = False

                if self._post_turn_align:
                    self._handle_post_turn_alignment(
                        lane_result, w_f,
                        frame=img_bgr,
                        debug_dir=debug_dir,
                    )
                    last_cmd_time = time.time()
                    continue

                # 6. 计算电机指令 (完全对齐 notebook Step4-5)
                rr, fr, fl, rl, action = self._compute_wheels(
                    state_output, lane_result, h_f, w_f)
                left_display = fl
                right_display = -rr

                # 7. 发送
                now = time.time()
                elapsed_since_cmd = now - last_cmd_time

                if state_name == "CRUISE" and elapsed_since_cmd >= CMD_TIMEOUT:
                    log.info(
                        f'CRUISE_LATE {elapsed_since_cmd:.2f}s, send fresh command'
                    )

                self.ctrl.send_raw_wheels(rr, fr, fl, rl)
                last_cmd_time = now

                # 8. 保存调试帧
                if state_name != "TURN":
                    debug_img = img_bgr.copy()
                    cv2.putText(debug_img,
                                f"#{frame_count} {state_name} {action} L={left_display} R={right_display} d={left_display-right_display:+d}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(debug_dir, f"f{frame_count:05d}_{state_name}.jpg"), debug_img)

                # 9. 日志
                frame_count += 1
                loop_ms = (time.time() - loop_start) * 1000
                log.info(
                    f'#{frame_count} {state_name} {action} L={left_display} R={right_display} '
                    f'diff={left_display - right_display:+d} loop={loop_ms:.0f}ms'
                )

        except KeyboardInterrupt:
            pass
        finally:
            self.ctrl.execute(Stop())
            log.info(f'{self.__class__.__name__} stopped.')
