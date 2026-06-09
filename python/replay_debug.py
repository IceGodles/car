#!/usr/bin/env python3
"""Offline replay for capture/smart_debug images.

This script mirrors the SmartCruise perception/state logic closely enough to
test saved debug frames without driving the car.
"""

import glob
import math
import os
import sys
import time
import types

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.control.boundary_detector import detect_horizontal_boundary
from src.control.lane_alignment import (
    build_lane_row_info,
    compute_lane_row_correction,
    evaluate_lane_alignment,
)
from src.control.sign_detector import SignDetector
from src.control.state_machine import DrivingState, DrivingStateMachine

UTILS_DIR = os.path.join(BASE_DIR, "src", "utils")
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)
from visual_lane import YellowLaneFollower


class _ReplayFileLock:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def stub_replay_only_modules():
    sys.modules.setdefault("serial", types.ModuleType("serial"))
    filelock_module = types.ModuleType("filelock")
    filelock_module.FileLock = _ReplayFileLock
    sys.modules.setdefault("filelock", filelock_module)


def make_yolo_detector():
    yolo_path = os.path.join(BASE_DIR, "weights", "best.om")
    if not os.path.exists(yolo_path):
        yolo_path = os.path.join(BASE_DIR, "weights", "yolo.om")
    if not os.path.exists(yolo_path):
        print("YOLO not found, replay will use lane/boundary only.")
        return None
    try:
        stub_replay_only_modules()
        from src.models import YoloV5

        model = YoloV5(yolo_path)
        print(f"YOLO loaded: {os.path.basename(yolo_path)}")
        return SignDetector(model, {
            "detect_interval": int(os.environ.get("REPLAY_DETECT_INTERVAL", "3")),
            "roi_keep_ratio": 0.6,
            "conf_threshold": 0.85,
        })
    except Exception as exc:
        print(f"YOLO load failed: {exc}")
        return None


def compute_wheels(state_output, lane_result, h_frame, w_frame, lane_ref):
    state = state_output["state"]
    if state in ("STOP", "PAUSE", "TURN"):
        return 0, 0, 0, 0, state.title(), lane_ref, {
            "target_x": w_frame / 2.0,
            "lookahead_y": int(h_frame * 0.72),
            "correction": 0,
            "curve_slow": False,
        }

    image_center = w_frame / 2.0
    lane_width_px = w_frame * 0.20
    approach_mode = state_output.get("is_approaching", False)
    approach_target_x = state_output.get("approach_target_x")
    approach_target_y = state_output.get("approach_target_y")
    approach_sign = state_output.get("pending_sign")
    boundary_hit = state_output.get("_boundary_hit", False)
    candidates = lane_result.get("candidates", []) if lane_result and lane_result.get("ok") else []

    turn_bias = approach_sign if approach_sign in ("left", "right") else None
    row_info = build_lane_row_info(
        candidates,
        image_width=w_frame,
        expected_width=lane_width_px,
        reference_center=lane_ref,
        turn_bias=turn_bias,
    )

    target_x = image_center
    lookahead_y = int(h_frame * 0.72)
    correction = 0
    curve_slow = False
    action = "Diff(+0)"

    if boundary_hit:
        action = "BOUNDARY_TO_TURN" if approach_mode else "BOUNDARY_STOP"
    elif approach_mode and approach_target_x is not None:
        target_x = approach_target_x
        lookahead_y = int(approach_target_y) if approach_target_y else int(h_frame * 0.72)
        dx = approach_target_x - image_center
        dy = h_frame - (approach_target_y or h_frame / 2.0)
        angle_deg = math.degrees(math.atan2(dx, dy))
        abs_angle = abs(angle_deg)
        if abs_angle < 3:
            correction = 0
        elif abs_angle < 8:
            correction = 1
        elif abs_angle < 15:
            correction = 2
        elif abs_angle < 25:
            correction = 3
        else:
            correction = 4
        if angle_deg < 0:
            correction = -correction
        action = f"APPROACH->{approach_sign}"
    elif candidates:
        nearest_valid = None
        for i in range(len(row_info) - 1, -1, -1):
            if row_info[i][1]:
                nearest_valid = i
                break
        nearest_dual = None
        for i in range(len(row_info) - 1, -1, -1):
            if row_info[i][3] == "lane_center":
                nearest_dual = i
                break
        curve_slow = any(mode == "cluster" for _, _, _, mode in row_info)
        if nearest_valid is not None and nearest_dual is not None:
            if nearest_valid == nearest_dual:
                correction = 0
                action = "on_track"
            else:
                dist = nearest_valid - nearest_dual
                nearest_x = row_info[nearest_valid][1][0]
                correction = dist if nearest_x < image_center else -dist
                action = f"dist={dist}"
        elif nearest_valid is not None:
            dist = nearest_valid
            nearest_x = row_info[nearest_valid][1][0]
            correction = dist if nearest_x < image_center else -dist
            action = f"dist={dist}"
        else:
            action = "lost"
        row_control = compute_lane_row_correction(row_info, image_center)
        correction = row_control["correction"]
        action = row_control["mode"]
        target_row = row_control["target_row"]
        if target_row is not None:
            target_x = row_info[target_row][2]
            lookahead_y = row_info[target_row][0]
        if nearest_dual is not None:
            lane_ref = target_x
    else:
        speed = 12
        return -speed, -speed, speed, speed, "Recover", lane_ref, {
            "target_x": target_x,
            "lookahead_y": lookahead_y,
            "correction": 0,
            "curve_slow": False,
        }

    if approach_mode:
        base_speed = 18
        min_speed = 16
    else:
        base_speed = 23 if (boundary_hit or curve_slow) else 25
        min_speed = 22

    if boundary_hit:
        left_spd = 0
        right_spd = 0
    elif correction >= 0:
        left_spd = base_speed + correction
        right_spd = base_speed
    else:
        left_spd = base_speed
        right_spd = base_speed - correction

    left_spd = max(min_speed, min(40, left_spd))
    right_spd = max(min_speed, min(40, right_spd))
    if abs(left_spd - right_spd) > 8:
        if left_spd > right_spd:
            left_spd = min(right_spd + 8, 40)
        else:
            right_spd = min(left_spd + 8, 40)

    return -right_spd, -right_spd, left_spd, left_spd, action, lane_ref, {
        "target_x": target_x,
        "lookahead_y": lookahead_y,
        "correction": correction,
        "curve_slow": curve_slow,
    }


def main():
    follower = YellowLaneFollower({
        "adaptive_hsv": True,
        "scan_ratios": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.78],
        "scan_weights": [0.05, 0.10, 0.15, 0.25, 0.20, 0.15, 0.10],
        "single_line_gain": 0.25,
        "center_deadband_px": 20,
    })
    detector = make_yolo_detector()
    sm = DrivingStateMachine()
    lane_ref = None

    debug_dir = os.path.join(BASE_DIR, "capture", "smart_debug")
    output_dir = os.path.join(debug_dir, "replay_new")
    os.makedirs(output_dir, exist_ok=True)
    image_glob = os.environ.get("REPLAY_IMAGE_GLOB", "f*.jpg")
    images = sorted(p for p in glob.glob(os.path.join(debug_dir, image_glob))
                    if "replay" not in p)
    print(f"Found {len(images)} images")

    frame_dt = float(os.environ.get("REPLAY_FRAME_DT", "0.10"))
    boundary_warning_mode = False
    for idx, img_path in enumerate(images):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        boundary_result = detect_horizontal_boundary(frame)
        state_snapshot = sm._build_output()
        warning_allowed = state_snapshot.get("state") not in (
            "TURN", "PAUSE", "STOP")
        if not warning_allowed:
            boundary_warning_mode = False
        if boundary_result["warning"] and warning_allowed:
            boundary_warning_mode = True

        lane_result = follower.infer(frame)
        sign_result = None if boundary_warning_mode else (
            detector.detect(frame) if detector else None)

        if sign_result and sign_result.get("signs"):
            for sign in sign_result["signs"]:
                print(
                    f"YOLO frame={idx} {sign['type']} "
                    f"score={sign['score']:.3f} pos=({sign['x']:.0f},{sign['y']:.0f})"
                )

        state_output = sm.update(lane_result, sign_result)
        boundary_hit = False
        if boundary_warning_mode and warning_allowed:
            bottom_roi = frame[int(h * 0.70):, :]
            yellow_mask = cv2.inRange(
                cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2HSV),
                np.array([12, 70, 80], dtype=np.uint8),
                np.array([45, 255, 255], dtype=np.uint8),
            )
            yellow_ratio = cv2.countNonZero(yellow_mask) / float(yellow_mask.size)
            boundary_hit = yellow_ratio >= 0.30 or boundary_result["hit"]
            if boundary_hit:
                boundary_warning_mode = False
                if state_output.get("pending_sign") in (
                        "left", "right", "turnaround"):
                    sm.on_boundary_detected()
                    state_output = sm._build_output()
        elif state_output.get("is_approaching") and boundary_result["hit"]:
            boundary_hit = True
            sm.on_boundary_detected()
            state_output = sm._build_output()
        elif state_output.get("is_cruising"):
            bottom_roi = frame[int(h * 0.70):, :]
            yellow_mask = cv2.inRange(
                cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2HSV),
                np.array([12, 70, 80], dtype=np.uint8),
                np.array([45, 255, 255], dtype=np.uint8),
            )
            yellow_ratio = cv2.countNonZero(yellow_mask) / float(yellow_mask.size)
            if yellow_ratio >= 0.30 or boundary_result["hit"]:
                boundary_hit = True
                if state_output.get("pending_sign") in ("left", "right", "turnaround"):
                    sm.on_boundary_detected()
                    state_output = sm._build_output()
        state_output["_boundary_hit"] = boundary_hit

        rr, fr, fl, rl, action, lane_ref, extra = compute_wheels(
            state_output, lane_result, h, w, lane_ref)
        if boundary_warning_mode or (boundary_hit and boundary_result["warning"]):
            action = f"BOUNDARY_WARNING {action}"
        left_display = fl
        right_display = -rr

        turn_bias = state_output.get("pending_sign")
        alignment = evaluate_lane_alignment(
            lane_result.get("candidates", []) if lane_result else [],
            image_width=w,
            turn_bias=turn_bias if turn_bias in ("left", "right") else None,
        )

        print(
            f"#{idx:04d} {os.path.basename(img_path)} "
            f"state={state_output['state']} pending={state_output.get('pending_sign')} "
            f"{action} L={left_display} R={right_display} "
            f"boundary={boundary_hit} y={boundary_result['y_ratio']:.2f} "
            f"align_rows={alignment['valid_rows']}"
        )

        vis = frame.copy()
        target_x = extra["target_x"]
        lookahead_y = extra["lookahead_y"]
        cv2.circle(vis, (int(target_x), int(lookahead_y)), 10, (0, 0, 255), -1)
        cv2.line(vis, (w // 2, h), (int(target_x), int(lookahead_y)), (0, 0, 255), 2)
        cv2.line(vis, (w // 2, 0), (w // 2, h), (180, 180, 180), 1)
        cv2.line(vis, (0, boundary_result["y"]), (w, boundary_result["y"]), (0, 165, 255), 2)
        title = (
            f"#{idx} {state_output['state']} pending={state_output.get('pending_sign')} "
            f"{action} L={left_display} R={right_display} d={left_display-right_display:+d}"
        )
        info_y = max(24, h - 82)
        cv2.putText(vis, title, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(
            vis,
            f"Boundary hit={boundary_hit} cov={boundary_result['coverage']:.2f} "
            f"y={boundary_result['y_ratio']:.2f}",
            (10, info_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255) if boundary_hit else (0, 200, 255),
            2,
        )
        cv2.putText(
            vis,
            f"Align rows={alignment['valid_rows']} heading={alignment['heading_delta']:.0f} "
            f"offset={alignment['near_offset']:.0f}",
            (10, info_y + 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 0) if alignment["aligned"] else (0, 0, 255),
            2,
        )
        cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), vis)

        # Advance simulated time so APPROACH/PAUSE/TURN transitions can happen
        # even though replay runs much faster than real time.
        sm.state_enter_time -= frame_dt
        if sm.state == DrivingState.TURN:
            # There are no TURN debug frames, so mark the turn as consumed and
            # return to CRUISE for the next saved image.
            print(f"  TURN would execute: {state_output.get('pending_sign')}")
            sm.pending_sign = None
            sm.approach_target_x = None
            sm.approach_target_y = None
            sm._enter_state(DrivingState.CRUISE, reason="replay_turn_done")

    print(f"Done. Annotated images saved to {output_dir}")


if __name__ == "__main__":
    main()
