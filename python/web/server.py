#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能小车 Web 控制台
- Flask + WebSocket 实时通信
- MJPEG 视频流 + 录制（带算法可视化叠加）
- 通过 Controller 直接控制 ESP32 电机
- smart 模式下后台线程跑完整巡迹+检测+状态机
"""
import glob
import json
import logging
import os
import subprocess
import sys
import time
import threading
try:
    import pty
except ImportError:
    pty = None

import cv2
import numpy as np
import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(CURRENT_DIR, "..")
sys.path.insert(0, PROJECT_DIR)

from flask import Flask, render_template, Response, request, jsonify, send_file
from flask_socketio import SocketIO

app = Flask(__name__,
            template_folder=os.path.join(CURRENT_DIR, "templates"),
            static_folder=os.path.join(CURRENT_DIR, "static"))
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
log = logging.getLogger("web")

# ===================== Controller =====================
# Web端初始化Controller（手动模式用，smart模式用子进程不冲突）
ctrl = None
preview_mode = False

def init_controller():
    global ctrl, preview_mode
    try:
        from src.utils import Controller
        ctrl = Controller()
        log.info("Controller initialized.")
    except Exception as e:
        preview_mode = True
        log.warning(f"Controller init failed ({e}), preview mode.")

init_controller()

# ===================== 摄像头（单读取线程） =====================
_latest_frame = None
_frame_lock = threading.Lock()
_camera_started = False
_camera_cap = None
_camera_thread = None
_camera_stop_event = threading.Event()
_camera_lifecycle_lock = threading.Lock()

def _camera_reader_thread():
    """唯一摄像头读取线程，所有消费者从 _latest_frame 取帧"""
    global _latest_frame, _camera_started, _camera_cap
    cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
    with _camera_lifecycle_lock:
        _camera_cap = cap
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        log.error("Camera cannot open")
        with _camera_lifecycle_lock:
            _camera_cap = None
        return
    _camera_started = True
    log.info("Camera reader thread started")
    try:
        while not _camera_stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if _camera_stop_event.is_set():
                    break
                time.sleep(0.01)
                continue
            with _frame_lock:
                _latest_frame = frame
    finally:
        cap.release()
        with _camera_lifecycle_lock:
            if _camera_cap is cap:
                _camera_cap = None
        _camera_started = False
        log.info("Camera reader thread stopped")

def start_web_camera():
    global _camera_thread
    with _camera_lifecycle_lock:
        if _camera_thread is not None and _camera_thread.is_alive():
            return
        _camera_stop_event.clear()
        _camera_thread = threading.Thread(target=_camera_reader_thread, daemon=True)
        _camera_thread.start()

def stop_web_camera():
    global _latest_frame, _camera_thread
    _camera_stop_event.set()
    with _camera_lifecycle_lock:
        cap = _camera_cap
    if cap is not None:
        cap.release()
    thread = _camera_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=3)
    _camera_thread = None
    with _frame_lock:
        _latest_frame = None

def get_latest_frame():
    with _frame_lock:
        return _latest_frame.copy() if _latest_frame is not None else None

# 启动摄像头读取线程
start_web_camera()

# ===================== 录制（从共享帧缓冲取帧） =====================
_recording = False
_recording_writer = None
_recording_path = None
_recording_frame_count = 0
_recording_lock = threading.Lock()

def _recording_thread():
    """录制线程：从 _latest_frame 取帧写入文件"""
    global _recording_writer, _recording_frame_count
    # 等第一帧到来
    for _ in range(50):
        frame = get_latest_frame()
        if frame is not None:
            break
        time.sleep(0.05)
    else:
        log.error("Recording: no frame available, abort")
        return

    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(_recording_path, fourcc, 30, (w, h))
    if not writer.isOpened():
        log.error(f"VideoWriter cannot open: {_recording_path}")
        return

    with _recording_lock:
        _recording_writer = writer

    # 写第一帧
    writer.write(frame)
    _recording_frame_count = 1
    log.info(f"Recording started: {_recording_path} ({w}x{h})")

    while _recording:
        frame = get_latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        writer.write(frame)
        _recording_frame_count += 1

    writer.release()
    with _recording_lock:
        _recording_writer = None
    log.info(f"Recording ended: {_recording_frame_count} frames")

def start_recording():
    global _recording, _recording_path, _recording_frame_count
    if _recording:
        return
    video_dir = os.path.join(PROJECT_DIR, "video")
    os.makedirs(video_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    _recording_path = os.path.join(video_dir, f"rec_{ts}.avi")
    _recording_frame_count = 0
    _recording = True
    threading.Thread(target=_recording_thread, daemon=True).start()

def stop_recording():
    global _recording, _recording_frame_count, _recording_path
    _recording = False
    time.sleep(0.3)
    with _recording_lock:
        count = _recording_frame_count
        path = _recording_path
        _recording_path = None
        _recording_frame_count = 0
    return path, count

# ===================== 状态 =====================
car_state = {
    "mode": "idle", "state": "INIT", "fps": 0.0,
    "pid_error": 0.0, "pid_output": 0.0,
    "is_curve": False, "speed_factor": 1.0,
    "signs_detected": [], "speed": 0,
    "action": "Stop", "frame_count": 0,
    "infer_time_ms": 0.0,
    "wheel_speeds": {"rr": 0, "fr": 0, "fl": 0, "rl": 0},
    "preview_mode": preview_mode,
    "recording": False, "record_frames": 0,
    "distance_cm": -1, "obstacle_detected": False,
    "stop_threshold_cm": 20.0,
    "servo_angle": [110, 50],
    "debug_mode": False,
}

pid_params = {
    "kp": 0.0012, "ki": 0.0, "kd": 0.0005,
    "smooth_alpha": 0.3, "center_deadband_px": 20,
}
hsv_params = {
    "h_min": 12, "h_max": 45, "s_min": 70, "v_min": 80, "adaptive": True,
}
wheel_speeds = {"rr": 0, "fr": 0, "fl": 0, "rl": 0}
BEHAVIOR_SERVICE_URL = os.environ.get(
    "BEHAVIOR_SERVICE_URL",
    "http://192.168.8.200:5055",
).rstrip("/")

# ===================== 巡迹组件（启动时预加载） =====================
_follower = None
_state_machine = None
_sign_detector = None
_ctrl_lock = threading.Lock()

def init_models():
    """启动时预加载所有模型，避免运动时消耗算力"""
    global _follower, _state_machine, _sign_detector
    print("=== 初始化模型 ===", flush=True)

    from src.utils.visual_lane import YellowLaneFollower
    _follower = YellowLaneFollower({
        "adaptive_hsv": True,
        "scan_ratios": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.78],
        "scan_weights": [0.05, 0.10, 0.15, 0.25, 0.20, 0.15, 0.10],
        "single_line_gain": 0.25,
        "center_deadband_px": pid_params["center_deadband_px"],
        "kp": pid_params["kp"], "ki": pid_params["ki"],
        "kd": pid_params["kd"], "smooth_alpha": pid_params["smooth_alpha"],
    })
    print("  ✓ 巡迹模块", flush=True)

    from src.control.state_machine import DrivingStateMachine
    _state_machine = DrivingStateMachine()
    print("  ✓ 状态机", flush=True)

    # YOLO 在主线程加载（NPU 上下文绑定创建线程）
    yolo_candidates = [
        os.path.join(PROJECT_DIR, "weights", "best.om"),
        os.path.join(PROJECT_DIR, "weights", "yolo.om"),
    ]
    for yolo_path in yolo_candidates:
        if not os.path.exists(yolo_path):
            continue
        try:
            from src.models import YoloV5
            from src.control.sign_detector import SignDetector
            model = YoloV5(yolo_path)
            _sign_detector = SignDetector(model, {
                "detect_interval": 3, "roi_keep_ratio": 0.6, "conf_threshold": 0.85,
            })
            print(f"  ✓ YOLO模型 ({os.path.basename(yolo_path)})", flush=True)
            break
        except Exception as e:
            print(f"  ✗ {os.path.basename(yolo_path)}: {e}", flush=True)
            _sign_detector = None
    else:
        print("  ✗ 无可用YOLO模型", flush=True)

    print("=== 初始化完成 ===")

def sync_pid_to_follower():
    if _follower is not None:
        _follower.pid.kp = pid_params["kp"]
        _follower.pid.ki = pid_params["ki"]
        _follower.pid.kd = pid_params["kd"]
        _follower.pid.smooth_alpha = pid_params["smooth_alpha"]
        _follower.center_deadband_px = pid_params["center_deadband_px"]

# ===================== 电机控制 =====================
def send_wheels_to_esp32(rr, fr, fl, rl):
    if ctrl is None:
        return "PREVIEW", {}
    with _ctrl_lock:
        try:
            ret, obstacle_info = ctrl.send_raw_wheels(rr, fr, fl, rl)
            # 提取超声波距离更新到状态
            if obstacle_info:
                dist = obstacle_info.get("distance_cm")
                if dist is not None:
                    car_state["distance_cm"] = dist
                car_state["obstacle_detected"] = obstacle_info.get("obstacle_detected", False)
                threshold = obstacle_info.get("stop_threshold_cm")
                if threshold is not None:
                    car_state["stop_threshold_cm"] = threshold
            return ret, obstacle_info
        except Exception as e:
            log.error(f"send_raw_wheels failed: {e}")
            return "ERROR", {}

def direction_to_wheels(direction, speed):
    """方向 → 四轮速度 (rr, fr, fl, rl)
    电机通道: CH0=RearRight, CH1=FrontRight, CH2=FrontLeft, CH3=RearLeft
    前进: [-s, -s, +s, +s] (右轮负=正转, 左轮正=正转)
    """
    speed = int(speed)
    if direction == "forward":
        return (-speed, -speed, speed, speed)
    elif direction == "backward":
        return (speed, speed, -speed, -speed)
    elif direction == "left":
        # 左转: 原地逆时针旋转 SpinAntiClockwise(speed=60)
        spin_speed = 60
        return (spin_speed, spin_speed, spin_speed, spin_speed)
    elif direction == "right":
        # 右转: 原地顺时针旋转 SpinClockwise(speed=60)
        spin_speed = 60
        return (-spin_speed, -spin_speed, -spin_speed, -spin_speed)
    return (0, 0, 0, 0)

def apply_wheels(rr, fr, fl, rl):
    wheel_speeds.update({"rr": rr, "fr": fr, "fl": fl, "rl": rl})
    car_state["wheel_speeds"] = dict(wheel_speeds)
    # 手动模式发串口，smart模式不发（子进程控制）
    if ctrl is not None and car_state.get("mode") != "smart":
        try:
            ctrl.send_raw_wheels(rr, fr, fl, rl)
        except Exception as e:
            log.error(f"send failed: {e}")

def stop_motors():
    apply_wheels(0, 0, 0, 0)

# ===================== 算法可视化叠加 =====================
def draw_algorithm_overlay(frame, lane_result, state_output, infer_ms):
    """在帧上叠加算法调试信息（录制时使用）"""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # 左上角：状态 + 时间
    state = state_output.get("state", "?")
    action = car_state.get("action", "?")
    cv2.putText(vis, f"State: {state}  Action: {action}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(vis, f"Infer: {infer_ms:.1f}ms", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # 右上角：PID 数据
    if lane_result:
        err = lane_result.get("error", 0)
        pid = lane_result.get("pid", 0)
        mode = lane_result.get("mode", "?")
        is_curve = lane_result.get("is_curve", False)
        cv2.putText(vis, f"Err: {err:.1f}  PID: {pid:.3f}", (w-250, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(vis, f"Mode: {mode}  Curve: {is_curve}", (w-250, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # 画扫描线检测点
    if lane_result:
        candidates = lane_result.get("candidates", [])
        for y, xs in candidates:
            cv2.line(vis, (0, y), (w, y), (80, 80, 80), 1)
            for x in xs:
                cv2.circle(vis, (int(x), y), 5, (0, 255, 255), -1)

        # 目标点
        target_x = lane_result.get("target_x")
        if target_x is not None:
            target_y = int(h * 0.72)
            center_x = w // 2
            cv2.circle(vis, (int(target_x), target_y), 10, (0, 0, 255), -1)
            cv2.line(vis, (center_x, h), (int(target_x), target_y), (0, 0, 255), 2)

    # 四轮速度指示
    ws = car_state.get("wheel_speeds", {})
    y0 = h - 60
    for i, (key, label) in enumerate([("fl","FL"), ("fr","FR"), ("rl","RL"), ("rr","RR")]):
        v = ws.get(key, 0)
        color = (0, 200, 0) if v > 0 else (0, 0, 200) if v < 0 else (100, 100, 100)
        cv2.putText(vis, f"{label}:{v:+d}", (10 + i*110, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return vis

# ===================== smart 模式后台控制循环 =====================
_control_running = False

def control_loop():
    global _control_running
    _control_running = True
    global _follower, _state_machine, _sign_detector
    follower = _follower
    sm = _state_machine
    sm.reset()

    fps_counter = 0
    fps_time = time.time()
    frame_count = 0
    infer_times = []

    log.info("Smart control loop started.")
    print("[SC] 控制循环启动", flush=True)
    h_frame, w_frame = 480, 640  # 默认帧尺寸
    state_output = {"state": "INIT", "pending_sign": None, "is_approaching": False}

    try:
        while _control_running and car_state["mode"] == "smart":
            loop_start = time.time()
            frame = get_latest_frame()
            if frame is None:
                if frame_count == 0:
                    print("[SC] 等待摄像头帧...", flush=True)
                time.sleep(0.01)
                continue

            frame_count += 1
            h_frame, w_frame = frame.shape[:2]

            # 1. 巡迹感知
            sync_pid_to_follower()
            lane_result = follower.infer(frame)

            # 2. YOLO标志检测（低频）
            sign_result = None
            if _sign_detector:
                sign_result = _sign_detector.detect(frame)

            # 3. 状态机更新
            state_output = sm.update(lane_result, sign_result)

            # 4. 边界检测（在状态机更新之后）
            if state_output.get("is_approaching"):
                bottom_roi = frame[int(h_frame*0.85):, :]
                hsv_roi = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2HSV)
                yellow_mask = cv2.inRange(hsv_roi,
                                          np.array([12, 70, 80], dtype=np.uint8),
                                          np.array([45, 255, 255], dtype=np.uint8))
                yellow_count = cv2.countNonZero(yellow_mask)
                roi_area = yellow_mask.shape[0] * yellow_mask.shape[1]
                yellow_ratio = yellow_count / roi_area if roi_area > 0 else 0
                if yellow_ratio > 0.08:
                    sm.on_boundary_detected()
                    state_output = sm._build_output()
                    print(f"[SC] 边界触发: yellow={yellow_ratio:.1%}", flush=True)

            # 4. 计算电机指令
            state = state_output["state"]
            rr, fr, fl, rl = 0, 0, 0, 0
            action_name = "Stop"
            speed = 20

            if state == "STOP":
                action_name = "Stop"

            elif state == "APPROACH":
                # 向检测框中心运动, 速度23, 角度差速
                sign = state_output.get("pending_sign")
                target_x = state_output.get("approach_target_x")
                base_spd = 23
                if target_x is not None:
                    import math
                    dx = target_x - 320  # 图像中心=320
                    dy = 480 - state_output.get("approach_target_y", 240)
                    angle = math.degrees(math.atan2(dx, dy))
                    abs_a = abs(angle)
                    if abs_a < 3: diff = 0
                    elif abs_a < 8: diff = 1
                    elif abs_a < 15: diff = 2
                    elif abs_a < 25: diff = 3
                    else: diff = 4
                    if angle < 0: diff = -diff
                else:
                    diff = 0

                if diff >= 0:
                    left_spd = base_spd + diff
                    right_spd = base_spd
                else:
                    left_spd = base_spd
                    right_spd = base_spd - diff
                left_spd = max(22, min(40, left_spd))
                right_spd = max(22, min(40, right_spd))
                rr, fr, fl, rl = -right_spd, -right_spd, left_spd, left_spd
                action_name = f"Approach({sign},{diff:+d})"

            elif state == "PAUSE":
                # 停顿
                rr, fr, fl, rl = 0, 0, 0, 0
                action_name = "Pause"

            elif state == "TURN":
                # 旋转
                sign = state_output.get("pending_sign")
                turn_spd = 20
                if sign == "left":
                    rr, fr, fl, rl = turn_spd, turn_spd, turn_spd, turn_spd
                    action_name = "TurnLeft"
                elif sign == "right":
                    rr, fr, fl, rl = -turn_spd, -turn_spd, -turn_spd, -turn_spd
                    action_name = "TurnRight"
                elif sign == "turnaround":
                    rr, fr, fl, rl = turn_spd, turn_spd, turn_spd, turn_spd
                    action_name = "TurnAround"

            elif state == "CRUISE":
                if lane_result and lane_result.get("ok"):
                    candidates = lane_result.get("candidates", [])
                    has_cluster = any(len(xs) >= 2 and (xs[-1]-xs[0]) <= 640*0.20*0.3
                                      for y, xs in candidates)

                    # 找最近有检测的线 (从近往远)
                    nv = None
                    for i in range(len(candidates)-1, -1, -1):
                        if candidates[i][1]:
                            nv = i
                            break
                    # 找最近双点线
                    nd = None
                    for i in range(len(candidates)-1, -1, -1):
                        y, xs = candidates[i]
                        if len(xs) >= 2 and (xs[-1]-xs[0]) > 640*0.20*0.3:
                            nd = i
                            break

                    base_spd = 23 if has_cluster else 25

                    if nv is not None and nd is not None and nv != nd:
                        distance = nv - nd
                        nearest_xs = candidates[nv][1]
                        correction = -distance if nearest_xs[0] < 320 else distance
                    elif nv is not None:
                        nearest_xs = candidates[nv][1]
                        correction = -5 if nearest_xs[0] < 320 else 5
                    else:
                        correction = 0

                    # 一侧保持基础速度, 另一侧 ± correction
                    if correction >= 0:
                        left_spd = base_spd + correction
                        right_spd = base_spd
                    else:
                        left_spd = base_spd
                        right_spd = base_spd - correction
                    left_spd = max(22, min(40, left_spd))
                    right_spd = max(22, min(40, right_spd))
                    if abs(left_spd - right_spd) > 8:
                        if left_spd > right_spd:
                            left_spd = min(right_spd + 8, 40)
                        else:
                            right_spd = min(left_spd + 8, 40)

                    rr, fr, fl, rl = -right_spd, -right_spd, left_spd, left_spd
                    action_name = f"Diff({correction:+d})"
                else:
                    # 丢线 → 低速前进找回
                    base_spd = 12
                    rr, fr, fl, rl = -base_spd, -base_spd, base_spd, base_spd
                    action_name = "RecoverForward"

            # 5. 发送电机指令
            apply_wheels(rr, fr, fl, rl)

            # 每30帧输出一次运行日志
            if frame_count % 30 == 0:
                lk_x = state_output.get("approach_target_x", "-")
                msg = (f"#{frame_count} state={state} action={action_name} "
                       f"L={left_spd if 'left_spd' in dir() else '-'} "
                       f"R={right_spd if 'right_spd' in dir() else '-'} "
                       f"tgt={target_x if 'target_x' in dir() else '-'} "
                       f"sign={state_output.get('pending_sign', '-')}")
                print(f"[SC] {msg}")
                log_to_buffer(msg)
                car_state["fps"] = 30.0 / max(0.001, time.time() - fps_time)
                fps_time = time.time()

            # 6. 计算耗时
            infer_ms = (time.time() - loop_start) * 1000
            infer_times.append(infer_ms)
            if len(infer_times) > 300:
                infer_times = infer_times[-300:]

            # 7. 更新状态
            now = time.time()
            fps_counter += 1
            if now - fps_time >= 1.0:
                car_state["fps"] = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            car_state["state"] = state
            car_state["action"] = action_name
            car_state["frame_count"] = frame_count
            car_state["infer_time_ms"] = infer_ms
            if lane_result:
                car_state["pid_error"] = lane_result.get("error", 0)
                car_state["pid_output"] = lane_result.get("pid", 0)
                car_state["is_curve"] = lane_result.get("is_curve", False)
                car_state["speed_factor"] = lane_result.get("speed_factor", 1.0)
            if sign_result:
                car_state["signs_detected"] = [s["type"] for s in sign_result.get("signs", [])]

            # 8. 录制帧数同步（录制线程独立写入）
            if _recording:
                car_state["record_frames"] = _recording_frame_count

            time.sleep(0.01)

    except Exception as e:
        log.error(f"Control loop error: {e}")
    finally:
        stop_motors()
        if _recording:
            path, count = stop_recording()
            car_state["recording"] = False
            # 保存性能分析报告
            if infer_times:
                _save_analysis_report(path, infer_times)
        car_state["mode"] = "idle"
        car_state["state"] = "INIT"
        _control_running = False
        log.info("Smart control loop stopped.")

def _save_analysis_report(video_path, infer_times):
    """保存推理性能分析报告"""
    if not video_path or not infer_times:
        return
    report_path = video_path.replace(".mp4", "_analysis.txt")
    arr = np.array(infer_times)
    with open(report_path, "w") as f:
        f.write(f"视频文件: {os.path.basename(video_path)}\n")
        f.write(f"总帧数: {len(infer_times)}\n")
        f.write(f"推理耗时统计 (ms):\n")
        f.write(f"  平均: {arr.mean():.2f}\n")
        f.write(f"  中位数: {np.median(arr):.2f}\n")
        f.write(f"  最小: {arr.min():.2f}\n")
        f.write(f"  最大: {arr.max():.2f}\n")
        f.write(f"  P95: {np.percentile(arr, 95):.2f}\n")
        f.write(f"  P99: {np.percentile(arr, 99):.2f}\n")
        f.write(f"  标准差: {arr.std():.2f}\n")
        f.write(f"  理论最大FPS: {1000/arr.mean():.1f}\n")
    log.info(f"Analysis report saved: {report_path}")

# ===================== 路由 =====================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    def generate():
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera initializing...", (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        while True:
            frame = get_latest_frame()
            if frame is None:
                frame = placeholder

            if _recording:
                car_state["record_frames"] = _recording_frame_count

            if _recording and frame is not placeholder:
                disp = frame.copy()
                cv2.circle(disp, (30, 30), 12, (0, 0, 255), -1)
                cv2.putText(disp, f"REC {_recording_frame_count}", (48, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                _, buffer = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 60])
            else:
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + buffer.tobytes() + b"\r\n")
            time.sleep(0.033)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

_log_buffer = []
_log_lock = threading.Lock()

def log_to_buffer(msg):
    with _log_lock:
        _log_buffer.append({"time": time.strftime("%H:%M:%S"), "msg": msg})
        if len(_log_buffer) > 100:
            _log_buffer.pop(0)

# ===================== smart 子进程管理 =====================
_smart_proc = None
_smart_pty_master = None

def _smart_output_reader(proc):
    for line in iter(proc.stdout.readline, ''):
        if not line: break
        msg = line.strip()
        if msg:
            with _log_lock:
                _log_buffer.append({"time": time.strftime("%H:%M:%S"), "msg": msg})
                if len(_log_buffer) > 200:
                    _log_buffer[:] = _log_buffer[-200:]

def _monitor_smart_process(proc):
    global _smart_proc, _smart_pty_master
    return_code = proc.wait()
    if proc is not _smart_proc:
        return
    _smart_proc = None
    if _smart_pty_master is not None:
        try:
            os.close(_smart_pty_master)
        except OSError:
            pass
        _smart_pty_master = None
    car_state["mode"] = "idle"
    car_state["state"] = "INIT"
    start_web_camera()
    log_to_buffer(f"智能巡航进程已退出 (code={return_code})，摄像头预览已恢复")

def start_smart():
    global _smart_proc, _smart_pty_master
    if _smart_proc and _smart_proc.poll() is None:
        return
    if pty is None:
        log_to_buffer("当前系统不支持伪终端，无法启动智能巡航")
        return

    # 释放 Web 摄像头，让 main.py 的 CameraBroadcaster 独占设备。
    stop_web_camera()
    log_to_buffer("Web 摄像头已释放")
    time.sleep(0.5)

    main_py = os.path.join(PROJECT_DIR, "main.py")
    master_fd, slave_fd = pty.openpty()
    try:
        _smart_proc = subprocess.Popen(
            [sys.executable, main_py, "--mode", "smart"],
            stdin=slave_fd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_DIR,
            bufsize=1,
            universal_newlines=True,
        )
    except Exception:
        os.close(master_fd)
        os.close(slave_fd)
        start_web_camera()
        raise
    finally:
        try:
            os.close(slave_fd)
        except OSError:
            pass
    _smart_pty_master = master_fd
    car_state["mode"] = "smart"
    car_state["state"] = "RUNNING"
    log_to_buffer("智能巡航已启动")
    threading.Thread(target=_smart_output_reader, args=(_smart_proc,), daemon=True).start()
    threading.Thread(target=_monitor_smart_process, args=(_smart_proc,), daemon=True).start()

def stop_smart():
    global _smart_proc, _smart_pty_master
    # 写停止信号文件，SmartCruise循环会检测到并自行停车
    stop_file = os.path.join(PROJECT_DIR, ".smart_stop")
    try:
        with open(stop_file, "w") as f:
            f.write("stop")
    except: pass

    proc = _smart_proc
    if proc and proc.poll() is None:
        if _smart_pty_master is not None:
            try:
                os.write(_smart_pty_master, b"\x1b")
            except OSError:
                pass
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)

    # 清理信号文件
    try: os.remove(stop_file)
    except: pass

    car_state["mode"] = "idle"
    car_state["state"] = "INIT"
    start_web_camera()
    log_to_buffer("智能巡航已停止")

@app.route("/api/state")
def get_state():
    return jsonify(car_state)


def _behavior_request(method, path, **kwargs):
    kwargs.setdefault("timeout", 8)
    return requests.request(
        method,
        f"{BEHAVIOR_SERVICE_URL}{path}",
        **kwargs,
    )


def _behavior_proxy_response(response, default_type="application/json"):
    return Response(
        response.content,
        status=response.status_code,
        content_type=response.headers.get("Content-Type", default_type),
    )


@app.route("/api/behavior/health")
def behavior_health():
    try:
        return _behavior_proxy_response(
            _behavior_request("GET", "/api/health")
        )
    except requests.RequestException as error:
        return jsonify({"ok": False, "error": str(error)}), 503


@app.route("/api/behavior/status")
def behavior_status():
    try:
        return _behavior_proxy_response(
            _behavior_request("GET", "/api/status")
        )
    except requests.RequestException as error:
        return jsonify({
            "running": False,
            "state": "offline",
            "error": str(error),
        }), 503


@app.route("/api/behavior/alarms")
def behavior_alarms():
    try:
        return _behavior_proxy_response(
            _behavior_request("GET", "/api/alarms")
        )
    except requests.RequestException as error:
        return jsonify({"error": str(error), "alarms": []}), 503


@app.route("/api/behavior/alarms/<event_id>/image")
def behavior_alarm_image(event_id):
    if not event_id.replace("_", "").isalnum():
        return "", 400
    try:
        response = _behavior_request(
            "GET",
            f"/api/alarms/{event_id}/image",
            timeout=15,
        )
        return _behavior_proxy_response(response, "image/jpeg")
    except requests.RequestException:
        return "", 503


@app.route("/api/behavior/upload", methods=["POST"])
def behavior_upload():
    video = request.files.get("video")
    if video is None:
        return jsonify({"ok": False, "error": "请选择视频文件"}), 400
    try:
        response = _behavior_request(
            "POST",
            "/api/upload",
            files={
                "video": (
                    video.filename,
                    video.stream,
                    video.mimetype or "application/octet-stream",
                )
            },
            timeout=600,
        )
        return _behavior_proxy_response(response)
    except requests.RequestException as error:
        return jsonify({"ok": False, "error": str(error)}), 503


@app.route("/api/behavior/live", methods=["POST"])
def behavior_live():
    try:
        response = _behavior_request(
            "POST",
            "/api/live",
            json=request.get_json(silent=True) or {"source": 0},
        )
        return _behavior_proxy_response(response)
    except requests.RequestException as error:
        return jsonify({"ok": False, "error": str(error)}), 503


@app.route("/api/behavior/stop", methods=["POST"])
def behavior_stop():
    try:
        return _behavior_proxy_response(
            _behavior_request("POST", "/api/stop")
        )
    except requests.RequestException as error:
        return jsonify({"ok": False, "error": str(error)}), 503


@app.route("/api/behavior/stream")
def behavior_stream():
    try:
        upstream = requests.get(
            f"{BEHAVIOR_SERVICE_URL}/api/stream",
            stream=True,
            timeout=(5, None),
        )
        upstream.raise_for_status()
    except requests.RequestException:
        return "", 503

    def generate():
        try:
            for chunk in upstream.iter_content(chunk_size=16 * 1024):
                if chunk:
                    yield chunk
        finally:
            upstream.close()

    return Response(
        generate(),
        content_type=upstream.headers.get(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame",
        ),
    )


@app.route("/api/log")
def get_log():
    with _log_lock:
        return jsonify(list(_log_buffer))


@app.route("/api/detect/image")
def detect_image():
    """返回最新检测标注图"""
    detect_dir = os.path.join(PROJECT_DIR, "capture", "detect")
    if not os.path.exists(detect_dir):
        return "", 404
    files = sorted(glob.glob(os.path.join(detect_dir, "detect_*.jpg")), key=os.path.getmtime, reverse=True)
    if not files:
        return "", 404
    return send_file(files[0], mimetype="image/jpeg")

@app.route("/api/pid", methods=["GET", "POST"])
def api_pid():
    if request.method == "POST":
        pid_params.update(request.json)
        sync_pid_to_follower()
        return jsonify({"ok": True, "pid": pid_params})
    return jsonify(pid_params)

@app.route("/api/hsv", methods=["GET", "POST"])
def api_hsv():
    if request.method == "POST":
        hsv_params.update(request.json)
        return jsonify({"ok": True, "hsv": hsv_params})
    return jsonify(hsv_params)

@app.route("/api/wheels", methods=["GET", "POST"])
def api_wheels():
    if request.method == "POST":
        data = request.json
        apply_wheels(int(data.get("rr",0)), int(data.get("fr",0)),
                     int(data.get("fl",0)), int(data.get("rl",0)))
        return jsonify({"ok": True, "wheels": wheel_speeds})
    return jsonify(wheel_speeds)

@app.route("/api/detect", methods=["POST"])
def api_detect():
    """用子进程运行YOLO检测（独立NPU上下文）"""
    frame = get_latest_frame()
    if frame is None:
        return jsonify({"ok": False, "error": "no frame"})

    # 保存当前帧到临时文件
    tmp_path = os.path.join(PROJECT_DIR, "capture", "detect", "_tmp_input.jpg")
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    cv2.imwrite(tmp_path, frame)

    # 用子进程调用检测脚本
    detect_script = os.path.join(PROJECT_DIR, "web", "detect_one.py")
    if not os.path.exists(detect_script):
        return jsonify({"ok": False, "error": f"检测脚本不存在: {detect_script}"})

    try:
        result = subprocess.run(
            [sys.executable, detect_script, tmp_path],
            capture_output=True, text=True, timeout=10, cwd=PROJECT_DIR,
        )
        if result.returncode != 0:
            return jsonify({"ok": False, "error": result.stderr[-200:]})
        output = json.loads(result.stdout)
        return jsonify(output)
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "检测超时"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/servo", methods=["POST"])
def api_servo():
    angle = int(request.json.get("angle", 90))
    angle = max(0, min(180, angle))
    if ctrl is not None:
        with _ctrl_lock:
            try:
                ctrl.send_servo([angle, angle])
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)})
    car_state["servo_angle"] = [angle, angle]
    return jsonify({"ok": True, "angle": angle})


@app.route("/api/debug", methods=["GET", "POST"])
def api_debug():
    """获取或设置debug模式"""
    if request.method == "POST":
        data = request.json
        debug_enabled = data.get("enabled", False)
        car_state["debug_mode"] = debug_enabled
        # 写入环境变量，供子进程读取
        os.environ["SMART_CAR_DEBUG"] = "1" if debug_enabled else "0"
        log.info(f"Debug mode: {'ON' if debug_enabled else 'OFF'}")
        return jsonify({"ok": True, "debug_mode": debug_enabled})
    return jsonify({"debug_mode": car_state["debug_mode"]})


@app.route("/api/debug/clean", methods=["POST"])
def api_debug_clean():
    """删除所有debug图像文件"""
    import shutil
    debug_dirs = [
        os.path.join(PROJECT_DIR, "capture", "smart_debug"),
        os.path.join(PROJECT_DIR, "capture", "detect"),
        os.path.join(PROJECT_DIR, "capture", "web"),
    ]
    total_deleted = 0
    errors = []
    for debug_dir in debug_dirs:
        if os.path.exists(debug_dir):
            try:
                count = 0
                for root, dirs, files in os.walk(debug_dir):
                    for f in files:
                        if f.endswith(('.jpg', '.jpeg', '.png')):
                            os.remove(os.path.join(root, f))
                            count += 1
                total_deleted += count
                log.info(f"Cleaned {count} images from {debug_dir}")
            except Exception as e:
                errors.append(f"{debug_dir}: {str(e)}")
                log.error(f"Clean failed for {debug_dir}: {e}")
    return jsonify({
        "ok": len(errors) == 0,
        "deleted": total_deleted,
        "errors": errors,
    })

# ===================== WebSocket =====================
@socketio.on("connect")
def handle_connect():
    log.info("WebSocket client connected")

@socketio.on("command")
def handle_command(data):
    global _control_running
    action = data.get("action")

    if action == "set_mode":
        new_mode = data.get("mode", "idle")
        old_mode = car_state["mode"]
        if new_mode == "smart" and old_mode != "smart":
            start_smart()
        elif new_mode != "smart" and old_mode == "smart":
            stop_smart()
        else:
            car_state["mode"] = new_mode
        socketio.emit("state_update", car_state)

    elif action == "set_pid":
        pid_params.update({
            "kp": data.get("kp", pid_params["kp"]),
            "ki": data.get("ki", pid_params["ki"]),
            "kd": data.get("kd", pid_params["kd"]),
            "smooth_alpha": data.get("smooth_alpha", pid_params["smooth_alpha"]),
            "center_deadband_px": data.get("center_deadband_px", pid_params["center_deadband_px"]),
        })
        sync_pid_to_follower()
        socketio.emit("pid_update", pid_params)

    elif action == "set_stop_threshold":
        try:
            threshold_cm = max(5.0, min(50.0, float(data.get("threshold_cm", 20))))
            if ctrl is not None:
                with _ctrl_lock:
                    _, obstacle_info = ctrl.set_stop_threshold(threshold_cm)
                threshold_cm = float(obstacle_info.get("stop_threshold_cm", threshold_cm))
            car_state["stop_threshold_cm"] = threshold_cm
            socketio.emit("threshold_update", {"ok": True, "threshold_cm": threshold_cm})
            socketio.emit("state_update", car_state)
        except Exception as e:
            log.error(f"Stop threshold update failed: {e}")
            socketio.emit("threshold_update", {
                "ok": False,
                "threshold_cm": car_state["stop_threshold_cm"],
                "error": str(e),
            })

    elif action == "motor":
        if car_state["mode"] == "smart":
            return
        direction = data.get("direction", "stop")
        speed = data.get("speed", 20)
        rr, fr, fl, rl = direction_to_wheels(direction, speed)
        apply_wheels(rr, fr, fl, rl)
        car_state["action"] = f"Manual_{direction}"
        socketio.emit("state_update", car_state)

    elif action == "motor_raw":
        if car_state["mode"] == "smart":
            return
        w = data.get("wheels", {})
        apply_wheels(int(w.get("rr",0)), int(w.get("fr",0)),
                     int(w.get("fl",0)), int(w.get("rl",0)))
        socketio.emit("state_update", car_state)

    elif action == "record_start":
        if not _recording:
            start_recording()
            car_state["recording"] = True
            car_state["record_frames"] = 0
            socketio.emit("state_update", car_state)

    elif action == "record_stop":
        if _recording:
            path, count = stop_recording()
            car_state["recording"] = False
            car_state["record_frames"] = 0
            socketio.emit("record_done", {"path": path, "frames": count})
            socketio.emit("state_update", car_state)

    elif action == "capture":
        frame = get_latest_frame()
        if frame is not None:
            save_dir = os.path.join(PROJECT_DIR, "capture", "web")
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"capture_{int(time.time())}.jpg")
            cv2.imwrite(path, frame)
            socketio.emit("capture_done", {"path": path})

    elif action == "servo":
        channel = data.get("channel", 0)  # 1=S1俯仰, 2=S2航向, 0=两个都设
        angle = int(max(0, min(180, data.get("angle", 90))))
        if ctrl is not None:
            with _ctrl_lock:
                try:
                    current = list(car_state.get("servo_angle", [110, 50]))
                    if channel == 1:
                        current[0] = angle
                    elif channel == 2:
                        current[1] = angle
                    else:
                        current = [angle, angle]
                    ctrl.send_servo(current)
                    car_state["servo_angle"] = current
                except Exception as e:
                    log.error(f"Servo command failed: {e}")
        else:
            current = list(car_state.get("servo_angle", [110, 50]))
            if channel == 1:
                current[0] = angle
            elif channel == 2:
                current[1] = angle
            car_state["servo_angle"] = current
        socketio.emit("state_update", car_state)

    elif action == "set_debug":
        debug_enabled = data.get("enabled", False)
        car_state["debug_mode"] = debug_enabled
        os.environ["SMART_CAR_DEBUG"] = "1" if debug_enabled else "0"
        log.info(f"Debug mode: {'ON' if debug_enabled else 'OFF'}")
        socketio.emit("state_update", car_state)

    elif action == "clean_debug":
        import shutil
        debug_dirs = [
            os.path.join(PROJECT_DIR, "capture", "smart_debug"),
            os.path.join(PROJECT_DIR, "capture", "detect"),
            os.path.join(PROJECT_DIR, "capture", "web"),
        ]
        total_deleted = 0
        for debug_dir in debug_dirs:
            if os.path.exists(debug_dir):
                try:
                    for root, dirs, files in os.walk(debug_dir):
                        for f in files:
                            if f.endswith(('.jpg', '.jpeg', '.png')):
                                os.remove(os.path.join(root, f))
                                total_deleted += 1
                except Exception as e:
                    log.error(f"Clean failed for {debug_dir}: {e}")
        socketio.emit("clean_done", {"deleted": total_deleted})
        showNotification(f"已删除 {total_deleted} 张图像")


def broadcast_state():
    while True:
        socketio.emit("state_update", car_state)
        time.sleep(0.1)

if __name__ == "__main__":
    # 启动时预加载所有模型
    init_models()

    threading.Thread(target=broadcast_state, daemon=True).start()
    mode_str = "PREVIEW (no ESP32)" if preview_mode else "LIVE (ESP32 connected)"
    print("=" * 50)
    print(f"  智能小车 Web 控制台 [{mode_str}]")
    print("  访问 http://0.0.0.0:5000")
    print("  点击 [🧠 智能巡航] 启动自动巡迹")
    print("=" * 50)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False,
                 allow_unsafe_werkzeug=True)
