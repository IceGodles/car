#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行为状态机
CRUISE → APPROACH(1s) → PAUSE(0.3s) → TURN(1.5s) → CRUISE
"""
import time
from enum import Enum, auto


class DrivingState(Enum):
    INIT = auto()
    CRUISE = auto()     # 正常巡迹
    APPROACH = auto()   # 接近目标(1秒)
    PAUSE = auto()      # 短暂停顿(0.3秒)
    TURN = auto()       # 旋转(1.5秒)
    STOP = auto()


class DrivingStateMachine:
    def __init__(self, config=None):
        config = config or {}
        self.state = DrivingState.INIT
        self.state_enter_time = time.time()
        self.approach_duration = config.get("approach_duration", 1.0)
        self.pause_duration = config.get("pause_duration", 0.3)
        self.turn_duration = config.get("turn_duration", 1.5)
        self.approach_target_x = None
        self.approach_target_y = None
        self.pending_sign = None
        self._boundary_hit = False
        self._state_log = []

    @property
    def elapsed(self):
        return time.time() - self.state_enter_time

    def _enter_state(self, new_state, reason=""):
        old = self.state
        self.state = new_state
        self.state_enter_time = time.time()
        self._state_log.append({"from": old.name, "to": new_state.name,
                                 "reason": reason, "time": time.time()})
        if len(self._state_log) > 50:
            self._state_log = self._state_log[-50:]

    def on_sign_detected(self, sign_type, target_x=None, target_y=None):
        """YOLO检测到标志牌"""
        if self.state in (DrivingState.CRUISE, DrivingState.INIT):
            self.pending_sign = sign_type
            self.approach_target_x = target_x
            self.approach_target_y = target_y
            self._enter_state(DrivingState.APPROACH, reason=f"detect:{sign_type}")

    def on_boundary_detected(self):
        """检测到黄线边界 → 立即停止approach"""
        self._boundary_hit = True
        if self.state == DrivingState.APPROACH:
            self._enter_state(DrivingState.PAUSE, reason="boundary_hit")
        elif self.state == DrivingState.CRUISE:
            self._enter_state(DrivingState.TURN, reason="boundary_in_cruise")

    def on_stop_sign(self):
        self._enter_state(DrivingState.STOP, reason="stop_sign")

    def update(self, lane_result=None, sign_result=None):
        # 处理停车标志
        if sign_result and sign_result.get("stop"):
            self.on_stop_sign()
            return self._build_output()

        # 处理转向标志
        if sign_result and self.state in (DrivingState.CRUISE, DrivingState.INIT):
            for sign in sign_result.get("signs", []):
                if sign["type"] in ("left", "right", "turnaround"):
                    self.on_sign_detected(sign["type"], sign.get("x"), sign.get("y"))
                    return self._build_output()

        # APPROACH 中允许更近的高置信新识别修正最初误判的方向牌。
        if sign_result and self.state == DrivingState.APPROACH:
            for sign in sign_result.get("signs", []):
                sign_type = sign.get("type")
                if sign_type not in ("left", "right", "turnaround"):
                    continue
                score = float(sign.get("score", 0.0))
                y = sign.get("y")
                old_y = self.approach_target_y
                is_closer = old_y is None or y is None or y >= old_y + 40
                if sign_type != self.pending_sign and score >= 0.90 and is_closer:
                    self.pending_sign = sign_type
                    self.approach_target_x = sign.get("x")
                    self.approach_target_y = y
                    self._state_log.append({
                        "from": self.state.name,
                        "to": self.state.name,
                        "reason": f"update_sign:{sign_type}",
                        "time": time.time(),
                    })
                    if len(self._state_log) > 50:
                        self._state_log = self._state_log[-50:]
                    return self._build_output()

        # 状态转移
        if self.state == DrivingState.INIT:
            if lane_result and lane_result.get("ok"):
                self._boundary_hit = False
                self._enter_state(DrivingState.CRUISE, reason="lane_found")
            elif self.elapsed > 2.0:
                self._boundary_hit = False
                self._enter_state(DrivingState.CRUISE, reason="init_timeout")

        elif self.state == DrivingState.CRUISE:
            pass  # 由外部巡迹逻辑控制

        elif self.state == DrivingState.APPROACH:
            # 接近目标 1 秒后停顿
            if self.elapsed >= self.approach_duration:
                self._enter_state(DrivingState.PAUSE, reason="approach_done")

        elif self.state == DrivingState.PAUSE:
            # 停顿 0.3 秒后旋转
            if self.elapsed >= self.pause_duration:
                self._enter_state(DrivingState.TURN, reason="pause_done")

        elif self.state == DrivingState.TURN:
            # 旋转 1.5 秒后，等边界消失再恢复巡迹
            if self.elapsed >= self.turn_duration:
                if not self._boundary_hit:
                    self.pending_sign = None
                    self.approach_target_x = None
                    self.approach_target_y = None
                    self._enter_state(DrivingState.CRUISE, reason="turn_done")
                elif self.elapsed >= self.turn_duration + 1.0:
                    # 边界还在 → 强制回CRUISE让它重新寻路
                    self.pending_sign = None
                    self.approach_target_x = None
                    self.approach_target_y = None
                    self._enter_state(DrivingState.CRUISE, reason="turn_force_done")

        elif self.state == DrivingState.STOP:
            pass

        return self._build_output()

    def _build_output(self):
        return {
            "state": self.state.name,
            "elapsed": self.elapsed,
            "pending_sign": self.pending_sign,
            "approach_target_x": self.approach_target_x,
            "approach_target_y": self.approach_target_y,
            "is_approaching": self.state == DrivingState.APPROACH,
            "is_turning": self.state == DrivingState.TURN,
            "is_cruising": self.state == DrivingState.CRUISE,
            "is_stopped": self.state == DrivingState.STOP,
            "recent_log": self._state_log[-5:],
        }

    def reset(self):
        self._enter_state(DrivingState.INIT, reason="manual_reset")
        self.pending_sign = None
        self.approach_target_x = None
