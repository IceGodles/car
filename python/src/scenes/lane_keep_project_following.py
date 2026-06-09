#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time

import cv2
import numpy as np

from src.actions import Advance, SetServo, Start, Stop, TurnLeft, TurnRight
from src.scenes.base_scene import BaseScene
from src.utils import CAMERA_SERVO_ANGLE, log
from src.utils.lane_keep import LaneKeepFollower


class LKP(BaseScene):
    def __init__(self, memory_name, camera_info, msg_queue):
        super().__init__(memory_name, camera_info, msg_queue)
        self.forward_spd = 28
        self.follower = LaneKeepFollower()
        self.debug_dir = os.path.join(os.getcwd(), "capture", "lkp")
        os.makedirs(self.debug_dir, exist_ok=True)
        self.records = []

    def init_state(self):
        log.info(f'start init {self.__class__.__name__}')
        self.ctrl.execute(SetServo(servo=CAMERA_SERVO_ANGLE))
        log.info(f'{self.__class__.__name__} init succ.')
        return False

    def _action_from_pid(self, pid_output):
        degree = min(0.8, max(0.1, abs(pid_output)))
        if pid_output < 0:
            return TurnRight(speed=self.forward_spd, degree=degree)
        if pid_output > 0:
            return TurnLeft(speed=self.forward_spd, degree=degree)
        return Advance(speed=self.forward_spd)

    def _append_record(self, result):
        record = {
            "index": len(self.records),
            "time": time.time(),
            "ok": bool(result["ok"]),
            "mode": result["mode"],
            "error": float(result["error"]),
            "pid": float(result["pid"]),
            "target_x": "" if result["target_x"] is None else float(result["target_x"]),
            "road_area": int(result.get("road_area", 0)),
            "yellow_area": int(result.get("yellow_area", 0)),
        }
        self.records.append(record)
        return record

    def _draw_path(self):
        width, height = 900, 900
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)
        origin = np.array([width / 2.0, height - 60.0], dtype=np.float32)
        points = [origin.copy()]
        pos = origin.copy()
        heading = -np.pi / 2.0
        step = 16.0

        for record in self.records[-500:]:
            if record["ok"]:
                heading += record["pid"] * 0.08
            direction = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
            pos = pos + direction * step
            pos[0] = np.clip(pos[0], 20, width - 20)
            pos[1] = np.clip(pos[1], 20, height - 20)
            points.append(pos.copy())

        cv2.line(canvas, (width // 2, height - 40), (width // 2, 40), (210, 210, 210), 2)
        active = self.records[-500:]
        for idx in range(1, len(points)):
            p0 = tuple(points[idx - 1].astype(int))
            p1 = tuple(points[idx].astype(int))
            color = (30, 130, 255) if active[idx - 1]["ok"] else (80, 80, 80)
            cv2.line(canvas, p0, p1, color, 3)
        if points:
            cv2.circle(canvas, tuple(points[0].astype(int)), 8, (0, 200, 0), -1)
            cv2.circle(canvas, tuple(points[-1].astype(int)), 8, (0, 0, 255), -1)
        cv2.putText(canvas, "LKP predicted path", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
        return canvas

    def _save_records(self):
        csv_path = os.path.join(self.debug_dir, "lkp_records.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("index,time,ok,mode,error,pid,target_x,road_area,yellow_area\n")
            for record in self.records[-1000:]:
                f.write(
                    "{index},{time:.3f},{ok},{mode},{error:.3f},{pid:.6f},{target_x},{road_area},{yellow_area}\n"
                    .format(**record)
                )

    def _save_debug(self, img_bgr, result):
        debug = self.follower.draw_debug(img_bgr, result)
        cv2.imwrite(os.path.join(self.debug_dir, "latest.jpg"), debug)
        cv2.imwrite(os.path.join(self.debug_dir, "latest_mask.jpg"), result["mask"])
        cv2.imwrite(os.path.join(self.debug_dir, "latest_yellow_mask.jpg"), result["yellow_mask"])
        cv2.imwrite(os.path.join(self.debug_dir, "latest_road_mask.jpg"), result["road_mask"])
        cv2.imwrite(os.path.join(self.debug_dir, "predicted_path.jpg"), self._draw_path())
        self._save_records()

    def loop(self):
        ret = self.init_state()
        if ret:
            log.error(f'{self.__class__.__name__} init failed.')
            return

        frame = np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=self.broadcaster.buf)
        log.info(f'{self.__class__.__name__} loop start')
        self.ctrl.execute(Start())
        last_debug_save = time.time()

        try:
            while True:
                if self.stop_sign.value:
                    break
                if self.pause_sign.value:
                    continue

                start = time.time()
                img_bgr = frame.copy()
                result = self.follower.infer(img_bgr)
                log.info(
                    'lkp: ok=%s mode=%s error=%.1f pid=%.3f road=%s yellow=%s',
                    result['ok'],
                    result['mode'],
                    result['error'],
                    result['pid'],
                    result.get('road_area', 0),
                    result.get('yellow_area', 0),
                )
                self._append_record(result)

                if result["ok"]:
                    action = self._action_from_pid(result["pid"])
                else:
                    action = Stop()
                self.ctrl.execute(action)

                now = time.time()
                if now - last_debug_save > 1.0:
                    self._save_debug(img_bgr, result)
                    last_debug_save = now

                log.info(f'lkp cost {time.time() - start}')
        except KeyboardInterrupt:
            self.ctrl.execute(Stop())
        finally:
            if self.records:
                self._save_records()
                cv2.imwrite(os.path.join(self.debug_dir, "predicted_path.jpg"), self._draw_path())
