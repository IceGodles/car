#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time

import cv2
import numpy as np
from src.actions import SetServo, Stop, TurnLeftInPlace, TurnRightInPlace, TurnAround
from src.models import YoloV5
from src.scenes.base_scene import BaseScene
from src.utils import log
from src.utils.constant import CAMERA_SERVO_ANGLE


class Helper(BaseScene):
    def __init__(self, memory_name, camera_info, msg_queue):
        super().__init__(memory_name, camera_info, msg_queue)
        self.det = None
        self.cls = None
        self.save_dir = os.path.join(os.getcwd(), "capture", "helper_yolo")
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_index = 0
        self.last_save_time = 0.0
        self.roi_keep_height_ratio = 0.6

    def _bottom_roi(self, img_bgr):
        height = img_bgr.shape[0]
        roi_top = int(height * (1 - self.roi_keep_height_ratio))
        return img_bgr[roi_top:, :].copy(), roi_top

    @staticmethod
    def _map_bboxes_to_full_frame(bboxes, y_offset):
        mapped = []
        for x1, y1, x2, y2, cate, score in bboxes:
            mapped.append([x1, y1 + y_offset, x2, y2 + y_offset, cate, score])
        return mapped

    def init_state(self):
        log.info(f'start init {self.__class__.__name__}')
        det_path = os.path.join(os.getcwd(), 'weights', 'yolo.om')
        if not os.path.exists(det_path):
            log.error(f'Cannot find the offline inference model(.om) file needed for {self.__class__.__name__}  scene.')
            return True
        self.det = YoloV5(det_path)
        log.info(f'{self.__class__.__name__} model init succ.')
        self.ctrl.execute(SetServo(servo=CAMERA_SERVO_ANGLE))
        return False

    def _save_yolo_result(self, img_bgr, bboxes, prefix="det"):
        annotated = img_bgr.copy()
        for x1, y1, x2, y2, cate, score in bboxes:
            x1 = max(0, min(int(x1), self.width - 1))
            y1 = max(0, min(int(y1), self.height - 1))
            x2 = max(0, min(int(x2), self.width - 1))
            y2 = max(0, min(int(y2), self.height - 1))
            label = f"{cate} {float(score):.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_y = max(y1, th + baseline + 4)
            cv2.rectangle(annotated, (x1, text_y - th - baseline - 4),
                          (x1 + tw + 6, text_y), (0, 255, 255), -1)
            cv2.putText(annotated, label, (x1 + 3, text_y - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        name = f"{prefix}_{self.save_index:06d}_{int(time.time() * 1000)}.jpg"
        path = os.path.join(self.save_dir, name)
        cv2.imwrite(path, annotated)
        cv2.imwrite(os.path.join(self.save_dir, "latest.jpg"), annotated)
        self.save_index += 1
        self.last_save_time = time.time()
        log.info(f'save yolo result: {path}')

    def loop(self):
        ret = self.init_state()
        if ret:
            log.error(f'{self.__class__.__name__} init failed.')
            return
        frame = np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=self.broadcaster.buf)
        log.info(f'{self.__class__.__name__} loop start')
        log.info(f'{self.__class__.__name__} uses bottom {self.roi_keep_height_ratio:.0%} camera ROI for inference')
        last_action = None
        try:
            while True:
                if self.stop_sign.value:
                    break
                if self.pause_sign.value:
                    continue
                start = time.time()
                img_bgr = frame.copy()
                roi_bgr, roi_top = self._bottom_roi(img_bgr)
                bboxes = self._map_bboxes_to_full_frame(self.det.infer(roi_bgr), roi_top)
                log.info(f'{bboxes}')
                bboxes = sorted(bboxes, key=lambda x: x[5], reverse=True)
                if bboxes and time.time() - self.last_save_time > 0.5:
                    self._save_yolo_result(img_bgr, bboxes)
                for x1, y1, x2, y2, cate, score in bboxes:
                    x, y = (x1 + x2) // 2, (y1 + y2) // 2
                    h, w = y2 - y1, x2 - x1
                    log.info(f'det: {cate}')
                    if last_action != cate and len(bboxes) > 1:
                        cate = last_action

                    if cate == 'left':
                        if 300 < x < 1000 and y >= 450:
                            self._save_yolo_result(img_bgr, bboxes, prefix="trigger_left")
                            self.ctrl.execute(TurnLeftInPlace())
                            time.sleep(1)
                            break

                    if cate == 'right':
                        if 420 < x < 950 and y >= 600:
                            self._save_yolo_result(img_bgr, bboxes, prefix="trigger_right")
                            self.ctrl.execute(TurnRightInPlace())
                            time.sleep(1)
                            break

                    if cate == 'turnaround':
                        if 420 < x < 800 and y > 350:
                            self._save_yolo_result(img_bgr, bboxes, prefix="trigger_turnaround")
                            self.ctrl.execute(TurnAround())
                            time.sleep(2)
                            break
                    last_action = cate
                    break
                log.info(f'infer cost {time.time() - start}')
        except KeyboardInterrupt:
            self.ctrl.execute(Stop())
