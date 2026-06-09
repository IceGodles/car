#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time

import numpy as np

from src.actions import SetServo, Stop, Start, TurnLeft, TurnRight, Advance, TurnAround
from src.models import LFNet
from src.scenes.base_scene import BaseScene
from src.utils import log
from src.utils.constant import CAMERA_SERVO_ANGLE


class LF(BaseScene):
    def __init__(self, memory_name, camera_info, msg_queue):
        super().__init__(memory_name, camera_info, msg_queue)
        self.net = None
        self.forward_spd = 28
        self.roi_keep_height_ratio = 1.0

    def _bottom_roi(self, img_bgr):
        height = img_bgr.shape[0]
        roi_top = int(height * (1 - self.roi_keep_height_ratio))
        return img_bgr[roi_top:, :].copy()

    def init_state(self):
        log.info(f'start init {self.__class__.__name__}')
        lfnet_path = os.path.join(os.getcwd(), 'weights', 'lfnet.om')
        if not os.path.exists(lfnet_path):
            log.error(f'Cannot find the offline inference model(.om) file needed for {self.__class__.__name__}  scene.')
            return True
        self.net = LFNet(lfnet_path)
        log.info(f'{self.__class__.__name__} model init succ.')
        self.ctrl.execute(SetServo(servo=CAMERA_SERVO_ANGLE))
        return False

    def loop(self):
        ret = self.init_state()
        if ret:
            log.error(f'{self.__class__.__name__} init failed.')
            return
        frame = np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=self.broadcaster.buf)
        log.info(f'{self.__class__.__name__} loop start')
        log.info(f'{self.__class__.__name__} uses full camera frame for inference')
        self.ctrl.execute(Start())
        try:
            while True:
                if self.stop_sign.value:
                    break
                if self.pause_sign.value:
                    continue
                start = time.time()
                img_bgr = frame.copy()
                output = self.net.infer(img_bgr)
                if output is None:
                    log.error('lfnet inference returned None, stop LF scene.')
                    self.ctrl.execute(Stop())
                    break
                curr_steering_val = float(output[0])
                log.info(f'lfnet: {curr_steering_val}')

                if curr_steering_val <= 0 or curr_steering_val > 135:
                    log.warning(f'lfnet output out of range, keep advance: {curr_steering_val}')
                    self.ctrl.execute(Advance(speed=self.forward_spd))
                elif curr_steering_val <= 65:
                    self.ctrl.execute(Advance(speed=self.forward_spd))
                elif 65 < curr_steering_val <= 75:
                    self.ctrl.execute(TurnLeft(speed=self.forward_spd, degree=0.7))
                elif 75 < curr_steering_val <= 80:
                    self.ctrl.execute(TurnLeft(speed=self.forward_spd, degree=0.6))
                elif 80 < curr_steering_val <= 85:
                    self.ctrl.execute(TurnLeft(speed=self.forward_spd, degree=0.5))
                elif 85 < curr_steering_val <= 88:
                    self.ctrl.execute(TurnLeft(speed=self.forward_spd, degree=0.4))
                elif 88 < curr_steering_val <= 92:
                    self.ctrl.execute(Advance(speed=self.forward_spd))
                elif 92 < curr_steering_val <= 95:
                    self.ctrl.execute(TurnRight(speed=self.forward_spd, degree=0.1))
                elif 95 < curr_steering_val <= 100:
                    self.ctrl.execute(TurnRight(speed=self.forward_spd, degree=0.2))
                elif 100 < curr_steering_val <= 102:
                    self.ctrl.execute(TurnRight(speed=self.forward_spd, degree=0.3))
                elif 102 < curr_steering_val <= 104:
                    self.ctrl.execute(TurnRight(speed=self.forward_spd, degree=0.3))
                elif 104 < curr_steering_val <= 108:
                    self.ctrl.execute(TurnRight(speed=self.forward_spd, degree=0.3))
                elif 108 < curr_steering_val <= 112:
                    self.ctrl.execute(TurnRight(speed=self.forward_spd, degree=0.4))
                elif 112 < curr_steering_val <= 115:
                    self.ctrl.execute(TurnRight(speed=self.forward_spd, degree=0.5))
                elif 115 < curr_steering_val <= 120:
                    self.ctrl.execute(TurnRight(speed=self.forward_spd, degree=0.6))
                elif 120 < curr_steering_val <= 135:
                    self.ctrl.execute(TurnRight(speed=self.forward_spd, degree=0.7))


                log.info(f'infer cost {time.time() - start}')
        except KeyboardInterrupt:
            self.ctrl.execute(Stop())
