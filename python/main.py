#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from argparse import ArgumentParser
from multiprocessing import Process, Queue, shared_memory

import cv2
import numpy as np

from src.scenes import Manual, scene_initiator
from src.utils import getkey, log, CameraBroadcaster, CAMERA_INFO, Controller
from src.actions import Stop


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, required=False, default='manual',
                        choices=['cmd', 'voice', 'manual', 'easy', 'smart'])
    parser.add_argument('--show', action='store_true',
                        help='Show realtime camera preview in an OpenCV window.')

    return parser.parse_args()


def camera_preview(memory_name, camera_info):
    height = camera_info.get('height', 720)
    width = camera_info.get('width', 1280)
    fps = camera_info.get('fps', 30)
    delay = max(1, int(1000 / max(fps, 1)))
    shm = shared_memory.SharedMemory(name=memory_name)
    frame = np.ndarray((height, width, 3), dtype=np.uint8, buffer=shm.buf)

    log.info('Camera preview start. Press q in preview window to close it.')
    try:
        while True:
            img = frame.copy()
            cv2.imshow('car camera preview', img)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
    except cv2.error as err:
        log.error(f'Camera preview failed: {err}')
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cv2.destroyAllWindows()
        shm.close()
        log.info('Camera preview closing')


def stop_process(process):
    if process is not None and process.is_alive():
        process.kill()
        process.join()


if __name__ == '__main__':
    args = parse_args()
    log.info('start')
    ctrl = Controller()
    msg_queue = Queue(maxsize=1)
    camera = CameraBroadcaster(CAMERA_INFO)
    shared_memory_name = camera.memory_name
    camera_process = Process(target=camera.run)
    camera_process.start()
    preview_process = None
    if args.show:
        preview_process = Process(target=camera_preview, args=(shared_memory_name, CAMERA_INFO))
        preview_process.start()
    if args.mode == 'manual':
        task = Manual(shared_memory_name, CAMERA_INFO, msg_queue)
        process = Process(target=task.loop)
        process.start()
        try:
            while True:
                key = getkey()
                if key == 'esc':
                    process.join()
                    camera.stop_sign.value = True
                    camera_process.join()
                    stop_process(preview_process)
                    break
                else:
                    msg_queue.put(key)
        except (KeyboardInterrupt, SystemExit):
            camera.stop_sign.value = True
            camera_process.join()
            stop_process(preview_process)
            os.system('stty sane')
            log.info('stopping.')
    elif args.mode == 'cmd':
        process_list = []
        record_map = {}
        try:
            log.info(f'start reading cmd')
            while True:
                command = input().strip()
                if command == 'stop':
                    for p in process_list:
                        p.kill()
                    log.info(f'start put stop sign')
                    ctrl.execute(Stop())
                    camera.stop_sign.value = True
                    camera_process.join()
                    stop_process(preview_process)
                    break
                elif command == 'clear':
                    for p in process_list:
                        p.kill()
                    process_list.clear()
                    ctrl.execute(Stop())
                    log.info(f'clear succ')
                    continue
                elif command == 'Manual':
                    log.error(f'Does not support switching from cmd mode to manual mode')
                    continue
                log.info(f'building scene {command}')
                scene = scene_initiator(command)
                log.info(f'{scene}')
                if scene is not None:
                    scene_obj = scene(shared_memory_name, CAMERA_INFO, msg_queue)
                    process = Process(target=scene_obj.loop)
                    process.start()
                    process_list.append(process)

        except (KeyboardInterrupt, SystemExit):
            camera.stop_sign.value = True
            camera_process.join()
            stop_process(preview_process)
            for process in process_list:
                process.kill()
            log.info('stopping.')

    elif args.mode == 'smart':
        process_list = []
        task = scene_initiator('SmartCruise')(shared_memory_name, CAMERA_INFO, msg_queue)
        process_list.append(Process(target=task.loop))

        for process in process_list:
            process.start()
        try:
            while True:
                key = getkey()
                if key == 'esc':
                    for process in process_list:
                        process.kill()
                    ctrl.execute(Stop())
                    camera.stop_sign.value = True
                    camera_process.join()
                    stop_process(preview_process)
                    break
                else:
                    msg_queue.put(key)
        except (KeyboardInterrupt, SystemExit):
            camera.stop_sign.value = True
            camera_process.join()
            stop_process(preview_process)
            os.system('stty sane')
            log.info('stopping.')
    elif args.mode == 'voice':
        raise NotImplementedError('voice control is not currently supported.')
    elif args.mode == 'easy':
        process_list = []
        task1 = scene_initiator('Helper')(shared_memory_name, CAMERA_INFO, msg_queue)
        process_list.append(Process(target=task1.loop))
        task2 = scene_initiator('VLF')(shared_memory_name, CAMERA_INFO, msg_queue)
        process_list.append(Process(target=task2.loop))

        for process in process_list:
            process.start()
        try:
            while True:
                key = getkey()
                if key == 'esc':
                    for process in process_list:
                        process.kill()
                    ctrl.execute(Stop())
                    camera.stop_sign.value = True
                    camera_process.join()
                    stop_process(preview_process)
                    break
                else:
                    msg_queue.put(key)
        except (KeyboardInterrupt, SystemExit):
            camera.stop_sign.value = True
            camera_process.join()
            stop_process(preview_process)
            os.system('stty sane')
            log.info('stopping.')
