#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from abc import ABC, abstractmethod

from src.utils.constant import (
    MOTOR_RATING_CH0_REAR_RIGHT,
    MOTOR_RATING_CH1_FRONT_RIGHT,
    MOTOR_RATING_CH2_FRONT_LEFT,
    MOTOR_RATING_CH3_REAR_LEFT,
)


class BaseAction(ABC):
    """
    基础动作的基类，所有基本动作均继承于该类
    """

    def __init__(self, *args, **kwds) -> None:
        """
        基础动作类的初始化方法，通过args与kwds控制输入参数
        :param args:
        :param kwds:
        """
        # 抽象的速度信息
        self.speed = kwds.get('speed', -1)
        # 电机角度
        self.servo_angle = kwds.get('servo', [-1, -1])

        # 按实际控制通道顺序设置电机比例：0=右后，1=右前，2=左前，3=左后
        self.motor_rating = [
            MOTOR_RATING_CH0_REAR_RIGHT,
            MOTOR_RATING_CH1_FRONT_RIGHT,
            MOTOR_RATING_CH2_FRONT_LEFT,
            MOTOR_RATING_CH3_REAR_LEFT,
        ]

        # 确定是否需要在运行时根据前动作更新电机角度及电机速度
        self.update_speed = False
        self.update_servo = False

        if self.speed == -1:
            self.update_speed = True

        if self.servo_angle[0] == -1 and self.servo_angle[1] == -1:
            self.update_servo = True

        # 由速度生成方法将抽象的总体速度计算为4个电机的速度并输出为list
        self.speed_setting = self.generate_speed_setting(self.speed)
        self.fix_speed()

    def fix_speed(self):
        self.speed_setting = [int(speed * ratio) for speed, ratio in zip(self.speed_setting, self.motor_rating)]

    @staticmethod
    @abstractmethod
    def generate_speed_setting(speed, degree=0):
        """
        生成4个电机的速度，并输出为列表
        抽象类，需要根据具体情况进行设置
        :param speed: 抽象的速度。 当前动作初始化时设置 或 控制器根据前一动作速度进行设置
        :param degree: 如需转弯，速度计算需要的角度信息
        :return:
        """
        pass

    def __call__(self, speed, servo_angle):
        """
        call魔法函数，两个输入参数由控制器输入
        当init方法设置了相关信息，则忽略控制器输入的参数
        当init方法没有设置相关信息，相关信息的将由控制器输入的参数进行更新

        :param speed: 抽象速度
        :param servo_angle: 舵机的角度
        :return: 长度为6的列表，前4位为4个电机的速度，后2位为舵机的两个角度
        """
        if self.update_servo:
            self.servo_angle = servo_angle
        if self.update_speed:
            degree = 0
            if hasattr(self, 'degree'):
                degree = self.degree
            self.speed_setting = self.generate_speed_setting(speed, degree)
            self.fix_speed()

        return self.speed_setting + self.servo_angle


class Advance(BaseAction):
    """
    小车前进
    """

    @staticmethod
    def generate_speed_setting(speed, degree=0):
        return [-speed, -speed, speed, speed]


class BackUp(BaseAction):
    """
    小车后退
    """

    @staticmethod
    def generate_speed_setting(speed, degree=0):
        return [speed, speed, -speed, -speed]


class CustomAction(BaseAction):
    """
    自定义动作
    """

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.speed_setting = kwds.get('motor_setting', [0, 0, 0, 0])
        self.update_controller_speed = False
        self.update_speed = False

    @staticmethod
    def generate_speed_setting(speed, degree=0):
        return [0, 0, 0, 0]


class Stop(BaseAction):
    """
    小车停止
    """

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.speed = 0

    @staticmethod
    def generate_speed_setting(speed, degree=0):
        return [0, 0, 0, 0]


class TurnLeft(BaseAction):
    """
    小车左转（差速转向：外侧轮加速，内侧轮减速）
    """
    
    @staticmethod
    def generate_speed_setting(speed, degree=0):
        # degree: 0-1之间，转向强度
        # 左转时，外侧（右）轮加速，内侧（左）轮减速
        # 保持平均速度不变，两轮都向前转
        outer_speed = int(speed * (1 + degree * 0.7))  # 外侧轮加速最多50%
        inner_speed = int(speed * (1 - degree * 0.5))  # 内侧轮减速最多50%
        # 确保内侧轮速度不为0
        inner_speed = max(1, inner_speed)
        
        # 右侧轮（外侧）加速前进，左侧轮（内侧）减速前进
        return [-outer_speed, -outer_speed, inner_speed, inner_speed]


class TurnRight(BaseAction):
    """
    小车右转（差速转向：外侧轮加速，内侧轮减速）
    """
    
    @staticmethod
    def generate_speed_setting(speed, degree=0):
        # degree: 0-1之间，转向强度
        # 右转时，外侧（左）轮加速，内侧（右）轮减速
        outer_speed = int(speed * (1 + degree * 0.7))  # 外侧轮加速最多50%
        inner_speed = int(speed * (1 - degree * 0.5))  # 内侧轮减速最多50%
        # 确保内侧轮速度不为0
        inner_speed = max(1, inner_speed)
        
        # 右侧轮（内侧）减速前进，左侧轮（外侧）加速前进
        return [-inner_speed, -inner_speed, outer_speed, outer_speed]
        
class ShiftLeft(BaseAction):
    """
    向左平移
    """

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.motor_rating = [1.5, 1.3, 1.25, 1.25]
        self.fix_speed()

    @staticmethod
    def generate_speed_setting(speed, degree=0):
        return [speed, -speed, -speed, speed]


class ShiftRight(BaseAction):
    """
    向右平移
    """

    @staticmethod
    def generate_speed_setting(speed, degree=0):
        return [-speed, speed, speed, -speed]


class LeftOblique(BaseAction):
    """
    斜向左前方
    """

    @staticmethod
    def generate_speed_setting(speed, degree=0):
        return [0, -speed, 0, speed]


class RightOblique(BaseAction):
    """
    斜向右前方
    """

    @staticmethod
    def generate_speed_setting(speed, degree=0):
        return [-speed, 0, speed, 0]


class SpinClockwise(BaseAction):
    """
    顺时针旋转
    """

    @staticmethod
    def generate_speed_setting(speed, degree=0):
        return [-speed] * 4


class SpinAntiClockwise(BaseAction):
    """
    逆时针旋转
    """

    @staticmethod
    def generate_speed_setting(speed, degree=0):
        return [speed] * 4


class SetServo(BaseAction):
    """
    舵机转动
    """

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.speed = 0

    @staticmethod
    def generate_speed_setting(speed, degree=0):
        return [0, 0, 0, 0]


class Sleep(BaseAction):
    """
    Sleep(1)等同于time.sleep(1)
    可加入至动作序列进行使用
    """

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.sleep_time = args[0]

    @staticmethod
    def generate_speed_setting(speed, degree=0):
        return []

    def __call__(self, speed, servo_angle):
        time.sleep(self.sleep_time)
        return None
