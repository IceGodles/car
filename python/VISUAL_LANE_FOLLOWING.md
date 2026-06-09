# 传统视觉黄线循迹实现说明

当前实现新增了一套不依赖 LF 模型的循迹方式：

1. 对摄像头画面做 HSV 黄色分割，只保留下半部分 ROI。
2. 在多个横向扫描线上寻找黄色线段中心点。
3. 根据几何规则选择当前应跟随的车道中心。
4. 用 PID 将车道中心偏差转换成转向修正量。
5. 通过现有 `Advance`、`TurnLeft`、`TurnRight` 动作下发四轮差速控制。

## 新增文件

- `src/utils/visual_lane.py`
  - `YellowLaneFollower`：黄线分割、候选线提取、几何选线、PID 输出。
  - `PIDController`：简单 PID 控制器。
- `src/scenes/visual_lane_following.py`
  - `VLF` 场景：读取实时摄像头共享内存画面，并执行传统视觉循迹。
- `test_visual_lane.py`
  - 离线图片、图片目录、摄像头测试工具，用来调试 HSV、ROI、车道宽度和 PID 参数。

## 几何选线逻辑

### 同时看到多条黄线

每条扫描线会得到多个黄色线段中心点。若至少有两条线，会优先寻找宽度合理的一对线，并把两条线的中点作为当前车道中心。

当画面里出现内圈线、外圈线、其他车道线时，程序会优先选择“车道中心最接近上一帧目标点”的一对线，避免目标在多条线之间跳变。

### 只看到一条黄线

如果当前扫描线只看到一条黄线，程序会根据历史判断这条线更像左边界还是右边界，然后用预设车道宽度估算车道中心：

- 看到左边界：目标中心 = 线位置 + 半个车道宽度
- 看到右边界：目标中心 = 线位置 - 半个车道宽度

这能处理接近路口、弯道、画面边缘只剩一条黄线的情况。

### 暂时丢线

如果短时间没有检测到黄线，会继续沿用上一帧目标点，最多保留 `max_lost_frames` 帧。连续丢线后执行停车。

## 离线测试

在小车 Python 目录运行：

```bash
cd /home/HwHiAiUser/E2ESamples/src/E2E-Sample/Car/python
python3 test_visual_lane.py --image lanetest1.jpg --show
```

批量测试图片目录：

```bash
python3 test_visual_lane.py --image-dir capture/lane_samples
```

摄像头实时测试：

```bash
python3 test_visual_lane.py --camera 0 --show
```

测试输出保存到：

```text
capture/visual_lane/
```

其中：

- `*_debug.jpg`：原图上画出扫描线、候选黄线点、目标中心点。
- `*_mask.jpg`：黄色分割结果。

## 上车运行

启动主程序的命令模式：

```bash
python3 main.py --mode cmd
```

看到 `cmd` 输入后，输入：

```text
VLF
```

程序会启动传统视觉循迹场景。运行中每秒保存一张调试图：

```text
capture/vlf/latest.jpg
```

## 常用调参位置

主要参数在 `src/utils/visual_lane.py` 的 `YellowLaneFollower.__init__` 中：

```python
self.lower_yellow = np.array(config.get("lower_yellow", [12, 70, 80]), dtype=np.uint8)
self.upper_yellow = np.array(config.get("upper_yellow", [45, 255, 255]), dtype=np.uint8)
self.roi_top_ratio = config.get("roi_top_ratio", 0.42)
self.scan_ratios = config.get("scan_ratios", [0.62, 0.70, 0.78])
self.lane_width_ratio = config.get("lane_width_ratio", 0.28)
self.center_deadband_px = config.get("center_deadband_px", 28)
```

PID 参数：

```python
kp=0.0028
ki=0.0
kd=0.0012
output_limit=0.8
```

运行速度在 `src/scenes/visual_lane_following.py`：

```python
self.forward_spd = 22
```

## 推荐调参顺序

1. 先用 `test_visual_lane.py --image` 看 `mask`，调 HSV，让黄色线稳定变白，蓝灰色路面尽量为黑。
2. 调 `roi_top_ratio`，只看小车前方道路，不看远处门、墙、反光。
3. 调 `lane_width_ratio`，让单线估计出的红色目标点落在当前车道中心附近。
4. 上车低速运行，先调 `forward_spd`。
5. 如果修正太慢，略增 `kp`；如果左右抖动，略增 `center_deadband_px` 或降低 `kp`。
6. 如果转向有来回振荡，略增 `kd`；如果噪声放大，降低 `kd`。

## 适用边界

这套方法比 LF 模型更可解释，适合你的多黄线、单黄线、地图颜色固定的场景。但它依赖黄色分割质量，强反光、阴影、黄线破损严重时仍需要通过 HSV、ROI、形态学参数继续调。
