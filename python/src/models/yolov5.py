import numpy as np
import torch

from ais_bench.infer.interface import InferSession
from src.utils.cv_utils import nms, scale_coords, preprocess_image_yolov5


class YoloV5:
    def __init__(self, model_path):
        self.model = InferSession(0, model_path)
        self.last_debug_info = {}
        self.input_specs = self._read_input_specs()
        self.input_dtype = self._dtype_from_spec(0, np.float32)
        self.img_info_dtype = self._dtype_from_spec(1, self.input_dtype)
        self.input_count = len(self.input_specs) if self.input_specs else 2
        self.neth = 640
        self.netw = 640
        self.conf_threshold = 0.1
        dic = {0: 'left',
               1: 'stop',
               2: 'right',
               3: 'turnaround'}
        self.names = list(dic.values())
        self.object_list = list(dic.values())
        self.cfg = {
            'conf_thres': 0.6,  # 模型置信度阈值，阈值越低，得到的预测框越多
            'iou_thres': 0.5,  # IOU阈值，高于这个阈值的重叠预测框会被过滤掉
            'input_shape': [640, 640],  # 模型输入尺寸
        }
        self.cfg['input_dtype'] = self.input_dtype

    def _read_input_specs(self):
        getter = getattr(self.model, "get_inputs", None)
        if getter is None:
            return []
        try:
            return list(getter())
        except Exception:
            return []

    def _dtype_from_spec(self, index, default):
        if index >= len(self.input_specs):
            return default
        datatype = str(getattr(self.input_specs[index], "datatype", "")).lower()
        if "float16" in datatype or "fp16" in datatype or datatype == "1":
            return np.float16
        if "float32" in datatype or "fp32" in datatype or datatype == "0" or datatype == "float":
            return np.float32
        return default

    def _xywh2xyxy(self, boxes):
        converted = np.copy(boxes)
        converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return converted

    def _box_iou(self, box, boxes):
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
        boxes_area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
        return inter / (box_area + boxes_area - inter + 1e-6)

    def _numpy_nms(self, detections):
        if detections.size == 0:
            return detections

        kept = []
        for class_id in np.unique(detections[:, 5]).astype(np.int32):
            class_dets = detections[detections[:, 5] == class_id]
            order = np.argsort(-class_dets[:, 4])
            while order.size > 0:
                best = order[0]
                kept.append(class_dets[best])
                if order.size == 1:
                    break
                ious = self._box_iou(class_dets[best, :4], class_dets[order[1:], :4])
                order = order[1:][ious <= self.cfg["iou_thres"]]

        return np.asarray(kept, dtype=np.float32)

    def _decode_flat_output(self, output, img_bgr_shape, scale_ratio, pad_size):
        output = np.asarray(output)
        if output.ndim == 1:
            flat = output.reshape(1, -1)
        elif output.ndim == 2:
            flat = output
        else:
            flat = output.reshape(output.shape[0], -1)

        attrs = None
        for candidate in (5 + len(self.names), 24):
            if flat.shape[1] % candidate == 0:
                attrs = candidate
                break
        if attrs is None:
            raise ValueError(f"Unsupported YOLO output shape: {output.shape}")

        pred = flat.reshape(1, -1, attrs)[0].astype(np.float32)
        box_xywh = pred[:, :4]
        obj_conf = pred[:, 4]
        cls_scores = pred[:, 5:5 + len(self.names)]
        cls_conf = cls_scores.max(axis=1)
        cls_id = cls_scores.argmax(axis=1)
        scores = obj_conf * cls_conf
        top_obj_idx = np.argsort(-obj_conf)[:5]
        self.last_debug_info.update({
            "output_shape": tuple(output.shape),
            "attrs": int(attrs),
            "anchors": int(pred.shape[0]),
            "max_obj": float(obj_conf.max()) if obj_conf.size else 0.0,
            "max_cls": float(cls_conf.max()) if cls_conf.size else 0.0,
            "max_score": float(scores.max()) if scores.size else 0.0,
            "top_obj": [
                {
                    "idx": int(idx),
                    "obj": float(obj_conf[idx]),
                    "cls": self.names[int(cls_id[idx])] if int(cls_id[idx]) < len(self.names) else str(int(cls_id[idx])),
                    "cls_conf": float(cls_conf[idx]),
                    "score": float(scores[idx]),
                    "xywh": [float(v) for v in box_xywh[idx]],
                }
                for idx in top_obj_idx
            ],
        })

        keep = scores > self.cfg["conf_thres"]
        if not keep.any():
            scores = obj_conf
            keep = scores > self.cfg["conf_thres"]
        if not keep.any():
            return []

        boxes_xyxy = self._xywh2xyxy(box_xywh[keep])
        detections = np.column_stack([boxes_xyxy, scores[keep], cls_id[keep]]).astype(np.float32)
        detections = self._numpy_nms(detections)
        if detections.size == 0:
            return []

        scale_coords(self.cfg['input_shape'], detections[:, :4], img_bgr_shape, ratio_pad=(scale_ratio, pad_size))
        pred_boxes = []
        for det in detections:
            class_id = int(det[5])
            if class_id >= len(self.names):
                continue
            x1, y1, x2, y2 = det[:4].astype(np.int32)
            pred_boxes.append([int(x1), int(y1), int(x2), int(y2), self.names[class_id], float(det[4])])
        return pred_boxes

    def infer(self, img_bgr):
        img, scale_ratio, pad_size = preprocess_image_yolov5(img_bgr, self.cfg)
        orig_h, orig_w = img_bgr.shape[:2]
        img_info = np.array([[orig_h, orig_w, 1.0, 0.0]], dtype=self.img_info_dtype)
        # 模型推理
        inputs = [img]
        if self.input_count >= 2:
            inputs.append(img_info)
        outputs = self.model.infer(inputs)
        self.last_debug_info = {
            "input_image_shape": tuple(img.shape),
            "input_image_dtype": str(img.dtype),
            "input_image_min": float(img.min()),
            "input_image_max": float(img.max()),
            "input_image_mean": float(img.mean()),
            "img_info": img_info.astype(np.float32).tolist(),
            "outputs": [
                {
                    "idx": idx,
                    "shape": tuple(out.shape),
                    "dtype": str(out.dtype),
                    "min": float(np.min(out)) if out.size else 0.0,
                    "max": float(np.max(out)) if out.size else 0.0,
                    "mean": float(np.mean(out)) if out.size else 0.0,
                    "nonzero": int(np.count_nonzero(out)),
                }
                for idx, out in enumerate(outputs)
            ],
        }
        output = outputs[0]

        if len(output.shape) < 3 or output.shape[-1] != 5 + len(self.names):
            return self._decode_flat_output(output, img_bgr.shape, scale_ratio, pad_size)

        output = torch.tensor(output)
        # 非极大值抑制后处理
        boxout = nms(output, conf_thres=self.cfg["conf_thres"], iou_thres=self.cfg["iou_thres"])
        pred_all = boxout[0].numpy()
        # 预测坐标转换
        scale_coords(self.cfg['input_shape'], pred_all[:, :4], img_bgr.shape, ratio_pad=(scale_ratio, pad_size))
        pred_boxes = []

        for idx, class_id in enumerate(pred_all[:, 5]):
            if float(pred_all[idx][4] < float(0.05)):
                continue
            obj_name = self.names[int(pred_all[idx][5])]
            confidence = pred_all[idx][4]
            x1 = int(pred_all[idx][0])
            y1 = int(pred_all[idx][1])
            x2 = int(pred_all[idx][2])
            y2 = int(pred_all[idx][3])

            pred_boxes.append([x1, y1, x2, y2, obj_name, confidence])

        return pred_boxes
