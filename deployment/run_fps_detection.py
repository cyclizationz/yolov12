#!/usr/bin/env python3
"""
Run YOLOv12-seg (FC5 gun) detection on a video. Record all bboxes and report
10s windows where gun appears most. Detection only (no mask decoding).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# FC5: single class 0 = gun
CLASS_NAMES = {0: "gun"}


class YOLOv12SegDetect:
    """YOLOv12 segmentation model, detection-only (bbox) inference."""

    def __init__(self, onnx_model: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.session = ort.InferenceSession(
            onnx_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"],
        )
        self.ndtype = (
            np.float16
            if "float16" in str(self.session.get_inputs()[0].type)
            else np.float32
        )
        shp = self.session.get_inputs()[0].shape
        self.model_height, self.model_width = int(shp[-2]), int(shp[-1])
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.nm = 32

    def preprocess(self, img: np.ndarray):
        shape = img.shape[:2]
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = (r, r)
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w = (new_shape[1] - new_unpad[0]) / 2
        pad_h = (new_shape[0] - new_unpad[1]) / 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top = int(round(pad_h - 0.1))
        bottom = int(round(pad_h + 0.1))
        left = int(round(pad_w - 0.1))
        right = int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)

    def postprocess_detect_only(self, preds, im0_shape, ratio, pad_w, pad_h):
        x = preds[0]
        x = np.einsum("bcn->bnc", x)
        nm = self.nm
        x = x[np.amax(x[..., 4:-nm], axis=-1) > self.conf_threshold]
        if x.size == 0:
            return []
        x = np.c_[
            x[..., :4],
            np.amax(x[..., 4:-nm], axis=-1),
            np.argmax(x[..., 4:-nm], axis=-1),
        ]
        # NMS expects [x, y, w, h] (top-left); we have [cx, cy, w, h]
        xywh = x[:, :4].copy()
        xywh[:, 0] -= xywh[:, 2] / 2
        xywh[:, 1] -= xywh[:, 3] / 2
        indices = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            x[:, 4].tolist(),
            self.conf_threshold,
            self.iou_threshold,
        )
        if len(indices) == 0:
            return []
        x = x[indices.flatten()]
        # cxcywh -> xyxy
        x[..., [0, 1]] -= x[..., [2, 3]] / 2
        x[..., [2, 3]] += x[..., [0, 1]]
        x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
        x[..., :4] /= min(ratio)
        x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0_shape[1])
        x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0_shape[0])
        return x[..., :6]  # x1, y1, x2, y2, conf, cls

    def __call__(self, frame: np.ndarray):
        im, ratio, (pad_w, pad_h) = self.preprocess(frame)
        preds = self.session.run(None, {self.session.get_inputs()[0].name: im})
        boxes = self.postprocess_detect_only(
            preds, frame.shape, ratio, pad_w, pad_h
        )
        return boxes


def main():
    parser = argparse.ArgumentParser(description="Run FC5 gun detection on video, record bboxes, report 10s peaks.")
    parser.add_argument("--model", type=str, default=None, help="Path to ONNX model")
    parser.add_argument("--video", type=str, default="/home/tiehangz/proj/datasets/fps/merged.mkv", help="Input video")
    parser.add_argument("--out", type=str, default=None, help="Output JSON/CSV path for all bboxes (default: next to video)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--window-sec", type=float, default=10.0, help="Aggregation window in seconds")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    model_path = args.model
    if model_path is None:
        for name in ["yolov12n_fc5_seg_v1_e300_p0.onnx", "models/yolov12n_fc5_seg_v1_e300_p0_best.onnx"]:
            p = root / name
            if p.exists():
                model_path = str(p)
                break
        if model_path is None:
            model_path = str(root / "models/yolov12n_fc5_seg_v1_e300_p0_best.onnx")
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    out_path = args.out
    if out_path is None:
        out_path = video_path.parent / (video_path.stem + "_detections.json")
    out_path = Path(out_path)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    print(f"Model: {model_path}")
    print(f"Video: {video_path} (fps={fps:.2f}, frames={n_frames})")
    print(f"Output: {out_path}")

    model = YOLOv12SegDetect(model_path, conf_threshold=args.conf, iou_threshold=args.iou)
    cap = cv2.VideoCapture(str(video_path))

    all_detections = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_sec = frame_idx / fps
        boxes = model(frame)
        for b in boxes:
            x1, y1, x2, y2, conf, cls_id = b
            cls_id = int(cls_id)
            all_detections.append({
                "frame": frame_idx,
                "time_sec": round(t_sec, 4),
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, f"cls{cls_id}"),
                "confidence": round(float(conf), 4),
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "x2": round(float(x2), 2),
                "y2": round(float(y2), 2),
            })
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  frame {frame_idx} ...")

    cap.release()
    total_frames = frame_idx
    duration_sec = total_frames / fps

    with open(out_path, "w") as f:
        json.dump({"fps": fps, "total_frames": total_frames, "duration_sec": round(duration_sec, 2), "detections": all_detections}, f, indent=2)

    print(f"Recorded {len(all_detections)} detections -> {out_path}")

    # Aggregate by 10s windows (gun = class 0)
    window_sec = args.window_sec
    gun_detections = [d for d in all_detections if d["class_id"] == 0]
    bucket_counts = {}
    for d in gun_detections:
        t = d["time_sec"]
        bucket = int(t // window_sec) * window_sec
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    if not bucket_counts:
        print("\nNo gun detections in the video.")
        return

    sorted_buckets = sorted(bucket_counts.items(), key=lambda x: -x[1])
    max_count = sorted_buckets[0][1]
    top_buckets = [(b, c) for b, c in sorted_buckets if c == max_count]

    print("\n--- Gun appearance by 10s window ---")
    for bucket_start, count in sorted_buckets[:15]:
        bar = "#" * min(50, count) + " " * (50 - min(50, count))
        print(f"  {bucket_start:6.1f}s - {bucket_start + window_sec:.1f}s: {count:5d}  {bar}")

    print("\n--- Where the gun appears the most (accurate to 10s) ---")
    for bucket_start, count in top_buckets:
        end = bucket_start + window_sec
        print(f"  {bucket_start:.1f}s - {end:.1f}s  ({count} gun detections)")

    return


if __name__ == "__main__":
    main()
