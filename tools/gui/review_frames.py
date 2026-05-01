#!/usr/bin/env python3
"""
Lightweight OpenCV GUI to review per-frame masking results and optionally label object counts.

Usage:
  conda run -n yolov12 python tools/gui/review_frames.py --out-dir <run_folder>

Controls:
  - a / d: prev / next frame
  - j / k: -10 / +10 frames
  - 0-9: set gt object count to that digit (0..9)
  - + / -: increment / decrement gt count
  - s: save annotations to <out-dir>/object_gt.csv
  - q or ESC: quit (prompts save if dirty)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


def load_gt(path: Path) -> dict[int, int]:
    if not path.exists():
        return {}
    out: dict[int, int] = {}
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                fid = int(row["frame_id"])
                gt = int(row["object_present_gt"])
            except Exception:
                continue
            out[fid] = gt
    return out


def save_gt(path: Path, gt: dict[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_id", "object_present_gt"])
        w.writeheader()
        for fid in sorted(gt.keys()):
            w.writerow({"frame_id": fid, "object_present_gt": gt[fid]})


def read_frame(cap: cv2.VideoCapture, idx0: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx0)
    ok, fr = cap.read()
    return ok, fr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--scale", type=float, default=1.0, help="Display scale (e.g. 0.75)")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    base_mp4 = out_dir / "original_output.mp4"
    masked_mp4 = out_dir / "segmented_output.mp4"
    if not base_mp4.exists() or not masked_mp4.exists():
        raise SystemExit("Expected original_output.mp4 and segmented_output.mp4 in out-dir")

    # Optional per-frame signals from report.json / per_frame_metrics.csv
    model_cnt: dict[int, int] = {}
    ref_cnt: dict[int, int] = {}
    try:
        rp = out_dir / "report.json"
        if rp.exists():
            rep = json.loads(rp.read_text())
            per = rep.get("per_frame", []) or []
            for i, it in enumerate(per):
                fid = i + 1
                mc = it.get("object_present_model", None)
                rc = it.get("object_successfully_masked", None)
                if mc is not None:
                    try:
                        model_cnt[fid] = int(mc)
                    except Exception:
                        pass
                if rc is not None:
                    try:
                        ref_cnt[fid] = int(rc)
                    except Exception:
                        pass
    except Exception:
        pass

    base_bytes: dict[int, int] = {}
    masked_bytes: dict[int, int] = {}
    try:
        csvp = out_dir / "per_frame_metrics.csv"
        if csvp.exists():
            with csvp.open("r", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    try:
                        fid = int(row["frame_id"])
                        base_bytes[fid] = int(row["baseline_bytes"])
                        masked_bytes[fid] = int(row["masked_bytes"])
                    except Exception:
                        continue
    except Exception:
        pass

    cap0 = cv2.VideoCapture(str(base_mp4))
    cap1 = cv2.VideoCapture(str(masked_mp4))
    if not cap0.isOpened() or not cap1.isOpened():
        raise SystemExit("Failed to open videos")

    n0 = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n = min(n0, n1)
    if n <= 0:
        raise SystemExit("No frames")

    gt_path = out_dir / "object_gt.csv"
    gt = load_gt(gt_path)
    dirty = False

    idx = 0
    win = "review_frames (original | masked)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok0, f0 = read_frame(cap0, idx)
        ok1, f1 = read_frame(cap1, idx)
        if not ok0 or not ok1:
            break

        if args.scale != 1.0:
            f0 = cv2.resize(f0, (0, 0), fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
            f1 = cv2.resize(f1, (0, 0), fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)

        h = min(f0.shape[0], f1.shape[0])
        f0 = f0[:h, :]
        f1 = f1[:h, :]

        gap = 6
        spacer = 255 * (f0[:, :gap, :] * 0 + 1)
        canvas = cv2.hconcat([f0, spacer, f1])

        frame_id = idx + 1
        mc = model_cnt.get(frame_id, "")
        rc = ref_cnt.get(frame_id, "")
        bb = base_bytes.get(frame_id, "")
        mb = masked_bytes.get(frame_id, "")
        label = (
            f"frame_id={frame_id}/{n}  model_count={mc}  ref_count={rc}  "
            f"baseB={bb}  maskedB={mb}  gt={gt.get(frame_id, '')}  "
            f"(a/d prev/next, +/- adjust, s save, q quit)"
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (255, 255, 255), -1)
        cv2.putText(canvas, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(0) & 0xFF

        if key in (27, ord("q")):
            if dirty:
                print("Unsaved changes. Press 's' to save, or 'q' again to quit without saving.")
                key2 = cv2.waitKey(0) & 0xFF
                if key2 == ord("s"):
                    save_gt(gt_path, gt)
                    print(f"Saved {gt_path}")
                    dirty = False
                    break
                if key2 in (27, ord("q")):
                    break
            else:
                break
        elif key == ord("a"):
            idx = max(0, idx - 1)
        elif key == ord("d"):
            idx = min(n - 1, idx + 1)
        elif key == ord("j"):
            idx = max(0, idx - 10)
        elif key == ord("k"):
            idx = min(n - 1, idx + 10)
        elif key == ord("s"):
            save_gt(gt_path, gt)
            print(f"Saved {gt_path}")
            dirty = False
        elif key in (ord("+"), ord("=")):
            cur = int(gt.get(frame_id, 0) or 0)
            gt[frame_id] = cur + 1
            dirty = True
        elif key in (ord("-"), ord("_")):
            cur = int(gt.get(frame_id, 0) or 0)
            gt[frame_id] = max(0, cur - 1)
            dirty = True
        elif ord("0") <= key <= ord("9"):
            gt[frame_id] = int(chr(key))
            dirty = True

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


