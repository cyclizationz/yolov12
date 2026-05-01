#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/home/tiehangz/proj/yolov12")
VIDEO_DIR = REPO_ROOT / "video"
RECORD_DIR = REPO_ROOT / "record" / "experiments"
RESPAWN2026_DIR = REPO_ROOT / "record" / "RESPAWN2026"
DATASETS_ROOT = Path("/home/tiehangz/proj/datasets")
DEPLOYMENT_BUILD_DIR = REPO_ROOT / "deployment" / "build"
OFFLINE_BIN = DEPLOYMENT_BUILD_DIR / "Yolov12Deployment"
CURRENT_BASELINE_BIN = DEPLOYMENT_BUILD_DIR / "Yolov12DeploymentCurrentBaseline"
DUO_BIN = DEPLOYMENT_BUILD_DIR / "Yolov12DeploymentDuo"
DEFAULT_MODEL = REPO_ROOT / "deployment" / "yolov12n-seg.onnx"
DEFAULT_THREADS = 1
EXPERIMENT_VENV_PYTHON = REPO_ROOT / ".venv-experiments" / "bin" / "python"


@dataclass(frozen=True)
class SourceVideo:
    game: str
    tag: str
    path: Path
    family: str
    notes: str = ""


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    game: str
    family: str
    source_tag: str
    source_path: str
    clip_path: str
    normalized_path: str
    start_s: float
    duration_s: float
    source_width: int
    source_height: int
    source_fps: float
    target_width: int
    target_height: int
    target_fps: float
    eligible_duo: bool
    eligible_rd: bool
    notes: str = ""


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def python_bin() -> str:
    return str(EXPERIMENT_VENV_PYTHON) if EXPERIMENT_VENV_PYTHON.exists() else "python3"


def capture_json(cmd: list[str]) -> dict[str, Any]:
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def ffprobe_stream_info(path: Path) -> dict[str, Any]:
    doc = capture_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,duration,nb_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    stream = ((doc.get("streams") or [{}])[0]) if doc.get("streams") else {}
    return {
        "width": int(stream.get("width", 0) or 0),
        "height": int(stream.get("height", 0) or 0),
        "fps": _parse_fps(str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")),
        "duration_s": _parse_float(stream.get("duration", None), default=0.0)
        or _parse_float((doc.get("format") or {}).get("duration", None), default=0.0),
        "nb_frames": int(_parse_float(stream.get("nb_frames", None), default=0.0)),
    }


def _parse_float(value: Any, *, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _parse_fps(value: str) -> float:
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(value)
    except Exception:
        return 0.0


def discover_source_videos() -> dict[str, list[SourceVideo]]:
    candidates: dict[str, list[tuple[str, Path, str]]] = {
        "fc5": [
            ("fc5_crop_repo", VIDEO_DIR / "fc5_crop.mkv", "Repo-local FC5 cropped/zoomed-in clip."),
            ("fc5_roi_gunheavy", DATASETS_ROOT / "fps" / "roi_720p_gunheavy_full.mp4", "Gun-heavy zoomed-in ROI source."),
            ("fc5_merged_crop", DATASETS_ROOT / "fps" / "merged_crop.mkv", "Dataset crop fallback."),
        ],
        "fm6": [
            ("fm6_repo", VIDEO_DIR / "fm6.mkv", "Repo-local FM6 evaluation clip."),
            ("fm6_1080p_dataset", DATASETS_ROOT / "racing" / "fm6_1080p.mkv", "Dataset 1080p source."),
            ("fm6_dataset", DATASETS_ROOT / "racing" / "fm6.mkv", "Dataset fallback source."),
        ],
        "mario": [
            ("mario_repo", VIDEO_DIR / "pixel.mkv", "Repo-local Mario evaluation clip."),
            ("mario_720_dataset", DATASETS_ROOT / "pixel" / "supermario_720.mkv", "Dataset 720p source."),
            ("mario_clean_dataset", DATASETS_ROOT / "pixel" / "supermario_clean.mkv", "Clean fallback."),
            ("mario_pix_test", DATASETS_ROOT / "pixel" / "pix_test.mkv", "Short synthetic pixel test."),
            ("mario_pix_test_multi", DATASETS_ROOT / "pixel" / "pix_test_multi.mkv", "Multi-instance pixel test."),
        ],
    }
    out: dict[str, list[SourceVideo]] = {}
    for game, game_candidates in candidates.items():
        rows: list[SourceVideo] = []
        for tag, path, notes in game_candidates:
            if path.exists():
                rows.append(SourceVideo(game=game, tag=tag, path=path, family=_family_for_game(game), notes=notes))
        out[game] = rows
    return out


def _family_for_game(game: str) -> str:
    if game == "mario":
        return "pixel"
    return "learned"


def plan_clips_for_sources(
    sources: dict[str, list[SourceVideo]],
    *,
    target_count_per_game: int = 5,
    clip_duration_s: float = 100.0,
    min_duration_s: float = 30.0,
    target_width: int = 1920,
    target_height: int = 1080,
    target_fps: float = 60.0,
    manifest_root: Path | None = None,
) -> list[ClipSpec]:
    clips: list[ClipSpec] = []
    manifest_root = ensure_dir(manifest_root or (RECORD_DIR / "manifest"))
    for game, vids in sources.items():
        planned = _plan_game_clips(
            game=game,
            vids=vids,
            target_count=target_count_per_game,
            clip_duration_s=clip_duration_s,
            min_duration_s=min_duration_s,
            target_width=target_width,
            target_height=target_height,
            target_fps=target_fps,
            manifest_root=manifest_root,
        )
        clips.extend(planned)
    return clips


def _plan_game_clips(
    *,
    game: str,
    vids: list[SourceVideo],
    target_count: int,
    clip_duration_s: float,
    min_duration_s: float,
    target_width: int,
    target_height: int,
    target_fps: float,
    manifest_root: Path,
) -> list[ClipSpec]:
    clips: list[ClipSpec] = []
    normalized_root = ensure_dir(manifest_root / "normalized")
    actual_duration = clip_duration_s
    if not vids:
        return clips

    clip_idx = 0
    for src in vids:
        info = ffprobe_stream_info(src.path)
        dur = float(info["duration_s"])
        if dur <= 0.0:
            continue
        windows = _slice_windows(dur, actual_duration, min_duration_s)
        for start_s, dur_s, tag in windows:
            clip_id = f"{game}_{clip_idx:02d}"
            rel_norm = normalized_root / game / f"{clip_id}.mp4"
            clip_target_width = int(info["width"]) if game == "mario" else target_width
            clip_target_height = int(info["height"]) if game == "mario" else target_height
            clip_target_fps = float(info["fps"]) if game == "mario" else target_fps
            clips.append(
                ClipSpec(
                    clip_id=clip_id,
                    game=game,
                    family=src.family,
                    source_tag=src.tag,
                    source_path=str(src.path),
                    clip_path=str(src.path),
                    normalized_path=str(rel_norm),
                    start_s=start_s,
                    duration_s=dur_s,
                    source_width=int(info["width"]),
                    source_height=int(info["height"]),
                    source_fps=float(info["fps"]),
                    target_width=clip_target_width,
                    target_height=clip_target_height,
                    target_fps=clip_target_fps,
                    eligible_duo=True,
                    eligible_rd=True,
                    notes=tag or src.notes,
                )
            )
            clip_idx += 1
            if clip_idx >= target_count:
                return clips
    return clips


def _slice_windows(duration_s: float, target_duration_s: float, min_duration_s: float) -> list[tuple[float, float, str]]:
    if duration_s <= min_duration_s:
        return [(0.0, duration_s, "short_source_full_span")]
    if duration_s <= target_duration_s:
        return [(0.0, duration_s, "short_source_full_span")]
    count = max(1, int(duration_s // target_duration_s))
    if count == 1:
        return [(max(0.0, (duration_s - target_duration_s) * 0.5), target_duration_s, "single_center_window")]
    gap = max(0.0, (duration_s - count * target_duration_s) / max(1, count + 1))
    windows: list[tuple[float, float, str]] = []
    cursor = gap
    for idx in range(count):
        start = min(max(0.0, cursor), max(0.0, duration_s - target_duration_s))
        windows.append((start, target_duration_s, f"auto_window_{idx}"))
        cursor += target_duration_s + gap
    return windows


def normalize_clip(spec: ClipSpec, *, force: bool = False) -> Path:
    out_path = Path(spec.normalized_path)
    if out_path.exists() and not force:
        return out_path
    ensure_dir(out_path.parent)
    vf = f"scale={spec.target_width}:{spec.target_height}:flags=lanczos,fps={int(spec.target_fps)}"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{spec.start_s:.3f}",
        "-t",
        f"{spec.duration_s:.3f}",
        "-i",
        spec.clip_path,
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "12",
        str(out_path),
    ]
    run(cmd)
    return out_path


def write_manifest(clips: list[ClipSpec], out_json: Path, out_csv: Path) -> None:
    ensure_dir(out_json.parent)
    payload = [asdict(c) for c in clips]
    out_json.write_text(json.dumps(payload, indent=2))
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload[0].keys()) if payload else ["clip_id"])
        writer.writeheader()
        for row in payload:
            writer.writerow(row)


def load_manifest(path: Path) -> list[ClipSpec]:
    doc = json.loads(path.read_text())
    return [ClipSpec(**row) for row in doc]


def bitrate_label_mbps(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}mbps"
    return f"{value:.2f}mbps".replace(".", "p")


def classify_duo_viability(mean_run_length: float) -> str:
    if mean_run_length <= 2.0:
        return "doomed"
    if mean_run_length < 4.0:
        return "borderline"
    return "promising"


def moving_average(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def infer_template_bytes(dict_dir: Path) -> int:
    total = 0
    if not dict_dir.exists():
        return total
    for path in dict_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".webp", ".jpg", ".jpeg"}:
            total += path.stat().st_size
    return total


def seconds_to_frames(seconds: float, fps: float) -> int:
    return int(math.ceil(max(0.0, seconds) * max(1e-9, fps)))
