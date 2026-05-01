#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import RESPAWN2026_DIR, discover_source_videos, normalize_clip, plan_clips_for_sources, write_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-build the offline experiment manifest and normalized clips.")
    ap.add_argument("--out-dir", type=Path, default=RESPAWN2026_DIR / "manifest")
    ap.add_argument("--clips-per-game", type=int, default=5)
    ap.add_argument("--clip-seconds", type=float, default=100.0)
    ap.add_argument("--min-clip-seconds", type=float, default=30.0)
    ap.add_argument("--target-width", type=int, default=1920)
    ap.add_argument("--target-height", type=int, default=1080)
    ap.add_argument("--target-fps", type=float, default=60.0)
    ap.add_argument("--skip-normalize", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sources = discover_source_videos()
    clips = plan_clips_for_sources(
        sources,
        target_count_per_game=args.clips_per_game,
        clip_duration_s=args.clip_seconds,
        min_duration_s=args.min_clip_seconds,
        target_width=args.target_width,
        target_height=args.target_height,
        target_fps=args.target_fps,
        manifest_root=args.out_dir,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for clip in clips:
        if not args.skip_normalize:
            normalize_clip(clip, force=args.force)

    manifest_json = args.out_dir / "offline_manifest.json"
    manifest_csv = args.out_dir / "offline_manifest.csv"
    write_manifest(clips, manifest_json, manifest_csv)

    summary = {
        "manifest_json": str(manifest_json),
        "manifest_csv": str(manifest_csv),
        "clip_count": len(clips),
        "games": sorted({clip.game for clip in clips}),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
