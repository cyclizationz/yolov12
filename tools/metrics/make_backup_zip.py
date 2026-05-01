import argparse
import time
import zipfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a compact backup zip of logs/tables (exclude videos, dict images).")
    ap.add_argument("--repo", type=Path, default=Path("/home/tiehangz/proj/yolov12"))
    ap.add_argument("--out", type=Path, default=None, help="Output zip path (default: repo/backup/*.zip)")
    args = ap.parse_args()

    repo = args.repo
    roots = [repo / "record", repo / "experiments"]

    allow_exts = {
        ".csv",
        ".json",
        ".jsonl",
        ".md",
        ".tex",
        ".txt",
        ".log",
        ".bin",
        ".png",
        ".jpg",
        ".jpeg",
    }
    deny_exts = {".mp4", ".mkv", ".y4m", ".yuv", ".h264", ".hevc", ".ivf"}
    image_exts = {".png", ".jpg", ".jpeg"}

    # Drop huge template dumps. Keep normal figures/screenshots.
    def should_skip(p: Path) -> bool:
        ext = p.suffix.lower()
        if ext in deny_exts:
            return True
        if ext not in allow_exts:
            return True
        parts = {s.lower() for s in p.parts}
        if "dict" in parts and ext in image_exts:
            return True
        return False

    backup_dir = repo / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        ts = time.strftime("%Y-%m-%d")
        out = backup_dir / f"record_experiments_logs_tables_{ts}.zip"
    else:
        out = args.out
        out.parent.mkdir(parents=True, exist_ok=True)

    to_add: list[Path] = []
    for r in roots:
        if not r.exists():
            continue
        for p in r.rglob("*"):
            if not p.is_file():
                continue
            if should_skip(p):
                continue
            to_add.append(p)

    # Write zip (store=0 for already-compressed formats, deflate otherwise).
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(to_add):
            arc = p.relative_to(repo).as_posix()
            zf.write(p, arcname=arc)

    print("WROTE", out)
    print("FILES", len(to_add))


if __name__ == "__main__":
    main()

