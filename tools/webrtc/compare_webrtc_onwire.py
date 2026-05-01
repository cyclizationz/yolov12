#!/usr/bin/env python3
"""
Compare on-wire bytes for two MP4s by running a WebRTC loopback for each and reading "getStats"-like counters.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _run_one(path: Path, timeout_s: float) -> dict[str, Any]:
    cmd = [
        "python",
        str(Path(__file__).with_name("webrtc_loopback_stats.py")),
        "--in-mp4",
        str(path),
        "--timeout-s",
        str(timeout_s),
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def _bytes_from(res: dict[str, Any]) -> int | None:
    s = (res.get("stats_send") or {}) if isinstance(res.get("stats_send"), dict) else {}
    # Prefer candidate-pair bytes (transport). Fall back to outbound RTP bytes.
    b = s.get("selected_candidate_pair_bytes_sent")
    if isinstance(b, (int, float)):
        return int(b)
    b2 = s.get("outbound_rtp_bytes_sent")
    if isinstance(b2, (int, float)):
        return int(b2)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True, type=Path)
    ap.add_argument("--masked", required=True, type=Path)
    ap.add_argument("--timeout-s", type=float, default=120.0)
    args = ap.parse_args()

    r0 = _run_one(args.orig, args.timeout_s)
    r1 = _run_one(args.masked, args.timeout_s)

    b0 = _bytes_from(r0)
    b1 = _bytes_from(r1)

    saving_pct = None
    if b0 is not None and b0 > 0 and b1 is not None:
        saving_pct = 100.0 * (1.0 - (b1 / b0))

    out = {
        "orig": str(args.orig),
        "masked": str(args.masked),
        "orig_onwire_bytes": b0,
        "masked_onwire_bytes": b1,
        "saving_pct": saving_pct,
        "orig_stats": r0.get("stats_send"),
        "masked_stats": r1.get("stats_send"),
        "orig_duration_s": r0.get("duration_s"),
        "masked_duration_s": r1.get("duration_s"),
        "notes": "onwire_bytes prefers selected candidate-pair bytesSent; falls back to outbound-rtp bytesSent if unavailable",
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


