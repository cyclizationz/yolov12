import argparse
import json
import struct
from pathlib import Path


def ffprobe_video_packet_bytes(video: Path) -> int:
    """
    Sum of packet sizes for the video stream as reported by ffprobe -show_packets.
    This approximates coded payload bytes in the container for H.264/H.265 MP4/MKV.
    """
    import subprocess

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_entries",
        "packet=size",
        "-of",
        "json",
        str(video),
    ]
    out = subprocess.check_output(cmd, text=True)
    j = json.loads(out)
    total = 0
    for p in j.get("packets", []) or []:
        sz = p.get("size")
        if sz is None:
            continue
        total += int(sz)
    return total


def read_len_prefixed_payloads(path: Path) -> list[bytes]:
    data = path.read_bytes()
    out: list[bytes] = []
    off = 0
    while off + 4 <= len(data):
        (n,) = struct.unpack_from("<I", data, off)
        off += 4
        if n == 0:
            out.append(b"")
            continue
        if off + n > len(data):
            break
        out.append(data[off : off + n])
        off += n
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute MSK1 payload overhead statistics from msk1_payloads.bin")
    ap.add_argument("--msk1-bin", required=True, type=Path)
    ap.add_argument("--video", type=Path, default=None, help="Optional: video file to compute coded payload bytes via ffprobe")
    ap.add_argument("--video-bytes", type=int, default=None, help="Optional: encoded video payload bytes to compute percentages")
    args = ap.parse_args()

    payloads = read_len_prefixed_payloads(args.msk1_bin)
    payload_bytes = sum(len(p) for p in payloads)
    # msk1_payloads.bin stores <u32 len> + payload per frame.
    len_prefix_bytes = 4 * len(payloads)
    file_bytes = args.msk1_bin.stat().st_size

    nonempty = sum(1 for p in payloads if p)
    empty = len(payloads) - nonempty

    out = {
        "frames": len(payloads),
        "msk1_payload_bytes": payload_bytes,
        "len_prefix_bytes": len_prefix_bytes,
        "msk1_bin_file_bytes": file_bytes,
        "nonempty_frames": nonempty,
        "empty_frames": empty,
        "avg_payload_bytes_per_frame": (payload_bytes / len(payloads)) if payloads else 0.0,
    }
    vb = None
    if args.video is not None:
        try:
            vb = ffprobe_video_packet_bytes(args.video)
        except Exception:
            vb = None
    if vb is None and args.video_bytes is not None and args.video_bytes > 0:
        vb = int(args.video_bytes)
    if vb is not None and vb > 0:
        out["video_bytes"] = vb
        out["payload_pct_of_video_bytes"] = 100.0 * payload_bytes / vb
        out["msk1bin_pct_of_video_bytes"] = 100.0 * file_bytes / vb

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


