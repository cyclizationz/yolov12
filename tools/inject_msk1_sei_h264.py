import argparse
import struct
from pathlib import Path


UUID16 = bytes.fromhex("00112233445566778899aabbccddeeff")  # fixed UUID for user_data_unregistered


def read_msk1_bin(path: Path) -> list[bytes]:
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


def annexb_split(byts: bytes) -> list[tuple[bytes, bytes]]:
    """
    Return list of (start_code, nal_bytes_without_start_code).
    start_code is b'\\x00\\x00\\x01' or b'\\x00\\x00\\x00\\x01'.
    """
    n = len(byts)
    i = 0
    starts: list[int] = []
    sc_len: list[int] = []
    while i + 3 < n:
        if byts[i : i + 3] == b"\x00\x00\x01":
            starts.append(i)
            sc_len.append(3)
            i += 3
            continue
        if i + 4 < n and byts[i : i + 4] == b"\x00\x00\x00\x01":
            starts.append(i)
            sc_len.append(4)
            i += 4
            continue
        i += 1
    if not starts:
        return []
    out: list[tuple[bytes, bytes]] = []
    for k, st in enumerate(starts):
        sl = sc_len[k]
        nxt = starts[k + 1] if k + 1 < len(starts) else n
        nal = byts[st + sl : nxt]
        out.append((byts[st : st + sl], nal))
    return out


def ebsp_from_rbsp(rbsp: bytes) -> bytes:
    """
    Add emulation-prevention bytes: after 00 00, if next byte in [00..03], insert 03.
    """
    out = bytearray()
    zcount = 0
    for b in rbsp:
        if zcount >= 2 and b <= 3:
            out.append(3)
            zcount = 0
        out.append(b)
        if b == 0:
            zcount += 1
        else:
            zcount = 0
    return bytes(out)


def build_sei_nal(payload: bytes) -> bytes:
    """
    Build an AnnexB NAL unit (with 4-byte start code) carrying user_data_unregistered (SEI payloadType=5).
    We wrap the MSK1 payload as: uuid(16) + payload.
    """
    sei_data = UUID16 + payload
    payload_type = 5

    rbsp = bytearray()
    # payloadType encoding
    while payload_type >= 255:
        rbsp.append(255)
        payload_type -= 255
    rbsp.append(payload_type)
    # payloadSize encoding
    size = len(sei_data)
    while size >= 255:
        rbsp.append(255)
        size -= 255
    rbsp.append(size)
    rbsp.extend(sei_data)
    # rbsp_trailing_bits
    rbsp.append(0x80)

    nal_header = bytes([0x06])  # nal_unit_type=6 (SEI), nal_ref_idc=0
    ebsp = ebsp_from_rbsp(bytes(rbsp))
    return b"\x00\x00\x00\x01" + nal_header + ebsp


def nal_type(nal: bytes) -> int:
    if not nal:
        return -1
    return nal[0] & 0x1F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-h264", required=True, type=Path, help="Input AnnexB .h264 (must include AUD NALs)")
    ap.add_argument("--msk1-bin", required=True, type=Path, help="Input msk1_payloads.bin (len+payload per frame)")
    ap.add_argument("--out-h264", required=True, type=Path, help="Output AnnexB .h264 with injected SEI")
    args = ap.parse_args()

    payloads = read_msk1_bin(args.msk1_bin)
    byts = args.in_h264.read_bytes()
    nals = annexb_split(byts)
    if not nals:
        raise SystemExit("failed to parse AnnexB stream (no start codes found)")

    out = bytearray()
    frame_idx = -1
    for sc, nal in nals:
        t = nal_type(nal)
        # Access Unit Delimiter indicates a new frame boundary (when aud=1).
        if t == 9:
            frame_idx += 1
            out.extend(sc)
            out.extend(nal)
            if 0 <= frame_idx < len(payloads):
                out.extend(build_sei_nal(payloads[frame_idx]))
            continue
        out.extend(sc)
        out.extend(nal)

    args.out_h264.parent.mkdir(parents=True, exist_ok=True)
    args.out_h264.write_bytes(bytes(out))
    print(f"Injected SEI for {min(frame_idx+1, len(payloads))} frames into {args.out_h264}")


if __name__ == "__main__":
    main()


