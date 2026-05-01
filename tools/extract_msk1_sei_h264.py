import argparse
import struct
from pathlib import Path


UUID16 = bytes.fromhex("00112233445566778899aabbccddeeff")


def annexb_split(byts: bytes) -> list[tuple[bytes, bytes]]:
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


def nal_type(nal: bytes) -> int:
    if not nal:
        return -1
    return nal[0] & 0x1F


def rbsp_from_ebsp(ebsp: bytes) -> bytes:
    # remove emulation prevention bytes (0x03 after 00 00)
    out = bytearray()
    zcount = 0
    i = 0
    while i < len(ebsp):
        b = ebsp[i]
        if zcount >= 2 and b == 3:
            # skip EPB
            i += 1
            zcount = 0
            continue
        out.append(b)
        if b == 0:
            zcount += 1
        else:
            zcount = 0
        i += 1
    return bytes(out)


def parse_user_data_unregistered_from_sei(sei_nal: bytes) -> bytes | None:
    """
    Parse SEI RBSP, return MSK1 payload bytes if found (uuid matches and payload starts with 'MSK1').
    Only handles payloadType=5 (user_data_unregistered).
    """
    if not sei_nal or nal_type(sei_nal) != 6:
        return None
    rbsp = rbsp_from_ebsp(sei_nal[1:])  # skip NAL header
    # remove rbsp_trailing_bits (0x80 + alignment zeros). safest: strip from last 0x80.
    if rbsp and rbsp[-1] == 0x80:
        rbsp = rbsp[:-1]
    off = 0
    while off < len(rbsp):
        # payloadType
        pt = 0
        while off < len(rbsp) and rbsp[off] == 255:
            pt += 255
            off += 1
        if off >= len(rbsp):
            break
        pt += rbsp[off]
        off += 1
        # payloadSize
        ps = 0
        while off < len(rbsp) and rbsp[off] == 255:
            ps += 255
            off += 1
        if off >= len(rbsp):
            break
        ps += rbsp[off]
        off += 1
        if off + ps > len(rbsp):
            break
        payload = rbsp[off : off + ps]
        off += ps
        if pt == 5 and len(payload) >= 16 + 4:
            uuid = payload[:16]
            data = payload[16:]
            if uuid == UUID16 and data[:4] == b"MSK1":
                return data
        # move to next message; there may be padding
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-h264", required=True, type=Path, help="Input AnnexB .h264 with injected SEI and AUD")
    ap.add_argument("--out-bin", required=True, type=Path, help="Output len-prefixed MSK1 payloads.bin")
    args = ap.parse_args()

    byts = args.in_h264.read_bytes()
    nals = annexb_split(byts)
    if not nals:
        raise SystemExit("failed to parse AnnexB stream (no start codes found)")

    payloads: list[bytes] = []
    expecting = False
    for _, nal in nals:
        t = nal_type(nal)
        if t == 9:  # AUD
            expecting = True
            continue
        if expecting and t == 6:
            p = parse_user_data_unregistered_from_sei(nal)
            if p is not None:
                payloads.append(p)
                expecting = False
                continue
        # If we hit VCL without SEI, still advance by emitting empty payload for that AU.
        if expecting and t in (1, 5):
            payloads.append(b"")
            expecting = False

    out = bytearray()
    for p in payloads:
        out += struct.pack("<I", len(p))
        out += p
    args.out_bin.parent.mkdir(parents=True, exist_ok=True)
    args.out_bin.write_bytes(bytes(out))
    print(f"Extracted {len(payloads)} payloads to {args.out_bin}")


if __name__ == "__main__":
    main()


