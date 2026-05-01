#!/usr/bin/env python3
"""
Run a local WebRTC loopback (sender <-> receiver) using GStreamer webrtcbin and report on-wire bytes.

Design goals:
- Minimal change to existing pipeline: stream an existing MP4 (H.264) without re-encoding.
- Report "getStats"-style numbers: outbound-rtp bytesSent and selected candidate-pair bytesSent.

Notes:
- This uses system Python's PyGObject (`gi`). It is typically not available in conda envs.
- This is a localhost loopback. Real WAN overhead and congestion control behavior will differ.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import gi  # type: ignore

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
from gi.repository import GLib, Gst, GstSdp  # type: ignore  # noqa: E402


def _gst_value_to_py(v: Any) -> Any:
    # Best-effort conversion for Gst/GLib values commonly returned in webrtcbin stats.
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    # Gst.Structure
    if hasattr(v, "n_fields") and hasattr(v, "get_name"):
        return _structure_to_dict(v)
    # Gst.ValueList / list-like
    if hasattr(v, "__len__") and not isinstance(v, (bytes, bytearray)):
        try:
            return [_gst_value_to_py(x) for x in list(v)]
        except Exception:
            pass
    # Fallback: stringify (still useful for debugging)
    return str(v)


def _structure_to_dict(s: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"_name": str(s.get_name())}
    try:
        n = s.n_fields()
    except Exception:
        return d
    for i in range(n):
        k = s.nth_field_name(i)
        try:
            d[k] = _gst_value_to_py(s.get_value(k))
        except Exception:
            d[k] = None
    return d


def _parse_webrtc_stats(reply_struct: Any) -> dict[str, Any]:
    # webrtcbin stats format varies by version:
    # - Some builds return a top-level "stats" array
    # - Others return a structure with many fields, each field being a stat structure keyed by id
    top = _structure_to_dict(reply_struct)

    stats_list: list[dict[str, Any]] = []
    if isinstance(top.get("stats"), list):
        for x in top["stats"]:
            if isinstance(x, dict):
                stats_list.append(x)
    else:
        for k, v in top.items():
            if k == "_name":
                continue
            if isinstance(v, dict) and "type" in v and "id" in v:
                stats_list.append(v)

    outbound_rtp_bytes = 0
    candidate_pair_bytes = None
    selected_pair_id = None

    for st in stats_list:
        if not isinstance(st, dict):
            continue
        typ = st.get("type") or st.get("statsType") or st.get("stats-type")
        # Some builds expose `type` as an enum/int and `_name` holds the human-readable type.
        if isinstance(typ, (int, float)):
            typ = st.get("_name")
        if typ == "outbound-rtp":
            bs = st.get("bytesSent") or st.get("bytes-sent")
            if isinstance(bs, (int, float)):
                outbound_rtp_bytes += int(bs)
        if typ == "candidate-pair":
            sel = st.get("selected")
            # Some implementations use "selected" boolean, others use "nominated"/"state".
            is_selected = bool(sel) if sel is not None else False
            if not is_selected:
                continue
            bs = st.get("bytesSent") or st.get("bytes-sent")
            if isinstance(bs, (int, float)):
                candidate_pair_bytes = int(bs)
                selected_pair_id = st.get("id")

    return {
        "outbound_rtp_bytes_sent": outbound_rtp_bytes,
        "selected_candidate_pair_bytes_sent": candidate_pair_bytes,
        "selected_candidate_pair_id": selected_pair_id,
        "stats_entries": len(stats_list),
    }


class Loopback:
    def __init__(self, mp4: Path, timeout_s: float, run_s: float, debug_dump: bool) -> None:
        self.mp4 = mp4
        self.timeout_s = timeout_s
        self.run_s = run_s
        self.debug_dump = debug_dump
        self.loop = GLib.MainLoop()
        self.t0 = time.time()
        self.t_end: float | None = None
        self.result: dict[str, Any] = {}

        # Build pipeline programmatically to avoid fragile parse-launch linking with request pads.
        self.pipeline = Gst.Pipeline.new("webrtc-loopback")
        if self.pipeline is None:
            raise RuntimeError("failed to create pipeline")

        self.filesrc = Gst.ElementFactory.make("filesrc", None)
        self.demux = Gst.ElementFactory.make("qtdemux", "demux")
        self.q1 = Gst.ElementFactory.make("queue", None)
        self.parse = Gst.ElementFactory.make("h264parse", None)
        self.pay = Gst.ElementFactory.make("rtph264pay", None)
        self.capsf = Gst.ElementFactory.make("capsfilter", None)
        self.q2 = Gst.ElementFactory.make("queue", None)
        self.send = Gst.ElementFactory.make("webrtcbin", "send")
        self.recv = Gst.ElementFactory.make("webrtcbin", "recv")

        elems = [self.filesrc, self.demux, self.q1, self.parse, self.pay, self.capsf, self.q2, self.send, self.recv]
        if any(e is None for e in elems):
            raise RuntimeError("failed to create one or more GStreamer elements")

        self.filesrc.set_property("location", str(mp4))
        self.parse.set_property("config-interval", -1)
        self.pay.set_property("pt", 96)
        self.pay.set_property("config-interval", 1)
        self.send.set_property("bundle-policy", "max-bundle")
        self.recv.set_property("bundle-policy", "max-bundle")
        # Avoid pipeline state change deadlocks when DTLS/ICE are still negotiating.
        self.send.set_property("async-handling", True)
        self.recv.set_property("async-handling", True)
        # For loopback measurement we don't want extra buffering.
        self.send.set_property("latency", 0)
        self.recv.set_property("latency", 0)

        caps = Gst.Caps.from_string("application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000")
        self.capsf.set_property("caps", caps)

        for e in elems:
            self.pipeline.add(e)

        # Static links
        self.filesrc.link(self.demux)
        self.q1.link(self.parse)
        self.parse.link(self.pay)
        self.pay.link(self.capsf)
        self.capsf.link(self.q2)

        # Link RTP into webrtcbin request sink pad
        sinkpad = self.send.get_request_pad("sink_%u")
        if sinkpad is None:
            raise RuntimeError("failed to request webrtcbin sink pad")
        srcpad = self.q2.get_static_pad("src")
        if srcpad is None:
            raise RuntimeError("failed to get q2 src pad")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("failed to link RTP into webrtcbin")

        # Dynamic link: qtdemux pad-added -> q1 sink
        self.demux.connect("pad-added", self._on_demux_pad_added)

        self.send.connect("on-negotiation-needed", self._on_negotiation_needed)
        self.send.connect("on-ice-candidate", self._on_ice_candidate, self.recv)
        self.recv.connect("on-ice-candidate", self._on_ice_candidate, self.send)
        self.recv.connect("pad-added", self._on_recv_pad_added)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        # Timeout watchdog
        GLib.timeout_add(int(self.timeout_s * 1000), self._on_timeout)
        # Stop condition: run for run_s, then collect stats and stop (can't rely on EOS because recv is an open-ended source).
        GLib.timeout_add(int(self.run_s * 1000), self._on_run_done)

    def _on_timeout(self) -> bool:
        if self.t_end is None:
            self.result["error"] = f"timeout after {self.timeout_s}s"
            self._stop()
        return False

    def _on_run_done(self) -> bool:
        if self.t_end is None:
            self.result["stats_send"] = self._get_stats(self.send)
            self.result["stats_recv"] = self._get_stats(self.recv)
            self._stop()
        return False

    def _stop(self) -> None:
        if self.t_end is None:
            self.t_end = time.time()
        self.pipeline.set_state(Gst.State.NULL)
        try:
            self.loop.quit()
        except Exception:
            pass

    def _on_bus_message(self, _bus: Gst.Bus, msg: Gst.Message) -> None:
        t = msg.type
        if t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            self.result["error"] = str(err)
            if dbg:
                self.result["debug"] = str(dbg)
            self._stop()
        # Note: we don't rely on EOS because the receiver is open-ended and may prevent the pipeline from reaching EOS.

    def _on_recv_pad_added(self, _webrtc: Gst.Element, pad: Gst.Pad) -> None:
        # Link incoming RTP to a depay+fakesink so packets are consumed.
        caps = pad.get_current_caps() or pad.get_caps()
        if caps is None or caps.get_size() == 0:
            return
        s = caps.get_structure(0)
        if s is None:
            return
        if s.get_name() != "application/x-rtp":
            return
        # Create: queue ! rtph264depay ! fakesink sync=false
        q = Gst.ElementFactory.make("queue", None)
        depay = Gst.ElementFactory.make("rtph264depay", None)
        sink = Gst.ElementFactory.make("fakesink", None)
        if sink is not None:
            sink.set_property("sync", False)
        if q is None or depay is None or sink is None:
            return
        self.pipeline.add(q)
        self.pipeline.add(depay)
        self.pipeline.add(sink)
        q.sync_state_with_parent()
        depay.sync_state_with_parent()
        sink.sync_state_with_parent()
        q.link(depay)
        depay.link(sink)
        pad.link(q.get_static_pad("sink"))

    def _on_demux_pad_added(self, _demux: Gst.Element, pad: Gst.Pad) -> None:
        # Connect demuxed H264 to q1.
        caps = pad.get_current_caps() or pad.get_caps()
        if caps is None or caps.get_size() == 0:
            return
        s = caps.get_structure(0)
        if s is None:
            return
        name = s.get_name()
        # qtdemux typically outputs "video/x-h264"
        if not name.startswith("video/"):
            return
        sinkpad = self.q1.get_static_pad("sink")
        if sinkpad is None or sinkpad.is_linked():
            return
        pad.link(sinkpad)

    def _on_ice_candidate(self, _webrtc: Gst.Element, mlineindex: int, candidate: str, other: Gst.Element) -> None:
        other.emit("add-ice-candidate", mlineindex, candidate)

    def _on_negotiation_needed(self, element: Gst.Element) -> None:
        promise = Gst.Promise.new_with_change_func(self._on_offer_created, element)
        element.emit("create-offer", None, promise)

    def _on_offer_created(self, promise: Gst.Promise, element: Gst.Element) -> None:
        promise.wait()
        reply = promise.get_reply()
        offer = reply.get_value("offer")
        element.emit("set-local-description", offer, None)

        # Send offer to receiver
        self.recv.emit("set-remote-description", offer, None)
        promise2 = Gst.Promise.new_with_change_func(self._on_answer_created, self.recv)
        self.recv.emit("create-answer", None, promise2)

    def _on_answer_created(self, promise: Gst.Promise, element: Gst.Element) -> None:
        promise.wait()
        reply = promise.get_reply()
        answer = reply.get_value("answer")
        element.emit("set-local-description", answer, None)
        self.send.emit("set-remote-description", answer, None)

    def _get_stats(self, webrtc: Gst.Element) -> dict[str, Any]:
        p = Gst.Promise.new()
        webrtc.emit("get-stats", None, p)
        p.wait()
        r = p.get_reply()
        if r is None:
            return {"error": "no-reply"}
        parsed = _parse_webrtc_stats(r)
        if self.debug_dump:
            parsed["raw"] = _structure_to_dict(r)
        return parsed

    def run(self) -> dict[str, Any]:
        self.pipeline.set_state(Gst.State.PLAYING)
        self.loop.run()
        self.result.setdefault("input_mp4", str(self.mp4))
        self.result.setdefault("duration_s", (self.t_end or time.time()) - self.t0)
        return self.result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-mp4", required=True, type=Path)
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument(
        "--run-s",
        type=float,
        default=0.0,
        help="How long to run before sampling stats. If 0, infer from ffprobe duration + 0.5s.",
    )
    ap.add_argument("--debug-dump", action="store_true", help="Include raw stats structure dump (large).")
    args = ap.parse_args()

    Gst.init(None)
    # Some distros require explicit SDP init for offer/answer parsing (safe no-op otherwise)
    try:
        GstSdp.sdp_init()
    except Exception:
        pass

    run_s = args.run_s
    if run_s <= 0:
        # Infer file duration (seconds)
        try:
            out = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(args.in_mp4),
                ],
                text=True,
            ).strip()
            run_s = float(out) + 0.5
        except Exception:
            run_s = 10.0
    # Ensure timeout >= run duration
    timeout_s = max(args.timeout_s, run_s + 5.0)

    res = Loopback(args.in_mp4, timeout_s, run_s, args.debug_dump).run()
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()


