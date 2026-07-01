"""Shared Mario pixel + FC5 mask defaults for Exp1 and pixel sweeps."""
from __future__ import annotations

from pathlib import Path

from common import REPO_ROOT

PIXEL_TEMPLATES = REPO_ROOT / "experiments" / "encoder_eval/_pixel_single_template"

# Pixel sweep best-effort winner: p90_b15_f1_k0.80_og1 (record/RESPAWN2026/pixel_enc_sweep)
MARIO_SWEEP_PEAKS = 90
MARIO_SWEEP_BOOTSTRAP = 15
MARIO_SWEEP_FLOW = 1
MARIO_SWEEP_THR_K = 0.80

FC5_MASK_GLOBAL = "global_period200"
FC5_MASK_LOCAL_PAD64 = "local_pad64_mode"
FC5_MASK_PROFILES = (FC5_MASK_GLOBAL, FC5_MASK_LOCAL_PAD64)


def mario_pixel_args(
    *,
    peaks: int = MARIO_SWEEP_PEAKS,
    bootstrap: int = MARIO_SWEEP_BOOTSTRAP,
    flow: int = MARIO_SWEEP_FLOW,
    thr_k: float = MARIO_SWEEP_THR_K,
) -> list[str]:
    return [
        "--pixel",
        "-T",
        str(PIXEL_TEMPLATES),
        "--pixel-force-scale",
        "1.0",
        "--pixel-mask-pad",
        "2",
        "--pixel-grid-header",
        "--pixel-grid-snap",
        "12",
        "--pixel-band-pad-y",
        "80",
        "--pixel-bands",
        "4",
        "--pixel-max-peaks",
        str(peaks),
        "--pixel-bootstrap",
        str(bootstrap),
        "--pixel-flow",
        str(flow),
        "--pixel-adaptive-thr",
        "1",
        "--pixel-thr-k",
        str(thr_k),
        "--pixel-thr-lo",
        "0.30",
        "--pixel-thr-hi",
        "0.65",
        "--pixel-min-score",
        "0.65",
        "--mask-color",
        "dominant",
        "--mask-color-period",
        "200",
        "--fill-mode",
        "solid",
        "--feather-px",
        "0",
    ]


def fc5_mask_profile_args(profile: str) -> list[str]:
    if profile == FC5_MASK_GLOBAL:
        return []
    if profile == FC5_MASK_LOCAL_PAD64:
        return [
            "--mask-color-local-pad",
            "64",
            "--mask-color-local-per-region",
            "--mask-color-local-stat",
            "mode",
        ]
    raise ValueError(f"unknown FC5 mask profile: {profile!r}")


def fc5_mask_profile_protocol_line(profile: str) -> str:
    if profile == FC5_MASK_GLOBAL:
        return (
            "- FC5 mask profile `global_period200`: global dominant color, "
            "`mask-color-period=200`, `fill-mode=solid`, `feather-px=4`"
        )
    if profile == FC5_MASK_LOCAL_PAD64:
        return (
            "- FC5 mask profile `local_pad64_mode`: per-region local dominant (stat=mode), "
            "`mask-color-local-pad=64`, `feather-px=4` (gop_analysis full-clip winner)"
        )
    raise ValueError(f"unknown FC5 mask profile: {profile!r}")


def mario_encoder_protocol_line() -> str:
    return (
        "- Exp1 encoder (matched pure + RESPAWN, all games): `libx264`, `medium`, `tune=none`, "
        "`--enc-open-gop-defaults`, fixed VBV sweep (no CRF)"
    )


def mario_pixel_protocol_line() -> str:
    return (
        "- Mario pixel mode (50×44 template): `pixel-force-scale=1.0`, "
        f"`pixel-max-peaks={MARIO_SWEEP_PEAKS}`, `pixel-bootstrap={MARIO_SWEEP_BOOTSTRAP}`, "
        f"`pixel-flow={MARIO_SWEEP_FLOW}`, `pixel-adaptive-thr=1`, "
        f"`pixel-thr-k={MARIO_SWEEP_THR_K}`, grid header/snap, band scan"
    )


def exp1_refresh_note() -> str:
    return (
        "## Exp1 refresh (May 2026)\n\n"
        "- Mario uses the 50×44 `brick_large_brown.png` template with pixel-sweep winner "
        f"`p{MARIO_SWEEP_PEAKS}_b{MARIO_SWEEP_BOOTSTRAP}_f{MARIO_SWEEP_FLOW}_k{MARIO_SWEEP_THR_K:.2f}_og1`.\n"
        "- FC5 uses the selected gop_analysis mask profile under the fixed 8–24 Mbps VBV RD protocol.\n"
        "- Mario sweep winner did not meet VMAF≥55 at the 16 Mbps anchor; Exp1 uses a multi-rate sweep instead.\n"
    )
