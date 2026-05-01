#!/usr/bin/env bash
set -euo pipefail

# End-to-end demo:
# 1) Run deployment tool to produce masked video + dict templates + per-frame MSK1 payloads (sidecar).
# 2) Encode masked stream to AnnexB H.264 with AUD enabled.
# 3) Inject MSK1 into in-band SEI NALs.
# 4) Mux to MP4.
# 5) Stitch/recover by reading MSK1 sidecar (or extracted, if you add an extractor) + dict templates.
#
# NOTE: True "client" extraction from bitstream is not implemented here; injection is.
#       Stitching uses the same payloads that were injected.

ROOT=/home/tiehangz/proj/yolov12
DEP=$ROOT/deployment

IN=${1:-/home/tiehangz/proj/datasets/fps/roi_720p.mp4}
OUT=${2:-$ROOT/experiments/fc5_inband_flow}

mkdir -p "$OUT"
mkdir -p "$OUT/server"

echo "[1] Run Yolov12Deployment (masked + payloads + dict)..."
cd "$DEP"
./build/Yolov12Deployment \
  -i "$IN" \
  -o "$OUT/server" \
  -m "$DEP/yolov12n_fc5_seg_v1.onnx" \
  --latent-key \
  --latent-thr 2.0 \
  --latent-period 1 \
  -c 0.05 \
  -s 0.30 \
  --yolo-class-color \
  --paint-alpha 1.0 \
  --max-frames 120 \
  --timing \
  > "$OUT/server/run.log" 2>&1

MASKED_MP4="$OUT/server/segmented_output.mp4"
MSK1_BIN="$OUT/server/msk1_payloads.bin"
DICT_DIR="$OUT/server/dict"

echo "[2] Encode masked MP4 -> AnnexB H264 (AUD enabled)..."
ffmpeg -y -v error -i "$MASKED_MP4" \
  -c:v libx264 -preset veryfast -crf 23 \
  -x264-params "aud=1:annexb=1:keyint=60:min-keyint=60:scenecut=0:bframes=0" \
  -an "$OUT/masked_annexb.h264"

echo "[3] Inject MSK1 into SEI NALs..."
conda run -n yolov12 python "$ROOT/tools/inject_msk1_sei_h264.py" \
  --in-h264 "$OUT/masked_annexb.h264" \
  --msk1-bin "$MSK1_BIN" \
  --out-h264 "$OUT/masked_inband_sei.h264"

echo "[4] Mux injected H264 -> MP4..."
ffmpeg -y -v error -r 30 -i "$OUT/masked_inband_sei.h264" -c copy "$OUT/masked_inband_sei.mp4"

echo "[5] Extract MSK1 back from in-band SEI (proof of in-band transport)..."
conda run -n yolov12 python "$ROOT/tools/extract_msk1_sei_h264.py" \
  --in-h264 "$OUT/masked_inband_sei.h264" \
  --out-bin "$OUT/msk1_extracted.bin"

echo "[6] Stitch (client) using EXTRACTED MSK1 + dict templates..."
conda run -n yolov12 python "$ROOT/tools/stitch_from_msk1_payloads.py" \
  --video "$OUT/masked_inband_sei.mp4" \
  --msk1-bin "$OUT/msk1_extracted.bin" \
  --dict-dir "$DICT_DIR" \
  --out "$OUT/recovered_from_sei.mp4" \
  --assume-templates-available

echo "DONE. Outputs in $OUT"


