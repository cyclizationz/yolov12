#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


REPO_ROOT = Path("/home/tiehangz/proj/yolov12")
RESULT_TEX = Path("/home/tiehangz/Downloads/Result.tex")
TARGET_ROOT = Path("/home/tiehangz/Downloads/RESPAWN_2026 (2)/results_artifacts")
RD_POINTS_CSV = REPO_ROOT / "record/RESPAWN2026/exp1/rd_suite_points.csv"
OVERHEAD_SUMMARY_CSV = REPO_ROOT / "record/RESPAWN2026/exp3/overhead_summary.csv"
GENERALITY_SUMMARY_CSV = REPO_ROOT / "record/RESPAWN2026/exp5/generality_summary.csv"
FIGURE7_RUN_NAMES = ("fm6_03_23p0", "fm6_04_23p0")


def col_to_a1(col: int) -> str:
    out = []
    c = col
    while c >= 0:
        out.append(chr((c % 26) + ord("A")))
        c = c // 26 - 1
    return "".join(reversed(out))


def _cell_xml(r: int, c: int, value: Any) -> str:
    ref = f"{col_to_a1(c)}{r + 1}"
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def write_xlsx(path: Path, sheets: dict[str, list[list[Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_items = list(sheets.items())
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    ]
    for i in range(1, len(sheet_items) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
"""
    wb_sheets = []
    wb_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for i, (name, _) in enumerate(sheet_items, start=1):
        wb_sheets.append(
            f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
        )
        wb_rels.append(
            f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
        )
    wb_rels.append("</Relationships>")
    workbook = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
            "<sheets>",
            *wb_sheets,
            "</sheets>",
            "</workbook>",
        ]
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "\n".join(content_types))
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", "\n".join(wb_rels))
        for i, (_, rows) in enumerate(sheet_items, start=1):
            sheet_rows = []
            for r, row in enumerate(rows):
                cells = "".join(_cell_xml(r, c, val) for c, val in enumerate(row))
                sheet_rows.append(f"<row r=\"{r + 1}\">{cells}</row>")
            sheet_xml = "\n".join(
                [
                    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                    '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
                    "<sheetData>",
                    *sheet_rows,
                    "</sheetData>",
                    "</worksheet>",
                ]
            )
            zf.writestr(f"xl/worksheets/sheet{i}.xml", sheet_xml)


def write_csv_rows(path: Path, rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def parse_float_maybe(value: str) -> float | None:
    text = value.strip()
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", text)
    text = text.replace("$", "").replace("\\%", "%")
    text = text.replace("\\texttt{", "").replace("}", "")
    text = text.replace("%", "").replace("+", "")
    text = text.strip()
    if text in ("", "--", "-"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def latex_strip(value: str) -> str:
    text = value.strip()
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", text)
    text = text.replace("$", "").replace("\\%", "%")
    text = text.replace("\\texttt{", "").replace("}", "")
    return text.strip()


def parse_tabular_rows_from_tex(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if "&" not in s or "\\\\" not in s:
            continue
        if s.startswith("%"):
            continue
        s = s.split("\\\\", 1)[0].strip()
        cells = [latex_strip(x) for x in s.split("&")]
        if cells:
            rows.append(cells)
    return rows


def parse_table7_rows(result_tex_path: Path) -> list[list[str]]:
    lines = result_tex_path.read_text(encoding="utf-8").splitlines()
    start = None
    for i, line in enumerate(lines):
        if "\\label{tab:exp1-fixed-budget}" in line:
            start = i
            break
    if start is None:
        raise RuntimeError("Could not find tab:exp1-fixed-budget in Result.tex")
    block = []
    in_tabular = False
    for line in lines[start:]:
        if "\\begin{tabular}" in line:
            in_tabular = True
            continue
        if in_tabular and "\\end{tabular}" in line:
            break
        if in_tabular:
            block.append(line)
    rows = []
    for line in block:
        s = line.strip()
        if "&" not in s or "\\\\" not in s:
            continue
        s = s.split("\\\\", 1)[0]
        cells = [latex_strip(x) for x in s.split("&")]
        if cells and cells[0] in ("FC5", "FM6", "Mario"):
            rows.append(cells)
    return rows


def copy_assets(paths: list[Path], item_dir: Path) -> list[str]:
    copied = []
    for i, src in enumerate(paths, start=1):
        ext = src.suffix
        if len(paths) == 1:
            dst = item_dir / f"paper_asset{ext}"
        else:
            dst = item_dir / f"paper_asset_{i}{ext}"
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def copy_source_csvs(paths: list[Path], item_dir: Path) -> list[str]:
    copied = []
    for i, src in enumerate(paths, start=1):
        if src.suffix.lower() != ".csv":
            continue
        dst = item_dir / f"source_data_{i}.csv"
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def build_table8_from_rd_points() -> tuple[list[list[Any]], list[list[Any]]]:
    rate_order = [8.0, 12.0, 16.0, 20.0, 24.0]
    games = ["fc5", "fm6", "mario"]
    accum: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    with RD_POINTS_CSV.open("r", newline="") as f:
        for row in csv.DictReader(f):
            variant = row.get("variant", "")
            if variant not in ("pure_streaming", "respawn"):
                continue
            game = row.get("game", "")
            if game not in games:
                continue
            rate = float(row.get("rate_point_mbps", "0") or 0)
            if rate not in rate_order:
                continue
            ssim = row.get("ssim_mean", "")
            if ssim in ("", None):
                continue
            accum[(game, rate, variant)].append(float(ssim))

    data_long = [[
        "experiment",
        "item_number",
        "game",
        "rate_mbps",
        "variant",
        "metric",
        "value",
        "unit",
        "source_file",
    ]]
    for game in games:
        for rate in rate_order:
            for variant in ("pure_streaming", "respawn"):
                vals = accum.get((game, rate, variant), [])
                mean = sum(vals) / len(vals) if vals else None
                data_long.append(
                    [
                        "exp1",
                        "table8",
                        game.upper(),
                        rate,
                        variant,
                        "full_frame_ssim",
                        mean if mean is not None else "",
                        "unitless",
                        str(RD_POINTS_CSV),
                    ]
                )

    paper_view = [["Game", "8", "12", "16", "20", "24"]]
    for game in games:
        row = [game.upper()]
        for rate in rate_order:
            pure = accum.get((game, rate, "pure_streaming"), [])
            resp = accum.get((game, rate, "respawn"), [])
            p = (sum(pure) / len(pure)) if pure else None
            r = (sum(resp) / len(resp)) if resp else None
            if p is None or r is None:
                row.append("")
            else:
                row.append(f"{p:.3f} / {r:.3f}")
        paper_view.append(row)
    return data_long, paper_view


def table_tex_to_sheets(exp: str, item_number: str, tex_path: Path) -> tuple[list[list[Any]], list[list[Any]]]:
    rows = parse_tabular_rows_from_tex(tex_path)
    if not rows:
        return (
            [["experiment", "item_number", "source_file"]],
            [["No rows parsed", str(tex_path)]],
        )
    header = rows[0]
    data_long = [[
        "experiment",
        "item_number",
        "row_label",
        "col_label",
        "value_raw",
        "value_numeric",
        "source_file",
    ]]
    for row in rows[1:]:
        if not row:
            continue
        row_label = row[0]
        for idx in range(1, len(row)):
            col_label = header[idx] if idx < len(header) else f"col_{idx + 1}"
            raw = row[idx]
            data_long.append(
                [
                    exp,
                    item_number,
                    row_label,
                    col_label,
                    raw,
                    parse_float_maybe(raw) if parse_float_maybe(raw) is not None else "",
                    str(tex_path),
                ]
            )
    return data_long, rows


def build_figure7_sheets() -> tuple[list[list[Any]], list[list[Any]]]:
    rows = []
    with OVERHEAD_SUMMARY_CSV.open("r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("run_name") not in FIGURE7_RUN_NAMES:
                continue
            if row.get("cache_mode") not in ("cold_start", "warm_start"):
                continue
            rows.append(row)
    rows.sort(key=lambda r: (r.get("run_name", ""), r.get("cache_mode", ""), float(r.get("rtt_ms", "0") or 0), int(r.get("delay_rtts", "0") or 0)))
    data_long = [[
        "experiment",
        "item_number",
        "run_name",
        "game",
        "cache_mode",
        "rtt_ms",
        "delay_rtts",
        "forced_raw_fraction",
        "template_bytes_sent_pct",
        "final_net_saved_pct",
        "source_file",
    ]]
    for row in rows:
        data_long.append(
            [
                "exp2",
                "figure7",
                row.get("run_name", ""),
                row.get("game", ""),
                row.get("cache_mode", ""),
                float(row.get("rtt_ms", "0") or 0),
                int(float(row.get("delay_rtts", "0") or 0)),
                float(row.get("forced_raw_fraction", "0") or 0),
                float(row.get("template_bytes_sent_pct", "0") or 0),
                float(row.get("final_net_saved_pct", "0") or 0),
                str(OVERHEAD_SUMMARY_CSV),
            ]
        )
    paper_view = [["run_name", "cache_mode", "rtt_ms", "delay_rtts", "final_net_saved_pct"]]
    for row in rows:
        paper_view.append([
            row.get("run_name", ""),
            row.get("cache_mode", ""),
            row.get("rtt_ms", ""),
            row.get("delay_rtts", ""),
            row.get("final_net_saved_pct", ""),
        ])
    return data_long, paper_view


def build_figure7_subset_rows() -> list[list[Any]]:
    keep = [
        "run_name",
        "clip_id",
        "game",
        "cache_mode",
        "rtt_ms",
        "delay_rtts",
        "forced_raw_fraction",
        "template_bytes_sent_pct",
        "final_net_saved_pct",
        "break_even_frame",
    ]
    out = [keep]
    with OVERHEAD_SUMMARY_CSV.open("r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("run_name") not in FIGURE7_RUN_NAMES:
                continue
            if row.get("cache_mode") not in ("cold_start", "warm_start"):
                continue
            out.append([row.get(k, "") for k in keep])
    return out


def build_figure8_sheets() -> tuple[list[list[Any]], list[list[Any]]]:
    keep_cols = [
        "clip_id",
        "game",
        "content_style",
        "avg_masked_area_pct",
        "bsp_pct",
        "ref_ratio",
    ]
    rows = []
    with GENERALITY_SUMMARY_CSV.open("r", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    data_long = [[
        "experiment",
        "item_number",
        *keep_cols,
        "source_file",
    ]]
    for row in rows:
        data_long.append(
            [
                "exp4",
                "figure8",
                row.get("clip_id", ""),
                row.get("game", ""),
                row.get("content_style", ""),
                float(row.get("avg_masked_area_pct", "0") or 0),
                float(row.get("bsp_pct", "0") or 0),
                float(row.get("ref_ratio", "0") or 0),
                str(GENERALITY_SUMMARY_CSV),
            ]
        )
    paper_view = [keep_cols]
    for row in rows:
        paper_view.append([row.get(c, "") for c in keep_cols])
    return data_long, paper_view


def len_prefixed_sizes(path: Path) -> list[int]:
    if not path.exists():
        return []
    data = path.read_bytes()
    out = []
    off = 0
    while off + 4 <= len(data):
        n = int.from_bytes(data[off : off + 4], "little")
        off += 4
        out.append(4 + n)
        off += n
    return out


def build_figure9_sheets() -> tuple[list[list[Any]], list[list[Any]]]:
    gop_savings_by_game: dict[str, list[float]] = defaultdict(list)
    with GENERALITY_SUMMARY_CSV.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        run_dir = Path(row.get("run_dir", ""))
        baseline_dir = Path(row.get("baseline_run_dir", ""))
        per_csv = run_dir / "per_frame_metrics.csv"
        base_csv = baseline_dir / "per_frame_metrics.csv"
        if not per_csv.exists() or not base_csv.exists():
            continue
        msk1_sizes = len_prefixed_sizes(run_dir / "msk1_payloads.bin")
        with base_csv.open("r", newline="") as fb, per_csv.open("r", newline="") as fr:
            rb = list(csv.DictReader(fb))
            rr = list(csv.DictReader(fr))
        n = min(len(rb), len(rr))
        for start in range(0, n, 60):
            stop = min(n, start + 60)
            base_sum = 0.0
            resp_sum = 0.0
            for idx in range(start, stop):
                base_sum += float(rb[idx].get("masked_bytes", "0") or 0)
                resp_sum += float(rr[idx].get("masked_bytes", "0") or 0)
                if idx < len(msk1_sizes):
                    resp_sum += float(msk1_sizes[idx])
            if base_sum > 1e-9:
                save = (1.0 - resp_sum / base_sum) * 100.0
                if -50.0 <= save <= 50.0:
                    gop_savings_by_game[row.get("game", "unknown")].append(save)

    data_long = [[
        "experiment",
        "item_number",
        "game",
        "gop_savings_pct",
        "cdf",
        "source_file",
    ]]
    for game in sorted(gop_savings_by_game):
        vals = sorted(gop_savings_by_game[game])
        if not vals:
            continue
        for i, val in enumerate(vals, start=1):
            data_long.append(
                [
                    "exp4",
                    "figure9",
                    game,
                    val,
                    i / len(vals),
                    str(GENERALITY_SUMMARY_CSV),
                ]
            )
    paper_view = [["game", "gop_savings_pct", "cdf"]]
    for row in data_long[1:]:
        paper_view.append([row[2], row[3], row[4]])
    return data_long, paper_view


def build_figure9_run_index() -> list[list[Any]]:
    rows = [["clip_id", "game", "run_dir", "baseline_run_dir", "respawn_per_frame_csv", "baseline_per_frame_csv", "msk1_payloads_bin"]]
    with GENERALITY_SUMMARY_CSV.open("r", newline="") as f:
        for row in csv.DictReader(f):
            run_dir = Path(row.get("run_dir", ""))
            baseline_dir = Path(row.get("baseline_run_dir", ""))
            rows.append(
                [
                    row.get("clip_id", ""),
                    row.get("game", ""),
                    str(run_dir),
                    str(baseline_dir),
                    str(run_dir / "per_frame_metrics.csv"),
                    str(baseline_dir / "per_frame_metrics.csv"),
                    str(run_dir / "msk1_payloads.bin"),
                ]
            )
    return rows


def write_readme(
    item_dir: Path,
    *,
    experiment: str,
    item_number: str,
    item_type: str,
    source_paths: list[Path],
    note: str = "",
) -> None:
    lines = [
        f"# {experiment.upper()} {item_type.title()} {item_number}",
        "",
        "## Contents",
        "- `paper_asset*`: copied asset(s) used in the Results section",
        "- `data.xlsx`: tidy data workbook with `data_long` and `paper_view` sheets",
        "- `data_long.csv` and `paper_view.csv`: CSV exports used to build the workbook",
        "- `source_data_*.csv`: copied raw CSV inputs (when source is CSV)",
        "- `README.md`: provenance and notes",
        "",
        "## Source files",
    ]
    for p in source_paths:
        lines.append(f"- `{p}`")
    if note:
        lines += ["", "## Notes", f"- {note}"]
    (item_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    items = [
        {
            "experiment": "exp1",
            "item_number": "table7",
            "item_type": "table",
            "assets": [RESULT_TEX],
            "builder": "table7",
            "sources": [RESULT_TEX, RD_POINTS_CSV],
            "note": "Values parsed from the inline table in Result.tex and cross-check source listed as rd_suite_points.csv.",
        },
        {
            "experiment": "exp1",
            "item_number": "table8",
            "item_type": "table",
            "assets": [RD_POINTS_CSV],
            "builder": "table8",
            "sources": [RD_POINTS_CSV],
            "note": "Reconstructed from rd_suite_points.csv because the included source Tiehang/figure/exp1/rd-quality-fixed_rate is missing in the repo.",
        },
        {
            "experiment": "exp2",
            "item_number": "table9",
            "item_type": "table",
            "assets": [REPO_ROOT / "record/RESPAWN2026/exp3/partial_warm_fm6_table.tex"],
            "builder": "table_tex",
            "sources": [REPO_ROOT / "record/RESPAWN2026/exp3/partial_warm_fm6_table.tex"],
            "note": "",
        },
        {
            "experiment": "exp2",
            "item_number": "figure7",
            "item_type": "figure",
            "assets": [
                REPO_ROOT / "record/RESPAWN2026/exp3/fc5_00_23p0_cumulative_net_bytes.png",
                REPO_ROOT / "record/RESPAWN2026/exp3/fm6_04_23p0_cumulative_net_bytes.png",
            ],
            "builder": "figure7",
            "sources": [OVERHEAD_SUMMARY_CSV],
            "note": "data_long uses overhead_summary rows for fc5_00_23p0 and fm6_04_23p0 sensitivity settings.",
        },
        {
            "experiment": "exp3",
            "item_number": "table10",
            "item_type": "table",
            "assets": [REPO_ROOT / "record/RESPAWN2026/exp4/exp4_feather_table.tex"],
            "builder": "table_tex",
            "sources": [REPO_ROOT / "record/RESPAWN2026/exp4/exp4_feather_table.tex"],
            "note": "",
        },
        {
            "experiment": "exp3",
            "item_number": "table11",
            "item_type": "table",
            "assets": [REPO_ROOT / "record/RESPAWN2026/exp4/exp4_fill_strategy_table.tex"],
            "builder": "table_tex",
            "sources": [REPO_ROOT / "record/RESPAWN2026/exp4/exp4_fill_strategy_table.tex"],
            "note": "",
        },
        {
            "experiment": "exp3",
            "item_number": "table12",
            "item_type": "table",
            "assets": [REPO_ROOT / "record/RESPAWN2026/exp4/exp4_matching_strategy_table.tex"],
            "builder": "table_tex",
            "sources": [REPO_ROOT / "record/RESPAWN2026/exp4/exp4_matching_strategy_table.tex"],
            "note": "",
        },
        {
            "experiment": "exp3",
            "item_number": "table13",
            "item_type": "table",
            "assets": [REPO_ROOT / "record/RESPAWN2026/exp4/exp4_mario_path_ablation_table.tex"],
            "builder": "table_tex",
            "sources": [REPO_ROOT / "record/RESPAWN2026/exp4/exp4_mario_path_ablation_table.tex"],
            "note": "",
        },
        {
            "experiment": "exp4",
            "item_number": "table14",
            "item_type": "table",
            "assets": [REPO_ROOT / "record/RESPAWN2026/exp5/bsp_vs_ref_ratio.tex"],
            "builder": "table_tex",
            "sources": [REPO_ROOT / "record/RESPAWN2026/exp5/bsp_vs_ref_ratio.tex"],
            "note": "",
        },
        {
            "experiment": "exp4",
            "item_number": "figure8",
            "item_type": "figure",
            "assets": [REPO_ROOT / "record/RESPAWN2026/exp5/bsp_vs_masked_area.png"],
            "builder": "figure8",
            "sources": [GENERALITY_SUMMARY_CSV],
            "note": "data_long contains clip-level scatter points used for BSP vs masked-area plotting.",
        },
        {
            "experiment": "exp4",
            "item_number": "figure9",
            "item_type": "figure",
            "assets": [REPO_ROOT / "record/RESPAWN2026/exp5/per_gop_savings_cdf.png"],
            "builder": "figure9",
            "sources": [GENERALITY_SUMMARY_CSV],
            "note": "data_long contains reconstructed per-game CDF points from per-frame metrics and MSK1 payload sizes.",
        },
    ]

    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_rows = [[
        "experiment",
        "item_number",
        "item_type",
        "folder",
        "has_assets",
        "has_xlsx",
        "has_csv_exports",
        "has_readme",
        "status",
    ]]

    table7_rows = parse_table7_rows(RESULT_TEX)
    rate_cols = [8, 12, 16, 20, 24, "All"]

    for item in items:
        item_dir = TARGET_ROOT / item["experiment"] / item["item_number"]
        item_dir.mkdir(parents=True, exist_ok=True)
        copied_assets = copy_assets(item["assets"], item_dir)
        copied_source_csvs = copy_source_csvs(item["sources"], item_dir)

        if item["builder"] == "table7":
            data_long = [[
                "experiment",
                "item_number",
                "game",
                "rate_mbps",
                "metric",
                "value",
                "unit",
                "source_file",
            ]]
            paper_view = [["Game", "8", "12", "16", "20", "24", "All"]]
            for row in table7_rows:
                game = row[0]
                paper_view.append(row)
                for idx, col in enumerate(rate_cols, start=1):
                    raw = row[idx]
                    val = parse_float_maybe(raw)
                    data_long.append(
                        [
                            "exp1",
                            "table7",
                            game,
                            col,
                            "net_delivered_saving_pct",
                            val if val is not None else "",
                            "percent",
                            str(RESULT_TEX),
                        ]
                    )
        elif item["builder"] == "table8":
            data_long, paper_view = build_table8_from_rd_points()
        elif item["builder"] == "table_tex":
            tex_path = item["assets"][0]
            data_long, paper_view = table_tex_to_sheets(
                item["experiment"], item["item_number"], tex_path
            )
        elif item["builder"] == "figure7":
            data_long, paper_view = build_figure7_sheets()
        elif item["builder"] == "figure8":
            data_long, paper_view = build_figure8_sheets()
        elif item["builder"] == "figure9":
            data_long, paper_view = build_figure9_sheets()
        else:
            raise RuntimeError(f"Unknown builder: {item['builder']}")

        write_xlsx(item_dir / "data.xlsx", {"data_long": data_long, "paper_view": paper_view})
        write_csv_rows(item_dir / "data_long.csv", data_long)
        write_csv_rows(item_dir / "paper_view.csv", paper_view)

        if item["builder"] == "figure7":
            write_csv_rows(item_dir / "run_data_overhead_subset.csv", build_figure7_subset_rows())
        if item["builder"] == "figure9":
            write_csv_rows(item_dir / "run_data_index.csv", build_figure9_run_index())

        write_readme(
            item_dir,
            experiment=item["experiment"],
            item_number=item["item_number"],
            item_type=item["item_type"],
            source_paths=item["sources"],
            note=item.get("note", ""),
        )
        has_assets = "1" if copied_assets else "0"
        has_xlsx = "1" if (item_dir / "data.xlsx").exists() else "0"
        has_csv_exports = "1" if (item_dir / "data_long.csv").exists() and (item_dir / "paper_view.csv").exists() else "0"
        has_readme = "1" if (item_dir / "README.md").exists() else "0"
        status = "ok" if has_assets == "1" and has_xlsx == "1" and has_csv_exports == "1" and has_readme == "1" else "incomplete"
        manifest_rows.append(
            [
                item["experiment"],
                item["item_number"],
                item["item_type"],
                str(item_dir),
                has_assets,
                has_xlsx,
                has_csv_exports,
                has_readme,
                status,
            ]
        )

    manifest_path = TARGET_ROOT / "results_artifacts_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(manifest_rows)
    print(str(TARGET_ROOT))
    print(str(manifest_path))


if __name__ == "__main__":
    main()
