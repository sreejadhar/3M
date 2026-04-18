"""
Excel report generation for DataNanite insight export.

build_excel_report(results, title) → bytes (.xlsx)

Workbook layout:
  Sheet 1 : Dashboard  — title, generated time, per-result KPI summary, metric tiles
  Sheet N : [Result N] — banner, data table with alternating rows, auto-chart below data
  Pivot_N : cross-tab  — created automatically when data has 2 cat cols + 1 numeric col

Chart type is inferred from the data shape (mirrors the JS detectChartConfig logic).
"""
from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openpyxl import Workbook
from openpyxl.chart import BarChart, DoughnutChart, LineChart, PieChart, Reference, ScatterChart, Series
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# ── Brand palette ──────────────────────────────────────────────────────────────
_DARK       = "1E293B"
_WHITE      = "F8FAFC"
_ALT        = "F1F5F9"
_ACCENT     = "6366F1"
_ACCENT_BG  = "EEF2FF"
_MUTE       = "64748B"
_MAX_ROWS   = 10_000


# ── Style helpers ──────────────────────────────────────────────────────────────

def _fill(hex_color: str) -> PatternFill:
    return PatternFill("solid", fgColor=hex_color)

def _border() -> Border:
    s = Side(style="thin", color="CBD5E1")
    return Border(left=s, right=s, top=s, bottom=s)

def _style_header_row(ws, row: int, n_cols: int):
    for col in range(1, n_cols + 1):
        c = ws.cell(row, col)
        c.fill  = _fill(_DARK)
        c.font  = Font(bold=True, color=_WHITE, size=10)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = _border()
    ws.row_dimensions[row].height = 20

def _col_width(col_name: str, rows: List[Dict]) -> float:
    sample = [str(r.get(col_name, "")) for r in rows[:20]]
    return min(max(len(col_name) + 2, max((len(v) for v in sample), default=0) + 2, 8), 42)

def _safe_name(text: str, idx: int) -> str:
    name = re.sub(r'[\\/*?:\[\]]', "", text).strip()[:28]
    return name or f"Result {idx + 1}"


# ── Column classification ──────────────────────────────────────────────────────

def _is_numeric(val) -> bool:
    if val is None:
        return False
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False

def _num_cols(cols: List[str], rows: List[Dict]) -> List[str]:
    sample = rows[:8]
    return [c for c in cols
            if sample and all(_is_numeric(r.get(c)) for r in sample if r.get(c) is not None)]

def _cat_cols(cols: List[str], num: List[str]) -> List[str]:
    return [c for c in cols if c not in num]


# ── Chart type detection ───────────────────────────────────────────────────────

def _detect_chart(cols: List[str], rows: List[Dict]) -> Optional[Dict]:
    if not rows or len(cols) < 2:
        return None

    num = _num_cols(cols, rows)
    cat = _cat_cols(cols, num)
    lbl = cat[0] if cat else None

    if lbl:
        is_id = bool(re.search(r"(_id|_key|_sk|_sku|_code|_uuid|_no|_num|_ref|_pk)$", lbl, re.I)
                     or re.fullmatch(r"id|key|pk", lbl, re.I))
        if is_id:
            return None

    if len(num) == 2 and not lbl and len(rows) >= 10:
        return {"type": "scatter", "x": num[0], "y": num[1]}

    if not lbl or not num:
        return None

    ll = lbl.lower()

    if len(rows) == 1 and len(num) >= 2:
        return None  # KPI tiles only

    if re.search(r"date|month|year|week|day|quarter|period|time|fiscal|yr|qtr|wk", ll):
        return {"type": "line", "lbl": lbl, "num": num[:5]}

    if re.search(r"bucket|bin|band|range|bracket|tier|cohort|decile|quartile|percentile", ll) and len(num) == 1:
        return {"type": "vbar", "lbl": lbl, "num": num}

    all_names = " ".join(cols).lower()
    if re.search(r"variance|delta|bridge|impact|contrib|change|prior|current|opening|closing|movement", all_names) and len(rows) <= 20:
        return {"type": "vbar", "lbl": lbl, "num": num[:1]}

    if len(num) >= 2 and len(rows) <= 30:
        is_comp = bool(re.search(r"channel|segment|region|category|brand|product|division|dept|territory|country|market", ll))
        return {"type": "stacked" if is_comp else "grouped", "lbl": lbl, "num": num[:6]}

    is_share = bool(re.search(r"channel|segment|region|category|brand|type|division", ll))
    if is_share and len(rows) <= 8 and len(num) == 1:
        return {"type": "doughnut", "lbl": lbl, "num": num}

    if len(rows) <= 8 and len(num) == 1:
        return {"type": "vbar", "lbl": lbl, "num": num}

    if len(rows) <= 80 and len(num) == 1:
        return {"type": "hbar", "lbl": lbl, "num": num}

    return None


# ── Chart rendering ────────────────────────────────────────────────────────────

def _add_chart(ws, cols: List[str], rows: List[Dict], cfg: Dict,
               data_row: int, n_rows: int, anchor: str):
    ctype = cfg["type"]

    if ctype == "scatter":
        xi = cols.index(cfg["x"]) + 1
        yi = cols.index(cfg["y"]) + 1
        chart = ScatterChart()
        chart.style = 10
        xd = Reference(ws, min_col=xi, min_row=data_row, max_row=data_row + n_rows - 1)
        yd = Reference(ws, min_col=yi, min_row=data_row - 1, max_row=data_row + n_rows - 1)
        chart.series.append(Series(yd, xd, title=cfg["y"]))
        chart.width, chart.height = 18, 10
        ws.add_chart(chart, anchor)
        return

    lbl = cfg.get("lbl")
    num = cfg.get("num", [])
    if not lbl or not num:
        return

    li = cols.index(lbl) + 1
    ni_min = min(cols.index(c) + 1 for c in num)
    ni_max = max(cols.index(c) + 1 for c in num)

    cats = Reference(ws, min_col=li, min_row=data_row, max_row=data_row + n_rows - 1)
    data = Reference(ws, min_col=ni_min, max_col=ni_max,
                     min_row=data_row - 1, max_row=data_row + n_rows - 1)

    if ctype == "line":
        chart = LineChart()
        chart.grouping = "standard"
    elif ctype == "hbar":
        chart = BarChart()
        chart.type = "bar"
        chart.grouping = "clustered"
    elif ctype == "stacked":
        chart = BarChart()
        chart.type = "col"
        chart.grouping = "stacked"
    elif ctype == "grouped":
        chart = BarChart()
        chart.type = "col"
        chart.grouping = "clustered"
    elif ctype == "doughnut":
        chart = DoughnutChart()
        chart.style = 10
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)
        chart.width, chart.height = 18, 12
        ws.add_chart(chart, anchor)
        return
    else:  # vbar, histogram, waterfall, default
        chart = BarChart()
        chart.type = "col"
        chart.grouping = "clustered"

    chart.style = 10
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    chart.width, chart.height = 20, 12
    ws.add_chart(chart, anchor)


# ── Pivot sheet ────────────────────────────────────────────────────────────────

def _maybe_pivot(wb: Workbook, cols: List[str], rows: List[Dict], base_name: str) -> bool:
    num = _num_cols(cols, rows)
    cat = _cat_cols(cols, num)
    if len(cat) != 2 or len(num) != 1:
        return False

    row_dim, col_dim, measure = cat[0], cat[1], num[0]
    col_vals = sorted({str(r.get(col_dim, "")) for r in rows})
    if len(col_vals) > 15:
        return False

    row_vals = sorted({str(r.get(row_dim, "")) for r in rows})

    agg: Dict[Tuple, float] = {}
    for r in rows:
        key = (str(r.get(row_dim, "")), str(r.get(col_dim, "")))
        try:
            agg[key] = agg.get(key, 0.0) + float(r.get(measure) or 0)
        except (TypeError, ValueError):
            pass

    ws = wb.create_sheet(_safe_name(f"Pivot_{base_name}", 0))

    total_col = len(col_vals) + 2

    # Header row
    ws.cell(1, 1, row_dim)
    _style_header_row(ws, 1, total_col)
    for j, cv in enumerate(col_vals, 2):
        ws.cell(1, j, cv).alignment = Alignment(horizontal="center")
    ws.cell(1, total_col, "Total")

    # Data rows
    for i, rv in enumerate(row_vals, 2):
        alt = (i % 2 == 0)
        fill = _fill(_ALT) if alt else PatternFill()
        c = ws.cell(i, 1, rv)
        c.fill = fill
        c.border = _border()
        c.font = Font(bold=True)

        row_total = 0.0
        for j, cv in enumerate(col_vals, 2):
            val = agg.get((rv, cv), 0.0)
            row_total += val
            cell = ws.cell(i, j, round(val, 2))
            cell.fill = fill
            cell.border = _border()
            cell.alignment = Alignment(horizontal="right")
        tot = ws.cell(i, total_col, round(row_total, 2))
        tot.fill = fill
        tot.border = _border()
        tot.font = Font(bold=True)
        tot.alignment = Alignment(horizontal="right")

    # Grand total row
    gt_row = len(row_vals) + 2
    ws.cell(gt_row, 1, "Grand Total")
    _style_header_row(ws, gt_row, total_col)
    grand = 0.0
    for j, cv in enumerate(col_vals, 2):
        col_total = sum(agg.get((rv, cv), 0.0) for rv in row_vals)
        grand += col_total
        c = ws.cell(gt_row, j, round(col_total, 2))
        c.fill = _fill(_DARK)
        c.font = Font(bold=True, color=_WHITE)
        c.border = _border()
        c.alignment = Alignment(horizontal="right")
    tc = ws.cell(gt_row, total_col, round(grand, 2))
    tc.fill = _fill(_DARK)
    tc.font = Font(bold=True, color=_WHITE)
    tc.border = _border()
    tc.alignment = Alignment(horizontal="right")

    ws.column_dimensions["A"].width = max(len(row_dim) + 2, max((len(v) for v in row_vals), default=0) + 2, 14)
    for j, cv in enumerate(col_vals, 2):
        ws.column_dimensions[get_column_letter(j)].width = max(len(cv) + 2, 10)
    ws.column_dimensions[get_column_letter(total_col)].width = 12

    ws.freeze_panes = "B2"
    ws.sheet_view.showGridLines = False
    return True


# ── Data sheet ─────────────────────────────────────────────────────────────────

def _write_data_sheet(wb: Workbook, result: Dict, idx: int) -> Dict:
    desc   = result.get("description") or f"Query {idx + 1}"
    cols   = result.get("columns") or []
    rows   = (result.get("rows") or [])[:_MAX_ROWS]
    if not cols or not rows:
        return {"name": desc, "rows": 0, "cols": 0, "kpis": []}

    name = _safe_name(desc, idx)
    ws   = wb.create_sheet(name)
    nc   = len(cols)
    nr   = len(rows)
    nums = set(_num_cols(cols, rows))

    # Banner row
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=nc)
    b = ws.cell(1, 1, desc)
    b.fill = _fill(_ACCENT)
    b.font = Font(bold=True, color=_WHITE, size=12)
    b.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 24

    # Header row
    for j, col in enumerate(cols, 1):
        ws.cell(2, j, col)
    _style_header_row(ws, 2, nc)

    # Data rows
    for i, row in enumerate(rows, 3):
        alt  = (i % 2 == 0)
        fill = _fill(_ALT) if alt else PatternFill()
        for j, col in enumerate(cols, 1):
            val = row.get(col)
            if val is not None and col in nums:
                try:
                    fv = float(val)
                    val = int(fv) if fv == int(fv) else fv
                except (TypeError, ValueError):
                    pass
            c = ws.cell(i, j, val)
            c.fill   = fill
            c.border = _border()
            c.alignment = Alignment(
                horizontal="right" if col in nums else "left",
                vertical="center"
            )

    # Column widths
    for j, col in enumerate(cols, 1):
        ws.column_dimensions[get_column_letter(j)].width = _col_width(col, rows)

    ws.freeze_panes = "A3"
    ws.sheet_view.showGridLines = False

    # Chart (anchored below data with a gap row)
    cfg = _detect_chart(cols, rows)
    if cfg:
        _add_chart(ws, cols, rows, cfg, data_row=3, n_rows=nr, anchor=f"A{nr + 5}")

    # Pivot sheet when shape allows
    _maybe_pivot(wb, cols, rows, name)

    # KPI summary for dashboard
    kpis = []
    for nc_name in list(nums)[:4]:
        try:
            vals = [float(r.get(nc_name) or 0) for r in rows]
            kpis.append({
                "col": nc_name,
                "sum": round(sum(vals), 2),
                "avg": round(sum(vals) / len(vals), 2) if vals else 0,
            })
        except (TypeError, ValueError):
            pass

    return {"name": name, "desc": desc, "rows": nr, "cols": len(cols), "kpis": kpis}


# ── Dashboard sheet ────────────────────────────────────────────────────────────

def _write_dashboard(wb: Workbook, meta_list: List[Dict], title: str):
    ws = wb.create_sheet("Dashboard", 0)

    # Title
    ws.merge_cells("A1:H1")
    t = ws.cell(1, 1, title)
    t.fill = _fill(_DARK)
    t.font = Font(bold=True, color=_WHITE, size=16)
    t.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 36

    # Sub-title
    ws.merge_cells("A2:H2")
    s = ws.cell(2, 1, f"Generated: {datetime.now().strftime('%Y-%m-%d  %H:%M')}   ·   {len(meta_list)} result set(s)")
    s.fill = _fill("F8FAFC")
    s.font = Font(italic=True, color=_MUTE, size=10)
    s.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[2].height = 18

    # Summary table header
    hdrs = ["#", "Result", "Rows", "Columns", "Top Metric", "Sum", "Avg", "Sheet"]
    for j, h in enumerate(hdrs, 1):
        ws.cell(4, j, h)
    _style_header_row(ws, 4, len(hdrs))

    for i, m in enumerate(meta_list, 5):
        alt  = (i % 2 == 0)
        fill = _fill(_ALT) if alt else PatternFill()
        vals = [
            i - 4,
            m.get("desc", m.get("name", "")),
            m.get("rows", 0),
            m.get("cols", 0),
            m["kpis"][0]["col"] if m.get("kpis") else "—",
            m["kpis"][0]["sum"] if m.get("kpis") else "—",
            m["kpis"][0]["avg"] if m.get("kpis") else "—",
            m.get("name", ""),
        ]
        right_cols = {3, 4, 6, 7}
        for j, v in enumerate(vals, 1):
            c = ws.cell(i, j, v)
            c.fill   = fill
            c.border = _border()
            c.alignment = Alignment(horizontal="right" if j in right_cols else "left", vertical="center")

        # Hyperlink to tab
        link_cell = ws.cell(i, 8)
        link_cell.hyperlink = f"#'{m['name']}'!A1"
        link_cell.font = Font(color=_ACCENT, underline="single")

    # KPI tile section
    kpi_hdr_row = 6 + len(meta_list)
    ws.merge_cells(start_row=kpi_hdr_row, start_column=1, end_row=kpi_hdr_row, end_column=8)
    kh = ws.cell(kpi_hdr_row, 1, "KEY METRICS")
    kh.fill = _fill(_DARK)
    kh.font = Font(bold=True, color=_WHITE, size=10)
    kh.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[kpi_hdr_row].height = 18

    kpi_row  = kpi_hdr_row + 1
    tile_col = 1
    for m in meta_list:
        for kpi in m.get("kpis", [])[:2]:
            if tile_col > 8:
                kpi_row  += 4
                tile_col  = 1
            label = ws.cell(kpi_row, tile_col, kpi["col"])
            label.font      = Font(bold=True, size=9, color=_MUTE)
            label.alignment = Alignment(horizontal="center")

            val = ws.cell(kpi_row + 1, tile_col, kpi["sum"])
            val.font      = Font(bold=True, size=14, color=_ACCENT)
            val.fill      = _fill(_ACCENT_BG)
            val.border    = _border()
            val.alignment = Alignment(horizontal="center")

            ws.column_dimensions[get_column_letter(tile_col)].width = max(
                ws.column_dimensions[get_column_letter(tile_col)].width or 0,
                len(str(kpi["col"])) + 4,
                14
            )
            tile_col += 1

    # Fixed widths for summary table columns
    for col_letter, width in zip("ABCDEFGH", [5, 42, 10, 10, 24, 14, 14, 24]):
        if ws.column_dimensions[col_letter].width < width:
            ws.column_dimensions[col_letter].width = width

    ws.sheet_view.showGridLines = False


# ── Public entry point ─────────────────────────────────────────────────────────

def _normalise_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert list-style rows [[v1,v2,...]] → dict rows [{col: v,...}]."""
    cols = result.get("columns") or []
    rows = result.get("rows") or []
    if rows and isinstance(rows[0], (list, tuple)):
        rows = [dict(zip(cols, row)) for row in rows]
    return {**result, "columns": cols, "rows": rows}


def build_excel_report(results: List[Dict[str, Any]], title: str = "DataNanite Insight Report") -> bytes:
    """
    Build a multi-tab Excel workbook from query result dicts.
    Each result: {description, columns: [str], rows: [dict | list], ...}
    Returns raw .xlsx bytes.
    """
    wb   = Workbook()
    stub = wb.active
    if stub is not None:
        wb.remove(stub)

    valid = [_normalise_result(r) for r in results if r.get("columns") and r.get("rows")]
    meta_list = [_write_data_sheet(wb, r, i) for i, r in enumerate(valid)]

    if meta_list:
        _write_dashboard(wb, meta_list, title)
        wb.move_sheet("Dashboard", offset=-(len(wb.sheetnames) - 1))

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()
