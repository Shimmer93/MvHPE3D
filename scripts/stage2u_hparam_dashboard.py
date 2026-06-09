#!/usr/bin/env python
"""Build and serve a small Stage2U hparam results database.

The script intentionally uses only the Python standard library so it can run on
the remote training server without extra dashboard dependencies.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


DEFAULT_RUN_ROOT = Path("outputs/stage2_unified")
DEFAULT_DB_PATH = Path("outputs/stage2_unified/h36m_hparam_tuning/stage2u_hparams.sqlite")

IMPORTANT_HPARAMS = (
    "body_start_epoch",
    "root_lr",
    "body_lr",
    "body_delta_scale",
    "gate_bias",
    "gate_sparsity_weight",
    "camera_joint_weight",
    "camera_joint_loss_weights",
    "camera_joint_loss_weights_end_epoch",
    "pa_joint_weight",
    "pa_joint_loss_weights",
    "pa_joint_loss_weights_end_epoch",
    "gt_projection_weight",
    "preserve_joint_weight",
    "do_no_harm_weight",
    "global_orient_delta_weight",
    "body_delta_weight",
    "compose_body_update",
    "use_body_image_joint_feature",
    "use_body_image_mask_feature",
    "use_body_evidence_weighted_pose_fusion",
    "body_evidence_weighted_pose_joint_policy",
    "evidence_weighted_pose_project_so3",
    "body_extra_candidate_count",
    "detach_base_update_in_final_loss",
    "max_epochs",
    "ablation_base_run",
    "ablation_group",
    "ablation_target",
    "ablation_index",
    "disable_root_input_global_orient",
    "disable_root_input_transl",
    "disable_root_measurement_residual",
    "disable_root_measurement_confidence",
    "disable_root_measurement_valid",
    "disable_root_image_size",
    "disable_body_stage2_pose",
    "disable_body_input_pose_mean",
    "disable_body_input_pose_dispersion",
    "disable_body_input_betas",
    "disable_body_image_residual",
    "disable_body_image_confidence",
    "disable_body_image_valid",
    "disable_body_image_size",
)


def iter_json_objects(text: str):
    decoder = json.JSONDecoder()
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            return
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        index = start + end
        yield obj


def run_number(name: str) -> int | None:
    if not name.startswith("run"):
        return None
    digits = []
    for char in name[3:]:
        if char.isdigit():
            digits.append(char)
        else:
            break
    if not digits:
        return None
    return int("".join(digits))


def load_summary_run(run_dir: Path) -> tuple[dict, list[dict]] | None:
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(errors="ignore"))
    args = data.get("args") or {}
    rows = []
    for item in data.get("history", []):
        val = item.get("val") or {}
        row = epoch_row(item.get("epoch"), val)
        if row is not None:
            rows.append(row)
    return {"args": args, "source": str(path), "pipeline_note": data.get("pipeline_note")}, rows


def load_log_run(log_path: Path) -> tuple[dict, list[dict]] | None:
    text = log_path.read_text(errors="ignore")
    rows = []
    args = {}
    pipeline_note = None
    for obj in iter_json_objects(text):
        if not isinstance(obj, dict):
            continue
        if isinstance(obj.get("args"), dict):
            args.update(obj["args"])
        if obj.get("pipeline_note"):
            pipeline_note = obj.get("pipeline_note")
        if isinstance(obj.get("val"), dict):
            row = epoch_row(obj.get("epoch"), obj["val"])
            if row is not None:
                rows.append(row)
    return {"args": args, "source": str(log_path), "pipeline_note": pipeline_note}, rows


def load_sidecar_metadata(run_dir: Path | None) -> dict:
    if run_dir is None:
        return {}
    path = run_dir / "hparam_metadata.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(errors="ignore"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    args = data.get("args") if isinstance(data.get("args"), dict) else {}
    return {
        "args": args,
        "pipeline_note": data.get("pipeline_note"),
        "description": data.get("description"),
    }


def merge_sidecar_metadata(meta: dict, sidecar: dict) -> dict:
    if not sidecar:
        return meta
    merged_args = dict(sidecar.get("args") or {})
    merged_args.update(meta.get("args") or {})
    result = dict(meta)
    result["args"] = merged_args
    if not result.get("pipeline_note") and sidecar.get("pipeline_note"):
        result["pipeline_note"] = sidecar.get("pipeline_note")
    return result


def epoch_row(epoch, val: dict) -> dict | None:
    mpjpe = val.get("corrected/camera_mpjpe_mm")
    pa = val.get("corrected/camera_pa_mpjpe_mm")
    if mpjpe is None or pa is None:
        return None
    try:
        mpjpe = float(mpjpe)
        pa = float(pa)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(mpjpe) or not math.isfinite(pa):
        return None
    return {
        "epoch": int(epoch) if epoch is not None else None,
        "mpjpe": mpjpe,
        "pa_mpjpe": pa,
        "stage2_mpjpe": optional_float(val.get("stage2/camera_mpjpe_mm")),
        "stage2_pa_mpjpe": optional_float(val.get("stage2/camera_pa_mpjpe_mm")),
        "root_mpjpe": optional_float(val.get("root/camera_mpjpe_mm")),
        "gate_mean": optional_float(val.get("adapter/gate_mean")),
        "update_abs_mean": optional_float(val.get("adapter/update_abs_mean")),
        "base_mpjpe": optional_float(val.get("base/camera_mpjpe_mm")),
        "base_pa_mpjpe": optional_float(val.get("base/camera_pa_mpjpe_mm")),
        "body_enabled": optional_float(val.get("body_enabled")),
        "raw": val,
    }


def optional_float(value):
    try:
        if value is None:
            return None
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def discover_runs(run_root: Path) -> list[tuple[str, Path | None, Path | None]]:
    names: dict[str, tuple[Path | None, Path | None]] = {}
    if not run_root.exists():
        return []
    scan_roots = [run_root]
    if run_root.name == "stage2_unified":
        scan_roots = [
            run_root / "h36m_root_orientation_fair_research",
            run_root / "h36m_hparam_tuning",
        ]
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("run*"):
            if any(part.startswith("archive_") for part in path.parts):
                continue
            if path.is_dir():
                current = names.get(path.name, (None, None))
                names[path.name] = (path, current[1])
            elif path.is_file() and path.suffix == ".log":
                current = names.get(path.stem, (None, None))
                names[path.stem] = (current[0], path)
    return [(name, paths[0], paths[1]) for name, paths in sorted(names.items(), key=natural_key)]


def natural_key(item):
    if isinstance(item, tuple):
        item = item[0]
    parts = []
    token = ""
    numeric = False
    for char in str(item):
        is_digit = char.isdigit()
        if token and is_digit != numeric:
            parts.append(int(token) if numeric else token)
            token = ""
        token += char
        numeric = is_digit
    if token:
        parts.append(int(token) if numeric else token)
    return parts


def classify_run(name: str) -> str:
    if "smoke" in name:
        return "smoke"
    if "ablate" in name or "ablation" in name:
        return "run207_ablation"
    if "hparam" in name or name.startswith("run2"):
        return "hparam"
    if "weightedpose" in name:
        return "image_weighted_pose"
    if "extraimg" in name:
        return "extra_image_branch"
    if "composebody" in name:
        return "compose_body"
    if "limb2curr30_std" in name:
        return "run125_family"
    return "research_legacy"


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        drop table if exists run_hparams;
        drop table if exists epochs;
        drop table if exists runs;
        create table runs (
            run_name text primary key,
            run_number integer,
            family text,
            source text,
            output_dir text,
            epochs integer,
            latest_epoch integer,
            latest_mpjpe real,
            latest_pa_mpjpe real,
            best_mpjpe_epoch integer,
            best_mpjpe real,
            pa_at_best_mpjpe real,
            best_pa_epoch integer,
            best_pa_mpjpe real,
            mpjpe_at_best_pa real,
            hparam_json text,
            pipeline_note text
        );
        create table epochs (
            run_name text,
            epoch integer,
            mpjpe real,
            pa_mpjpe real,
            stage2_mpjpe real,
            stage2_pa_mpjpe real,
            root_mpjpe real,
            gate_mean real,
            update_abs_mean real,
            base_mpjpe real,
            base_pa_mpjpe real,
            body_enabled real,
            raw_json text,
            primary key (run_name, epoch)
        );
        create table run_hparams (
            run_name text,
            key text,
            value text,
            numeric_value real,
            primary key (run_name, key)
        );
        create index idx_epochs_metric on epochs (epoch, mpjpe, pa_mpjpe);
        create index idx_hparams_key on run_hparams (key, value);
        """
    )


def build_db(run_root: Path, db_path: Path, include_smoke: bool) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        create_schema(conn)
        for name, run_dir, log_path in discover_runs(run_root):
            candidate_paths = [path for path in (run_dir, log_path) if path is not None]
            if any(part.startswith("archive_") for path in candidate_paths for part in path.parts):
                continue
            if not include_smoke and "smoke" in name:
                continue
            meta_rows = None
            sidecar = load_sidecar_metadata(run_dir)
            if run_dir is not None:
                meta_rows = load_summary_run(run_dir)
            if meta_rows is None and log_path is not None:
                meta_rows = load_log_run(log_path)
            if meta_rows is None:
                continue
            meta, rows = meta_rows
            meta = merge_sidecar_metadata(meta, sidecar)
            if not rows:
                continue
            args = normalize_hparams(meta.get("args") or {})
            best_m = min(rows, key=lambda row: row["mpjpe"])
            best_pa = min(rows, key=lambda row: row["pa_mpjpe"])
            latest = rows[-1]
            conn.execute(
                """
                insert into runs values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    run_number(name),
                    classify_run(name),
                    meta.get("source"),
                    str(run_dir) if run_dir is not None else None,
                    len(rows),
                    latest["epoch"],
                    latest["mpjpe"],
                    latest["pa_mpjpe"],
                    best_m["epoch"],
                    best_m["mpjpe"],
                    best_m["pa_mpjpe"],
                    best_pa["epoch"],
                    best_pa["pa_mpjpe"],
                    best_pa["mpjpe"],
                    json.dumps(args, sort_keys=True),
                    meta.get("pipeline_note"),
                ),
            )
            for row in rows:
                conn.execute(
                    """
                    insert or replace into epochs values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        name,
                        row["epoch"],
                        row["mpjpe"],
                        row["pa_mpjpe"],
                        row["stage2_mpjpe"],
                        row["stage2_pa_mpjpe"],
                        row["root_mpjpe"],
                        row["gate_mean"],
                        row["update_abs_mean"],
                        row["base_mpjpe"],
                        row["base_pa_mpjpe"],
                        row["body_enabled"],
                        json.dumps(row["raw"], sort_keys=True),
                    ),
                )
            for key, value in args.items():
                conn.execute(
                    "insert or replace into run_hparams values (?, ?, ?, ?)",
                    (name, key, stringify(value), numeric(value)),
                )
        conn.commit()
    finally:
        conn.close()


def normalize_hparams(args: dict) -> dict:
    result = {}
    for key in IMPORTANT_HPARAMS:
        if key in args:
            result[key] = args[key]
    for key, value in args.items():
        if key.startswith(("body_", "camera_", "pa_", "gt_", "preserve_", "do_no_harm_", "gate_", "global_", "root_", "disable_", "ablation_")):
            result.setdefault(key, value)
    return result


def stringify(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def numeric(value):
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rows_to_dicts(cursor: sqlite3.Cursor) -> list[dict]:
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


class DashboardHandler(BaseHTTPRequestHandler):
    db_path: Path

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.respond_html(DASHBOARD_HTML)
            return
        if parsed.path == "/api/runs":
            self.respond_json(self.api_runs(parsed.query))
            return
        if parsed.path == "/api/epochs":
            self.respond_json(self.api_epochs(parsed.query))
            return
        if parsed.path == "/api/hparams":
            self.respond_json(self.api_hparams())
            return
        if parsed.path == "/api/effect":
            self.respond_json(self.api_effect(parsed.query))
            return
        self.send_error(404)

    def api_runs(self, query: str) -> list[dict]:
        params = parse_qs(query)
        family = params.get("family", [""])[0]
        where = ""
        values: list[str] = []
        if family:
            where = "where family = ?"
            values.append(family)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"""
                select * from runs
                {where}
                order by best_mpjpe asc, best_pa_mpjpe asc
                """,
                values,
            )
            return rows_to_dicts(cursor)

    def api_epochs(self, query: str) -> list[dict]:
        params = parse_qs(query)
        run = params.get("run", [""])[0]
        if not run:
            return []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                select epoch, mpjpe, pa_mpjpe, gate_mean, update_abs_mean
                from epochs where run_name = ? order by epoch
                """,
                (run,),
            )
            return rows_to_dicts(cursor)

    def api_hparams(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            keys = rows_to_dicts(
                conn.execute(
                    """
                    select key, count(distinct value) as values_count
                    from run_hparams group by key having values_count > 1
                    order by key
                    """
                )
            )
            families = rows_to_dicts(
                conn.execute("select family, count(*) as runs from runs group by family order by family")
            )
            return {"keys": keys, "families": families}

    def api_effect(self, query: str) -> list[dict]:
        params = parse_qs(query)
        key = params.get("key", [""])[0]
        metric = params.get("metric", ["best_mpjpe"])[0]
        family = params.get("family", [""])[0]
        if not key:
            return []
        allowed = {"best_mpjpe", "best_pa_mpjpe", "latest_mpjpe", "latest_pa_mpjpe", "pa_at_best_mpjpe"}
        if metric not in allowed:
            metric = "best_mpjpe"
        where = "h.key = ?"
        values = [key]
        if family:
            where += " and r.family = ?"
            values.append(family)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"""
                select r.run_name, r.family, h.value, h.numeric_value, r.{metric} as metric_value,
                       r.best_mpjpe, r.pa_at_best_mpjpe, r.best_mpjpe_epoch
                from runs r join run_hparams h on r.run_name = h.run_name
                where {where}
                order by case when h.numeric_value is null then 1 else 0 end,
                         h.numeric_value, h.value, r.{metric}
                """,
                values,
            )
            return rows_to_dicts(cursor)

    def respond_json(self, obj) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def respond_html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage2U Hparam Dashboard</title>
  <style>
    :root { color-scheme: light; --ink:#1c2430; --muted:#5c6675; --line:#d9dee7; --panel:#f7f8fb; --accent:#0f766e; }
    body { margin:0; font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif; color:var(--ink); background:white; }
    header { padding:18px 24px 10px; border-bottom:1px solid var(--line); }
    h1 { margin:0 0 6px; font-size:20px; font-weight:650; letter-spacing:0; }
    main { padding:16px 24px 28px; display:grid; grid-template-columns:minmax(440px,1.15fr) minmax(420px,.85fr); gap:18px; }
    section { min-width:0; }
    .toolbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:12px; }
    label { color:var(--muted); font-size:12px; }
    select,input { height:32px; border:1px solid var(--line); border-radius:6px; padding:0 8px; background:white; }
    button { height:32px; border:1px solid var(--line); border-radius:6px; padding:0 10px; background:var(--panel); cursor:pointer; }
    table { width:100%; border-collapse:collapse; table-layout:fixed; }
    th,td { border-bottom:1px solid var(--line); padding:7px 6px; text-align:left; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    th { color:var(--muted); font-size:12px; font-weight:600; background:var(--panel); position:sticky; top:0; }
    tr.selected { background:#e7f5f3; }
    .tablewrap { max-height:580px; overflow:auto; border:1px solid var(--line); border-radius:8px; }
    .cards { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; margin-bottom:12px; }
    .card { border:1px solid var(--line); border-radius:8px; padding:10px; background:var(--panel); }
    .metric { display:block; font-size:20px; font-weight:700; color:var(--accent); }
    canvas { width:100%; height:270px; border:1px solid var(--line); border-radius:8px; background:white; }
    .muted { color:var(--muted); }
    @media (max-width: 980px) { main { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<header>
  <h1>Stage2U Hparam Dashboard</h1>
  <div class="muted">Run table, per-run epoch curves, and metric curves against selected hparams.</div>
</header>
<main>
  <section>
    <div class="toolbar">
      <label>Family <select id="family"></select></label>
      <label>Search <input id="search" placeholder="run name or hparam"></label>
      <button id="refresh">Refresh</button>
    </div>
    <div class="tablewrap">
      <table>
        <thead><tr><th style="width:38%">Run</th><th>Family</th><th>Best MPJPE</th><th>PA @ Best</th><th>Best PA</th><th>Epochs</th></tr></thead>
        <tbody id="runs"></tbody>
      </table>
    </div>
  </section>
  <section>
    <div class="cards">
      <div class="card"><span class="muted">Selected best MPJPE</span><span id="bestm" class="metric">-</span></div>
      <div class="card"><span class="muted">Selected best PA</span><span id="bestpa" class="metric">-</span></div>
    </div>
    <canvas id="epochChart" width="760" height="270"></canvas>
    <div class="toolbar" style="margin-top:14px">
      <label>Hparam <select id="hparam"></select></label>
      <label>Metric <select id="metric">
        <option value="best_mpjpe">Best MPJPE</option>
        <option value="pa_at_best_mpjpe">PA at best MPJPE</option>
        <option value="best_pa_mpjpe">Best PA</option>
        <option value="latest_mpjpe">Latest MPJPE</option>
      </select></label>
    </div>
    <canvas id="effectChart" width="760" height="270"></canvas>
  </section>
</main>
<script>
const state = { runs: [], selected: null, hparams: [] };
const $ = id => document.getElementById(id);
const fmt = v => (v === null || v === undefined || Number.isNaN(Number(v))) ? '-' : Number(v).toFixed(3);
async function getJSON(url){ const r = await fetch(url); return await r.json(); }
async function loadMeta(){
  const meta = await getJSON('/api/hparams');
  $('family').innerHTML = '<option value="">all</option>' + meta.families.map(f => `<option>${f.family}</option>`).join('');
  state.hparams = meta.keys.map(k => k.key);
  $('hparam').innerHTML = state.hparams.map(k => `<option>${k}</option>`).join('');
}
async function loadRuns(){
  const fam = $('family').value;
  state.runs = await getJSON('/api/runs' + (fam ? '?family=' + encodeURIComponent(fam) : ''));
  renderRuns();
  if (!state.selected && state.runs.length) selectRun(state.runs[0].run_name);
  drawEffect();
}
function renderRuns(){
  const q = $('search').value.toLowerCase();
  const rows = state.runs.filter(r => !q || r.run_name.toLowerCase().includes(q) || (r.hparam_json||'').toLowerCase().includes(q));
  $('runs').innerHTML = rows.map(r => `<tr data-run="${r.run_name}" class="${state.selected===r.run_name?'selected':''}">
    <td title="${r.run_name}">${r.run_name}</td><td>${r.family}</td><td>${fmt(r.best_mpjpe)}</td>
    <td>${fmt(r.pa_at_best_mpjpe)}</td><td>${fmt(r.best_pa_mpjpe)}</td><td>${r.epochs}</td></tr>`).join('');
  [...$('runs').querySelectorAll('tr')].forEach(tr => tr.onclick = () => selectRun(tr.dataset.run));
}
async function selectRun(run){
  state.selected = run;
  renderRuns();
  const rows = await getJSON('/api/epochs?run=' + encodeURIComponent(run));
  const rec = state.runs.find(r => r.run_name === run);
  $('bestm').textContent = rec ? `${fmt(rec.best_mpjpe)} / ${fmt(rec.pa_at_best_mpjpe)} @ ${rec.best_mpjpe_epoch}` : '-';
  $('bestpa').textContent = rec ? `${fmt(rec.mpjpe_at_best_pa)} / ${fmt(rec.best_pa_mpjpe)} @ ${rec.best_pa_epoch}` : '-';
  drawEpoch(rows);
}
function drawAxes(ctx,w,h,x0,y0,x1,y1,label){
  ctx.clearRect(0,0,w,h); ctx.strokeStyle='#d9dee7'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x0,y1); ctx.lineTo(x1,y1); ctx.stroke();
  ctx.fillStyle='#5c6675'; ctx.font='12px system-ui'; ctx.fillText(label, x0, 16);
}
function drawEpoch(rows){
  const c=$('epochChart'), ctx=c.getContext('2d'), w=c.width, h=c.height, pad=38;
  drawAxes(ctx,w,h,pad,pad,w-pad,h-pad,'epoch curves: MPJPE teal, PA gray');
  if(!rows.length) return;
  const xs=rows.map(r=>r.epoch), ys=rows.flatMap(r=>[r.mpjpe,r.pa_mpjpe]);
  const xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=Math.min(...ys)-0.05, ymax=Math.max(...ys)+0.05;
  const x=v=>pad+(v-xmin)/(xmax-xmin||1)*(w-2*pad), y=v=>h-pad-(v-ymin)/(ymax-ymin||1)*(h-2*pad);
  [['mpjpe','#0f766e'],['pa_mpjpe','#64748b']].forEach(([key,color])=>{
    ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
    rows.forEach((r,i)=>{ const xx=x(r.epoch), yy=y(r[key]); if(i) ctx.lineTo(xx,yy); else ctx.moveTo(xx,yy); });
    ctx.stroke();
  });
}
async function drawEffect(){
  const key=$('hparam').value, metric=$('metric').value, fam=$('family').value;
  if(!key) return;
  const familyParam = fam ? `&family=${encodeURIComponent(fam)}` : '';
  const rows=await getJSON(`/api/effect?key=${encodeURIComponent(key)}&metric=${encodeURIComponent(metric)}${familyParam}`);
  const c=$('effectChart'), ctx=c.getContext('2d'), w=c.width, h=c.height, pad=46;
  drawAxes(ctx,w,h,pad,pad,w-pad,h-pad,`${metric} against ${key}`);
  const numeric=rows.filter(r=>r.numeric_value!==null && r.metric_value!==null);
  const plotted = numeric.length ? numeric : rows.filter(r=>r.metric_value!==null).map((r,i)=>({...r, numeric_value:i}));
  if(!plotted.length) return;
  const xs=plotted.map(r=>r.numeric_value), ys=plotted.map(r=>r.metric_value);
  const xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=Math.min(...ys)-0.05, ymax=Math.max(...ys)+0.05;
  const x=v=>pad+(v-xmin)/(xmax-xmin||1)*(w-2*pad), y=v=>h-pad-(v-ymin)/(ymax-ymin||1)*(h-2*pad);
  ctx.fillStyle='#0f766e';
  plotted.forEach(r=>{ ctx.beginPath(); ctx.arc(x(r.numeric_value),y(r.metric_value),4,0,Math.PI*2); ctx.fill(); });
  if(!numeric.length){
    ctx.fillStyle='#5c6675'; ctx.font='10px system-ui';
    plotted.slice(0,16).forEach(r=>ctx.fillText(String(r.value).slice(0,18), x(r.numeric_value)-20, h-pad+14));
  }
}
$('refresh').onclick=loadRuns; $('family').onchange=loadRuns; $('search').oninput=renderRuns; $('hparam').onchange=drawEffect; $('metric').onchange=drawEffect;
loadMeta().then(loadRuns);
</script>
</body>
</html>
"""


def serve(db_path: Path, host: str, port: int) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"Database does not exist: {db_path}")
    handler = type("BoundDashboardHandler", (DashboardHandler,), {"db_path": db_path})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serving Stage2U hparam dashboard at http://{host}:{port}")
    print(f"Database: {db_path}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build-db")
    build.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    build.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    build.add_argument("--include-smoke", action="store_true")
    server = sub.add_parser("serve")
    server.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=int(os.environ.get("STAGE2U_HPARAM_PORT", "8877")))
    args = parser.parse_args()
    if args.cmd == "build-db":
        build_db(args.run_root, args.db_path, include_smoke=bool(args.include_smoke))
        print(f"Wrote {args.db_path}")
    elif args.cmd == "serve":
        serve(args.db_path, args.host, args.port)


if __name__ == "__main__":
    main()
