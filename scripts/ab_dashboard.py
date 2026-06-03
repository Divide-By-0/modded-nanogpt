#!/usr/bin/env python3
"""Build overlaid train/val loss dashboard from logs/*.txt (optional wandb)."""
import base64
import glob
import io
import os
import re
import sys
from html import escape
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(REPO_ROOT, 'logs')
OUT_HTML = os.path.join(LOGS_DIR, 'ab_dashboard.html')

TRAIN_RE = re.compile(
    r'step:(\d+)/\d+ train_loss:([\d.]+)')
VAL_RE = re.compile(
    r'step:(\d+)/\d+ val_loss:([\d.]+)')
META_RE = {
    'ab_tag': re.compile(r'^ab_tag:(.*)$', re.M),
    'experiment_desc': re.compile(r'^experiment_desc:(.*)$', re.M),
    'wandb_run_name': re.compile(r'^wandb_run_name:(.*)$', re.M),
    'train_seed': re.compile(r'^train_seed:(.*)$', re.M),
    'max_train_seconds': re.compile(r'^max_train_seconds:(.*)$', re.M),
    'optimizer_mode': re.compile(r'^optimizer_mode:(.*)$', re.M),
    'learning_rate': re.compile(r'^learning_rate:(.*)$', re.M),
    'warmup_iters': re.compile(r'^warmup_iters:(.*)$', re.M),
    'warmdown_iters': re.compile(r'^warmdown_iters:(.*)$', re.M),
    'weight_decay': re.compile(r'^weight_decay:(.*)$', re.M),
    'muon_lr_multiplier': re.compile(r'^muon_lr_multiplier:(.*)$', re.M),
    'muon_momentum': re.compile(r'^muon_momentum:(.*)$', re.M),
    'muon_variant': re.compile(r'^muon_variant:(.*)$', re.M),
    'muon_beta2': re.compile(r'^muon_beta2:(.*)$', re.M),
    'aurora_beta': re.compile(r'^aurora_beta:(.*)$', re.M),
    'muon_nesterov': re.compile(r'^muon_nesterov:(.*)$', re.M),
    'muon_backend_steps': re.compile(r'^muon_backend_steps:(.*)$', re.M),
    'qk_norm_mode': re.compile(r'^qk_norm_mode:(.*)$', re.M),
    'embed_rmsnorm': re.compile(r'^embed_rmsnorm:(.*)$', re.M),
}
STOP_RE = re.compile(r'time_limit_hit:1 elapsed_wall_s:([\d.]+) stop_step:(\d+)')


@dataclass
class RunSeries:
    label: str
    key: str = ''
    mtime: float = 0.0
    main_change: str = ''
    details: str = ''
    source: str = ''
    status: str = ''
    train_steps: list = field(default_factory=list)
    train_losses: list = field(default_factory=list)
    val_steps: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)


def _meta(text, key):
    m = META_RE[key].search(text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return ''


def _settings_from_text(text):
    return {key: _meta(text, key) for key in META_RE}


def _label_from_settings(settings, path):
    tag = settings.get('ab_tag') or settings.get('wandb_run_name')
    desc = settings.get('experiment_desc')
    if tag and desc:
        return f'{tag}: {desc}'
    if not tag:
        for key in ('wandb_run_name',):
            if settings.get(key):
                tag = settings[key]
                break
    if not tag:
        short = os.path.splitext(os.path.basename(path))[0][:8]
        tag = f'legacy/no metadata ({short})'

    parts = []
    opt = settings.get('optimizer_mode')
    if opt:
        parts.append(f'opt={opt}')
    if settings.get('muon_backend_steps'):
        parts.append(f'NS={settings["muon_backend_steps"]}')
    if settings.get('muon_momentum'):
        parts.append(f'mom={settings["muon_momentum"]}')
    if settings.get('muon_variant') and settings['muon_variant'] != 'muon':
        parts.append(f'variant={settings["muon_variant"]}')
    if settings.get('muon_lr_multiplier'):
        parts.append(f'muon_lr={settings["muon_lr_multiplier"]}x')
    if settings.get('qk_norm_mode'):
        parts.append(f'qk={settings["qk_norm_mode"]}')
    if settings.get('embed_rmsnorm'):
        parts.append(f'embed_norm={settings["embed_rmsnorm"]}')
    if parts:
        return f'{tag}: ' + ', '.join(parts)
    return tag


def _details_from_settings(settings):
    preferred = [
        'train_seed',
        'max_train_seconds',
        'optimizer_mode',
        'learning_rate',
        'warmup_iters',
        'warmdown_iters',
        'weight_decay',
        'muon_lr_multiplier',
        'muon_momentum',
        'muon_nesterov',
        'muon_backend_steps',
        'qk_norm_mode',
        'embed_rmsnorm',
    ]
    parts = [f'{key}={settings[key]}' for key in preferred if settings.get(key)]
    if parts:
        return '; '.join(parts)
    return 'settings unavailable: this log was created before metadata logging was added'


def _has_settings_metadata(settings):
    keys = [
        'optimizer_mode',
        'learning_rate',
        'muon_lr_multiplier',
        'muon_momentum',
        'muon_backend_steps',
        'qk_norm_mode',
        'embed_rmsnorm',
        'experiment_desc',
    ]
    return any(settings.get(key) for key in keys)


def _main_change_from_settings(settings):
    desc = settings.get('experiment_desc')
    if desc:
        return desc
    tag = settings.get('ab_tag')
    if tag == 'baseline':
        return 'baseline'
    if tag == 'muon-steps3':
        return 'Muon NS steps 3'
    if tag == 'muon-mom98':
        return 'Muon momentum 0.98'
    if tag == 'adamw-all':
        return 'AdamW all params'
    if tag == 'qk-after-rope':
        return 'QK norm after RoPE'
    if tag == 'embed-rmsnorm':
        return 'embedding RMSNorm'
    if tag == 'triton-gemm-autotune':
        return 'Triton GEMM autotune'
    if settings.get('optimizer_mode') == 'adamw_all':
        return 'AdamW all params'
    if settings.get('qk_norm_mode') == 'after_rope':
        return 'QK norm after RoPE'
    if settings.get('embed_rmsnorm') in ('True', 'true', '1'):
        return 'embedding RMSNorm'
    return tag or 'settings ablation'


def _label_from_text(text, path):
    settings = _settings_from_text(text)
    return _label_from_settings(settings, path)


def _fallback_label_from_text(text, path):
    for key in ('ab_tag', 'wandb_run_name'):
        m = META_RE[key].search(text)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return os.path.splitext(os.path.basename(path))[0][:8]


def parse_log_file(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    settings = _settings_from_text(text)
    if not _has_settings_metadata(settings) and not os.environ.get('SHOW_LEGACY_RUNS'):
        return None
    stop = STOP_RE.search(text)
    series = RunSeries(
        label=_label_from_settings(settings, path),
        key=(settings.get('ab_tag') or _label_from_settings(settings, path)),
        mtime=os.path.getmtime(path),
        main_change=_main_change_from_settings(settings),
        details=_details_from_settings(settings),
        source=os.path.basename(path),
        status=(f'time limit at step {stop.group(2)} after {stop.group(1)}s' if stop else 'running/complete without time-limit marker'),
    )
    for m in TRAIN_RE.finditer(text):
        series.train_steps.append(int(m.group(1)))
        series.train_losses.append(float(m.group(2)))
    for m in VAL_RE.finditer(text):
        series.val_steps.append(int(m.group(1)))
        series.val_losses.append(float(m.group(2)))
    if not series.train_steps and not series.val_steps:
        return None
    return series


def load_runs_from_logs():
    pattern = os.path.join(LOGS_DIR, '*.txt')
    paths = sorted(glob.glob(pattern))
    runs = []
    for path in paths:
        parsed = parse_log_file(path)
        if parsed is not None:
            runs.append(parsed)
    return runs


def load_runs_from_wandb():
    if not os.environ.get('WANDB_API_KEY'):
        return []
    try:
        import wandb
    except ImportError:
        return []
    project = os.environ.get('WANDB_PROJECT', 'modded-nanogpt')
    api = wandb.Api()
    runs = []
    try:
        for run in api.runs(f'{project}'):
            hist = run.history(samples=10000)
            if hist.empty:
                continue
            cfg = run.config or {}
            settings = {
                'ab_tag': str(cfg.get('ab_tag') or ''),
                'experiment_desc': str(cfg.get('experiment_desc') or ''),
                'wandb_run_name': str(run.name or ''),
                'optimizer_mode': str(cfg.get('optimizer_mode') or ''),
                'muon_lr_multiplier': str(cfg.get('muon_lr_multiplier') or ''),
                'muon_momentum': str(cfg.get('muon_momentum') or ''),
                'muon_nesterov': str(cfg.get('muon_nesterov') or ''),
                'muon_backend_steps': str(cfg.get('muon_backend_steps') or ''),
                'qk_norm_mode': str(cfg.get('qk_norm_mode') or ''),
                'embed_rmsnorm': str(cfg.get('embed_rmsnorm') or ''),
            }
            series = RunSeries(
                label=_label_from_settings(settings, run.name or ''),
                key=(settings.get('ab_tag') or _label_from_settings(settings, run.name or '')),
                mtime=0.0,
                main_change=_main_change_from_settings(settings),
                details=_details_from_settings(settings),
                source=str(run.name or ''),
                status=str(run.state or ''),
            )
            if 'train_loss' in hist.columns and '_step' in hist.columns:
                mask = hist['train_loss'].notna()
                series.train_steps = hist.loc[mask, '_step'].astype(int).tolist()
                series.train_losses = hist.loc[mask, 'train_loss'].astype(float).tolist()
            if 'val_loss' in hist.columns and '_step' in hist.columns:
                mask = hist['val_loss'].notna()
                series.val_steps = hist.loc[mask, '_step'].astype(int).tolist()
                series.val_losses = hist.loc[mask, 'val_loss'].astype(float).tolist()
            if series.train_steps or series.val_steps:
                runs.append(series)
    except Exception:
        return []
    return runs


def _final(values):
    return values[-1] if values else None


def _fmt(v, nd=4):
    return f'{v:.{nd}f}' if isinstance(v, (int, float)) else ''


def plot_runs(runs, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # REASON: render the figure as inline SVG (vector) rather than a base64 PNG. SVG stays
    # razor-sharp at any browser zoom and on HiDPI/Retina displays, which the old dpi=120 PNG
    # did not. Point count here is tiny (a handful of val points per run) so SVG size is small.
    # A larger figure + readable fonts further help legibility with many overlaid runs.
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for series in runs:
        # label each curve with its final val loss so the legend doubles as a ranking key.
        fv = _final(series.val_losses)
        lab = series.label if fv is None else f'{series.label}  [val {fv:.4f}]'
        if series.train_steps:
            axes[0].plot(series.train_steps, series.train_losses, label=lab, alpha=0.9, linewidth=0.9)
        if series.val_steps:
            axes[1].plot(series.val_steps, series.val_losses, label=lab, alpha=0.9, linewidth=0.9, marker='o', markersize=2.5)
    for ax, title in ((axes[0], 'train_loss'), (axes[1], 'val_loss')):
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('step', fontsize=12)
        ax.set_ylabel('loss', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, which='both', alpha=0.3)
        ax.tick_params(labelsize=10)
    fig.tight_layout()

    buf = io.StringIO()
    fig.savefig(buf, format='svg')
    plt.close(fig)
    svg = buf.getvalue()
    # Strip the XML/doctype preamble so the <svg> can be inlined directly into the HTML body
    # (inline SVG scales responsively and avoids the blur of a rasterized data-URI image).
    idx = svg.find('<svg')
    svg_inline = svg[idx:] if idx != -1 else svg

    # Default ranking: best (lowest) final val loss first; runs without val data sink to the bottom.
    ranked = sorted(runs, key=lambda s: (_final(s.val_losses) is None, _final(s.val_losses) or 0.0))

    def cell(value, display=None, numeric=False):
        disp = display if display is not None else ('' if value is None else escape(str(value)))
        sortv = '' if value is None else escape(str(value))
        num_attr = ' data-numeric="1"' if numeric else ''
        return f'<td data-v="{sortv}"{num_attr}>{disp}</td>'

    rows = ''
    for s in ranked:
        fv = _final(s.val_losses)
        ft = _final(s.train_losses)
        rows += (
            '<tr>'
            + cell(s.label, escape(s.label))
            + cell(s.main_change, escape(s.main_change))
            + cell(fv, _fmt(fv), numeric=True)
            + cell(ft, _fmt(ft), numeric=True)
            + cell(s.val_steps[-1] if s.val_steps else None, numeric=True)
            + cell(s.train_steps[-1] if s.train_steps else None, numeric=True)
            + cell(len(s.val_steps), numeric=True)
            + cell(len(s.train_steps), numeric=True)
            + cell(s.status, escape(s.status))
            + cell(s.details, escape(s.details))
            + cell(s.source, escape(s.source))
            + '</tr>'
        )

    headers = [
        ('label', False), ('main change', False),
        ('final val loss', True), ('final train loss', True),
        ('last val step', True), ('last train step', True),
        ('val pts', True), ('train pts', True),
        ('status', False), ('settings', False), ('source', False),
    ]
    # column index 2 (final val loss) is the default sort, ascending.
    header_html = ''.join(
        f'<th onclick="sortTable(this,{i},{str(num).lower()})" title="click to sort">{escape(name)} &#8597;</th>'
        for i, (name, num) in enumerate(headers)
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>modded-nanogpt A/B dashboard</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; background: #111; color: #eee; }}
.plot {{ background: #fff; border-radius: 8px; padding: 8px; }}
.plot svg {{ width: 100%; height: auto; display: block; }}
table {{ border-collapse: collapse; margin-top: 16px; width: 100%; }}
td, th {{ border: 1px solid #444; padding: 6px 10px; }}
td {{ vertical-align: top; font-size: 13px; }}
th {{ font-size: 13px; cursor: pointer; user-select: none; background: #1d1d1d; position: sticky; top: 0; }}
th:hover {{ background: #2a2a2a; }}
tbody tr:nth-child(odd) {{ background: #161616; }}
td[data-numeric] {{ text-align: right; font-variant-numeric: tabular-nums; }}
tbody tr:first-child td[data-v]:nth-child(3) {{ font-weight: 700; color: #6fdc6f; }}
</style></head><body>
<h1>modded-nanogpt A/B loss comparison</h1>
<p>Runs: {len(runs)} — logs from <code>{LOGS_DIR}</code>. Table is sorted by <b>final val loss</b> (click any header to re-sort). Legacy logs without settings metadata are hidden (<code>SHOW_LEGACY_RUNS=1</code> to include). Same-<code>ab_tag</code> reruns de-duped to newest (<code>SHOW_DUPLICATE_RUNS=1</code> to include).</p>
<div class="plot">{svg_inline}</div>
<table id="runs"><thead><tr>{header_html}</tr></thead>
<tbody>
{rows}
</tbody></table>
<script>
function sortTable(th, col, numeric) {{
  const table = document.getElementById('runs');
  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.rows);
  const cur = table.getAttribute('data-sortcol');
  const dir = (cur == col && table.getAttribute('data-sortdir') == 'asc') ? 'desc' : 'asc';
  rows.sort((a, b) => {{
    let x = a.cells[col].getAttribute('data-v');
    let y = b.cells[col].getAttribute('data-v');
    if (numeric) {{
      x = (x === '' || x === null) ? Infinity : parseFloat(x);
      y = (y === '' || y === null) ? Infinity : parseFloat(y);
      return dir == 'asc' ? x - y : y - x;
    }}
    x = (x || '').toLowerCase(); y = (y || '').toLowerCase();
    return dir == 'asc' ? (x > y ? 1 : x < y ? -1 : 0) : (x < y ? 1 : x > y ? -1 : 0);
  }});
  rows.forEach(r => tbody.appendChild(r));
  table.setAttribute('data-sortcol', col);
  table.setAttribute('data-sortdir', dir);
}}
</script>
</body></html>"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    os.chdir(REPO_ROOT)
    runs = load_runs_from_logs()
    if not runs:
        runs = load_runs_from_wandb()
    if not runs:
        print(f'No runs found under {LOGS_DIR}/*.txt (and no wandb history).', file=sys.stderr)
        return 1
    # De-dupe reruns of the same ablation by ab_tag by default. Set
    # SHOW_DUPLICATE_RUNS=1 when investigating repeated runs.
    if not os.environ.get('SHOW_DUPLICATE_RUNS'):
        by_key = {}
        for r in runs:
            prev = by_key.get(r.key)
            if prev is None or r.mtime >= prev.mtime:
                by_key[r.key] = r
        runs = list(by_key.values())

    by_label = {}
    for r in runs:
        by_label[r.label] = r
    runs = list(by_label.values())
    plot_runs(runs, OUT_HTML)
    print(f'Wrote {OUT_HTML} ({len(runs)} run(s))')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
