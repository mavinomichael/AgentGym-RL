from pathlib import Path
import csv, json, re, math
from PIL import Image, ImageDraw, ImageFont

REPO = Path('/Users/mavinomichael/PycharmProjects/AgentGym-RL')
REPORT = REPO / 'reports' / '2026-03-30_multi_agent_collapse_paper_handoff'
FIG = REPORT / 'figures'

# ---------- Helpers ----------
def read_tsv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))

def to_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s in {'NA', 'None'}:
        return None
    try:
        return float(s)
    except ValueError:
        return None

def load_json(path):
    with open(path) as f:
        return json.load(f)

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

ensure_dir(REPORT)
ensure_dir(FIG)

# ---------- Source files ----------
README = REPO / 'README.md'
ARXIV_URL = 'https://arxiv.org/abs/2509.08755'
MAVINO_DIR = REPO / 'reports' / 'babyai_multi_agent_diagnostics_2026-03-21' / 'mavino_collapse_2agent_scaling_100'
FIXED2_DIR = REPO / 'reports' / 'babyai_multi_agent_diagnostics_2026-03-09'
FIXED2_DIAG60 = REPO / 'reports' / 'babyai_multi_agent_diagnostics_2026-03-10' / 'diagnostic_step60_analysis.txt'
FIXED2_DIAG100 = REPO / 'reports' / 'babyai_multi_agent_diagnostics_2026-03-11' / 'diagnostic_step100_trace_summary.txt'
PLAIN100_DIR = REPO / 'reports' / '2026-03-23_plain_split_retry_v2_deep_research'
DENSE500_DIR = REPO / 'reports' / '2026-03-23_dense500_eval_and_traces'
FIXED2_NOTAG_DIR = REPO / 'reports' / '20260402T123754Z_babyai_2agent_fixed20_8gpu_plain_split_notag_persist_v1'
THREE_AGENT_DIR = REPO / 'reports' / '20260403_130617_babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1'
PROMPT_ALIGN = REPO / 'reports' / 'babyai_multi_agent_diagnostics_2026-03-17' / 'prompt_alignment_single_vs_multi.md'

# ---------- Build run catalog ----------
rows = []
source_manifest = []

def add_source(category, path, note):
    source_manifest.append({'category': category, 'path': str(path), 'note': note})

# Single-agent baseline and fixed-round 2-agent composite from historical report
comparison_report = (FIXED2_DIR / 'babyai_eval_comparison_report.txt').read_text()
add_source('comparison_report', FIXED2_DIR / 'babyai_eval_comparison_report.txt', 'Historical composite comparison including single-agent baseline and fixed-round 2-agent checkpoints.')
add_source('comparison_plot', FIXED2_DIR / 'babyai_eval_comparison.png', 'Existing historical comparison figure from March 9.')

single_agent_baseline = {
    'run_family': 'single_agent_baseline',
    'run_variant': 'base_model_eval',
    'topology': 'single_agent',
    'prompt_regime': 'single_agent_native',
    'scaling_rl': False,
    'curriculum': 'none',
    'evidence_kind': 'reference',
    'step': 0,
    'Avg@1': 0.672827,
    'Pass@1': 0.733333,
    'ExecutorNativeFormatViolations': 0.0,
    'InvalidFormatTerminationRate': 0.0,
    'InvalidActionTerminationRate': 0.088889,
    'PlannerInvalidFormatRate': None,
    'PlannerFallbackRate': None,
    'PlannerTagOnlyRate': None,
    'source_file': str(FIXED2_DIR / 'babyai_eval_comparison_report.txt'),
    'notes': 'Single-agent BabyAI baseline from historical comparison report.'
}
rows.append(single_agent_baseline)

fixed2_points = [
    ('Sanity step-15', 15, 0.722397, 0.766667, 0.0, 0.0, 0.0, 0.0, 0.0),
    ('Sanity step-50', 50, 0.411854, 0.433333, 0.011111, 0.011111, 0.0, 0.677778, 0.677778),
    ('Diagnostic step-100 v2', 100, 0.657480, 0.688889, 0.0, 0.0, 0.011111, 0.011111, 0.011111),
    ('Resume step-236', 236, 0.431932, 0.533333, 0.033333, 0.033333, 0.422222, 1.0, 1.0),
    ('Resume step-350', 350, -0.200000, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0),
]
for name, step, avg, p1, execfmt, invfmt, invact, planner_inv, planner_fb in fixed2_points:
    rows.append({
        'run_family': 'fixed_round_2agent_tagged',
        'run_variant': name,
        'topology': 'planner_executor',
        'prompt_regime': 'tagged_planner_executor',
        'scaling_rl': False,
        'curriculum': 'fixed_20',
        'evidence_kind': 'composite_checkpoints',
        'step': step,
        'Avg@1': avg,
        'Pass@1': p1,
        'ExecutorNativeFormatViolations': execfmt,
        'InvalidFormatTerminationRate': invfmt,
        'InvalidActionTerminationRate': invact,
        'PlannerInvalidFormatRate': planner_inv,
        'PlannerFallbackRate': planner_fb,
        'PlannerTagOnlyRate': 0.0,
        'source_file': str(FIXED2_DIR / 'babyai_eval_comparison_report.txt'),
        'notes': 'Composite fixed-round 2-agent evidence assembled from multiple historical checkpoints; not a single uninterrupted training trajectory.'
    })

# Tagged scaling run (Mavino collapse)
tagged_scaling = read_tsv(MAVINO_DIR / 'eval_summary.tsv')
add_source('tagged_scaling_metrics', MAVINO_DIR / 'eval_summary.tsv', 'Tagged 2-agent ScalingRL checkpoint eval summary.')
add_source('tagged_scaling_report', MAVINO_DIR / 'MAVINO_COLLAPSE_REPORT.md', 'Narrative report for tagged prompt collapse run.')
add_source('tagged_scaling_traces', MAVINO_DIR / 'representative_trace_examples.md', 'Representative trace excerpts for tagged prompt collapse.')
for r in tagged_scaling:
    rows.append({
        'run_family': 'tagged_scaling_2agent',
        'run_variant': f"step_{r['step']}",
        'topology': 'planner_executor',
        'prompt_regime': 'tagged_planner_executor',
        'scaling_rl': True,
        'curriculum': '[6,13,20]',
        'evidence_kind': 'single_run_checkpoints',
        'step': int(r['step']),
        'Avg@1': to_float(r['Avg@1']),
        'Pass@1': to_float(r['Pass@1']),
        'ExecutorNativeFormatViolations': to_float(r['ExecutorNativeFormatViolations']),
        'InvalidFormatTerminationRate': to_float(r['InvalidFormatTerminationRate']),
        'InvalidActionTerminationRate': to_float(r['InvalidActionTerminationRate']),
        'PlannerInvalidFormatRate': to_float(r['PlannerInvalidFormatRate']),
        'PlannerFallbackRate': to_float(r['PlannerFallbackRate']),
        'PlannerTagOnlyRate': to_float(r['PlannerTagOnlyRate']),
        'source_file': str(MAVINO_DIR / 'eval_summary.tsv'),
        'notes': 'Tagged prompt ScalingRL run that exhibits recursive prompt-scaffold collapse.'
    })

# Plain split scaling 100-step schedule [6,13,20]
plain100 = read_tsv(PLAIN100_DIR / 'checkpoint_metrics.tsv')
add_source('plain_split_scaling_metrics', PLAIN100_DIR / 'checkpoint_metrics.tsv', 'No-tag ScalingRL run with plain split prompts and retry v2.')
add_source('plain_split_scaling_report', PLAIN100_DIR / 'DEEP_RESEARCH_ANALYSIS.md', 'Narrative analysis for no-tag ScalingRL run.')
for r in plain100:
    rows.append({
        'run_family': 'plain_split_scaling_2agent',
        'run_variant': f"step_{r['step']}",
        'topology': 'planner_executor',
        'prompt_regime': 'plain_split_no_tags',
        'scaling_rl': True,
        'curriculum': '[6,13,20]',
        'evidence_kind': 'single_run_checkpoints',
        'step': int(r['step']),
        'Avg@1': to_float(r['Avg@1']),
        'Pass@1': to_float(r['Pass@1']),
        'ExecutorNativeFormatViolations': to_float(r['ExecutorNativeFormatViolations']),
        'InvalidFormatTerminationRate': to_float(r['InvalidFormatTerminationRate']),
        'InvalidActionTerminationRate': to_float(r['InvalidActionTerminationRate']),
        'PlannerInvalidFormatRate': to_float(r['PlannerInvalidFormatRate']),
        'PlannerFallbackRate': to_float(r['PlannerFallbackRate']),
        'PlannerTagOnlyRate': to_float(r['PlannerTagOnlyRate']),
        'source_file': r.get('eval_log') or str(PLAIN100_DIR / 'checkpoint_metrics.tsv'),
        'notes': 'No-tag ScalingRL run with coarse curriculum.'
    })

# Dense curriculum no-tag scaling [6,8,10,13,16,20]
dense500 = read_tsv(DENSE500_DIR / 'checkpoint_metrics.tsv')
add_source('dense_scaling_metrics', DENSE500_DIR / 'checkpoint_metrics.tsv', 'No-tag dense-curriculum ScalingRL run with 500-step horizon.')
add_source('dense_scaling_report', DENSE500_DIR / 'SUMMARY.md', 'Narrative summary for dense-curriculum run.')
add_source('dense_scaling_traces', DENSE500_DIR / 'selected_trace_examples.md', 'Selected trace excerpts for dense-curriculum run.')
for r in dense500:
    rows.append({
        'run_family': 'dense_scaling_2agent',
        'run_variant': f"step_{r['step']}",
        'topology': 'planner_executor',
        'prompt_regime': 'plain_split_no_tags',
        'scaling_rl': True,
        'curriculum': '[6,8,10,13,16,20]',
        'evidence_kind': 'single_run_checkpoints',
        'step': int(r['step']),
        'Avg@1': to_float(r['Avg@1']),
        'Pass@1': to_float(r['Pass@1']),
        'ExecutorNativeFormatViolations': to_float(r['ExecutorNativeFormatViolations']),
        'InvalidFormatTerminationRate': to_float(r['InvalidFormatTerminationRate']),
        'InvalidActionTerminationRate': to_float(r['InvalidActionTerminationRate']),
        'PlannerInvalidFormatRate': to_float(r['PlannerInvalidFormatRate']),
        'PlannerFallbackRate': to_float(r['PlannerFallbackRate']),
        'PlannerTagOnlyRate': to_float(r['PlannerTagOnlyRate']),
        'source_file': r['log_path'],
        'notes': 'No-tag ScalingRL run with denser curriculum and retries.'
    })

# Fixed-round no-tag persistent run
fixed2_notag = read_tsv(FIXED2_NOTAG_DIR / 'checkpoint_metrics.tsv')
add_source('fixed_round_2agent_no_tag_metrics', FIXED2_NOTAG_DIR / 'checkpoint_metrics.tsv', 'Fixed-round 2-agent no-tag checkpoint metrics.')
add_source('fixed_round_2agent_no_tag_report', FIXED2_NOTAG_DIR / 'summary.md', 'Narrative summary for fixed-round 2-agent no-tag run.')
for r in fixed2_notag:
    rows.append({
        'run_family': 'fixed_round_2agent_no_tags',
        'run_variant': f"step_{r['step']}",
        'topology': 'planner_executor',
        'prompt_regime': 'plain_split_no_tags',
        'scaling_rl': False,
        'curriculum': 'fixed_20',
        'evidence_kind': 'single_run_checkpoints',
        'step': int(r['step']),
        'Avg@1': to_float(r['Avg@1']),
        'Pass@1': to_float(r['Pass@1']),
        'ExecutorNativeFormatViolations': to_float(r['ExecutorNativeFormatViolations']),
        'InvalidFormatTerminationRate': to_float(r['InvalidFormatTerminationRate']),
        'InvalidActionTerminationRate': to_float(r['InvalidActionTerminationRate']),
        'PlannerInvalidFormatRate': to_float(r['PlannerInvalidFormatRate']),
        'PlannerFallbackRate': to_float(r['PlannerFallbackRate']),
        'PlannerTagOnlyRate': to_float(r['PlannerTagOnlyRate']),
        'source_file': r.get('eval_log') or str(FIXED2_NOTAG_DIR / 'checkpoint_metrics.tsv'),
        'notes': 'Fixed-round 2-agent no-tag run with persistent disk-backed checkpoints.'
    })

# 3-agent no-tag scaling run with executor reviewer
three_agent = read_tsv(THREE_AGENT_DIR / 'checkpoint_metrics.tsv')
add_source('three_agent_scaling_metrics', THREE_AGENT_DIR / 'checkpoint_metrics.tsv', '3-agent no-tag executor-reviewer ScalingRL checkpoint metrics.')
add_source('three_agent_scaling_report', THREE_AGENT_DIR / 'summary.md', 'Narrative summary for 3-agent executor-reviewer run.')
add_source('three_agent_scaling_traces', THREE_AGENT_DIR / 'selected_trace_examples.md', 'Selected trace excerpts for the 3-agent executor-reviewer run.')
for r in three_agent:
    rows.append({
        'run_family': 'three_agent_executor_reviewer_scaling',
        'run_variant': f"step_{r['step']}",
        'topology': 'planner_executor_reviewer',
        'prompt_regime': 'plain_split_no_tags_with_executor_reviewer',
        'scaling_rl': True,
        'curriculum': '[6,8,10,13,16,20]',
        'evidence_kind': 'single_run_checkpoints',
        'step': int(r['step']),
        'Avg@1': to_float(r['Avg@1']),
        'Pass@1': to_float(r['Pass@1']),
        'ExecutorNativeFormatViolations': to_float(r['ExecutorNativeFormatViolations']),
        'InvalidFormatTerminationRate': to_float(r['InvalidFormatTerminationRate']),
        'InvalidActionTerminationRate': to_float(r['InvalidActionTerminationRate']),
        'PlannerInvalidFormatRate': to_float(r['PlannerInvalidFormatRate']),
        'PlannerFallbackRate': to_float(r['PlannerFallbackRate']),
        'PlannerTagOnlyRate': to_float(r['PlannerTagOnlyRate']),
        'source_file': r.get('eval_log') or str(THREE_AGENT_DIR / 'checkpoint_metrics.tsv'),
        'notes': '3-agent no-tag ScalingRL run with planner, executor, and executor-reviewer roles sharing one policy.'
    })

# Persist run catalog and source manifest
catalog_json = REPORT / 'run_catalog.json'
catalog_tsv = REPORT / 'run_catalog.tsv'
with catalog_json.open('w') as f:
    json.dump(rows, f, indent=2)
with catalog_tsv.open('w', newline='') as f:
    fieldnames = ['run_family','run_variant','topology','prompt_regime','scaling_rl','curriculum','evidence_kind','step','Avg@1','Pass@1','ExecutorNativeFormatViolations','InvalidFormatTerminationRate','InvalidActionTerminationRate','PlannerInvalidFormatRate','PlannerFallbackRate','PlannerTagOnlyRate','source_file','notes']
    w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
    w.writeheader()
    w.writerows(rows)
manifest_tsv = REPORT / 'source_manifest.tsv'
with manifest_tsv.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['category','path','note'], delimiter='\t')
    w.writeheader()
    w.writerows(source_manifest)

# ---------- Selected traces ----------
trace_packets = {}
# Tagged run traces
rep = load_json(MAVINO_DIR / 'representative_trace_examples.json')
trace_packets['tagged_step55_first_header_leak'] = {
    'regime': 'tagged_scaling_2agent',
    'description': 'Earliest visible header leak in the tagged prompt regime.',
    'source_file': str(MAVINO_DIR / 'representative_trace_examples.json'),
    'data': rep['first_header_leak_step55'],
}
trace_packets['tagged_step100_recursive_scaffold'] = {
    'regime': 'tagged_scaling_2agent',
    'description': 'Terminal recursive prompt-scaffold contamination in the tagged prompt regime.',
    'source_file': rep['terminal_step100_recursive_scaffold'].get('_source_file', str(MAVINO_DIR / 'representative_trace_examples.json')),
    'data': rep['terminal_step100_recursive_scaffold'],
}

# Fixed-round traces: onset and stable example
# step45 too_long onset
found = None
for path in sorted((REPO / 'reports' / 'babyai_multi_agent_diagnostics_2026-03-10' / 'diagnostic_step60_trace_train' / 'trace_train').glob('*.jsonl')):
    with path.open() as f:
        for line in f:
            o = json.loads(line)
            if o.get('training_step') == 45 and o.get('planner_validation_reason') == 'too_long':
                found = o
                source = str(path)
                break
    if found:
        break
if found:
    trace_packets['fixed_round_step45_planner_too_long'] = {
        'regime': 'fixed_round_2agent_tagged',
        'description': 'Planner verbosity drift onset in fixed-round 2-agent training.',
        'source_file': source,
        'data': found,
    }
# step100 v2 stable example
found = None
for path in sorted((REPO / 'reports' / 'babyai_multi_agent_diagnostics_2026-03-11' / 'diagnostic_step100_v2_trace_train').glob('*.jsonl')):
    with path.open() as f:
        for line in f:
            o = json.loads(line)
            if o.get('training_step') == 100 and o.get('validation_valid'):
                found = o
                source = str(path)
                break
    if found:
        break
if found:
    trace_packets['fixed_round_step100_v2_stable'] = {
        'regime': 'fixed_round_2agent_tagged',
        'description': 'Stable fixed-round 2-agent example after planner stabilization.',
        'source_file': source,
        'data': found,
    }

# Plain split no-tag traces from existing extracted report
plain_key = load_json(PLAIN100_DIR / 'key_trace_examples_selected.json')
for key in ['200_valid', '400_invalid_action', '450_invalid_format']:
    trace_packets[f'plain_split_{key}'] = {
        'regime': 'plain_split_scaling_2agent',
        'description': {
            '200_valid': 'Peak coarse-schedule no-tag example.',
            '400_invalid_action': 'Late invalid-action degradation in coarse no-tag run.',
            '450_invalid_format': 'Terminal executor format collapse in coarse no-tag run.',
        }[key],
        'source_file': plain_key[key].get('source_file', str(PLAIN100_DIR / 'key_trace_examples_selected.json')),
        'data': plain_key[key],
    }

# Dense500 traces from local extracted report
dense_key = load_json(DENSE500_DIR / 'selected_trace_examples.json')
for key in ['step300_peak_valid_example', 'step400_transition_example', 'step450_retry_exhaustion_example', 'step500_terminal_collapse_example']:
    trace_packets[f'dense500_{key}'] = {
        'regime': 'dense_scaling_2agent',
        'description': {
            'step300_peak_valid_example': 'Peak dense-curriculum no-tag example.',
            'step400_transition_example': 'Transition-stage degradation in dense-curriculum run.',
            'step450_retry_exhaustion_example': 'Retry exhaustion during late dense-curriculum collapse.',
            'step500_terminal_collapse_example': 'Terminal executor collapse in dense-curriculum run.',
        }[key],
        'source_file': str(DENSE500_DIR / 'selected_trace_examples.json'),
        'data': dense_key[key],
    }

# 3-agent reviewer traces from local copied report
three_agent_key = load_json(THREE_AGENT_DIR / 'selected_trace_examples.json')
for key in ['step100_healthy', 'step150_reviewer_false_retry', 'step350_schema_leak', 'step400_onset', 'step450_terminal']:
    trace_packets[f'three_agent_{key}'] = {
        'regime': 'three_agent_executor_reviewer_scaling',
        'description': {
            'step100_healthy': 'Peak early 3-agent executor-reviewer example.',
            'step150_reviewer_false_retry': 'Early instability where the reviewer requests retry even though executor validation is otherwise acceptable.',
            'step350_schema_leak': 'Recovered-performance example with reviewer schema leaking into executor output.',
            'step400_onset': 'Onset of the terminal collapse regime in the 3-agent reviewer run.',
            'step450_terminal': 'Terminal format collapse in the 3-agent reviewer run.',
        }[key],
        'source_file': str(THREE_AGENT_DIR / 'selected_trace_examples.json'),
        'data': three_agent_key[key],
    }

trace_json = REPORT / 'selected_trace_packets.json'
with trace_json.open('w') as f:
    json.dump(trace_packets, f, indent=2)

trace_md = REPORT / 'selected_trace_packets.md'
with trace_md.open('w') as f:
    f.write('# Selected Trace Packets\n\n')
    for name, pkt in trace_packets.items():
        d = pkt['data']
        f.write(f'## {name}\n\n')
        f.write(f'- Regime: `{pkt["regime"]}`\n')
        f.write(f'- Description: {pkt["description"]}\n')
        f.write(f'- Source file: `{pkt["source_file"]}`\n')
        for key in ['training_step','item_id','round','rank','planner_validation_reason','validation_reason','retry_count_total']:
            if key in d:
                f.write(f'- `{key}`: `{d.get(key)}`\n')
        for key in ['observation_excerpt','planner_prompt','planner_raw_output','planner_message','executor_prompt','executor_first_pass_raw_output','executor_raw_output','available_actions','env_state_excerpt']:
            val = d.get(key)
            if not val:
                continue
            f.write(f'\n**{key}**\n```text\n{val}\n```\n')
        f.write('\n')

# ---------- Plot helpers using PIL ----------
COLORS = {
    'baseline': (80, 80, 80),
    'fixed_round_2agent_tagged': (180, 90, 20),
    'fixed_round_2agent_no_tags': (87, 117, 144),
    'tagged_scaling_2agent': (214, 39, 40),
    'plain_split_scaling_2agent': (44, 123, 182),
    'dense_scaling_2agent': (49, 163, 84),
    'three_agent_executor_reviewer_scaling': (148, 103, 189),
}
LABELS = {
    'fixed_round_2agent_tagged': 'Fixed-round 2-agent (tagged, composite)',
    'fixed_round_2agent_no_tags': 'Fixed-round 2-agent (no tags)',
    'tagged_scaling_2agent': 'ScalingRL 2-agent with tags',
    'plain_split_scaling_2agent': 'ScalingRL 2-agent no tags [6,13,20]',
    'dense_scaling_2agent': 'ScalingRL 2-agent no tags [6,8,10,13,16,20]',
    'three_agent_executor_reviewer_scaling': 'ScalingRL 3-agent no tags + executor reviewer',
}
PRIMARY_COMPARE_FAMILIES = [
    'fixed_round_2agent_tagged',
    'fixed_round_2agent_no_tags',
    'plain_split_scaling_2agent',
    'dense_scaling_2agent',
    'three_agent_executor_reviewer_scaling',
]

try:
    FONT = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 22)
    SMALL = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 18)
    TITLE = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 28)
except Exception:
    FONT = SMALL = TITLE = ImageFont.load_default()

plot_rows = [r for r in rows if r['run_family'] != 'single_agent_baseline']

def series(run_family, metric):
    pts = [(r['step'], r[metric]) for r in plot_rows if r['run_family'] == run_family and r[metric] is not None]
    pts.sort()
    return pts

def draw_axes(draw, x0, y0, x1, y1, x_ticks, y_ticks, y_label, title, subtitle=None):
    draw.rectangle([x0, y0, x1, y1], outline=(0,0,0), width=2)
    draw.text((x0, y0-36), title, fill=(0,0,0), font=TITLE)
    if subtitle:
        draw.text((x0, y0-10), subtitle, fill=(80,80,80), font=SMALL)
    for xv, label in x_ticks:
        draw.line((xv, y1, xv, y1+8), fill=(0,0,0), width=2)
        draw.text((xv-10, y1+12), str(label), fill=(0,0,0), font=SMALL)
    for yv, label in y_ticks:
        draw.line((x0-8, yv, x0, yv), fill=(0,0,0), width=2)
        draw.text((x0-60, yv-10), str(label), fill=(0,0,0), font=SMALL)
    draw.text((x0 + (x1-x0)//2 - 40, y1+42), 'Step', fill=(0,0,0), font=FONT)
    draw.text((10, y0 + (y1-y0)//2), y_label, fill=(0,0,0), font=FONT)

def draw_legend(draw, items, x, y):
    yy = y
    for label, color in items:
        draw.line((x, yy+10, x+30, yy+10), fill=color, width=4)
        draw.ellipse((x+12, yy+2, x+20, yy+18), fill=color)
        draw.text((x+40, yy), label, fill=(0,0,0), font=SMALL)
        yy += 26

def scale_x(step, x0, x1, xmax):
    return x0 + (step / float(xmax)) * (x1 - x0)

def scale_y(val, ymin, ymax, y0, y1):
    if val is None:
        return None
    if ymax == ymin:
        return y1
    frac = (val - ymin) / (ymax - ymin)
    return y1 - frac * (y1 - y0)

def draw_series(draw, pts, color, x0, y0, x1, y1, ymin, ymax, width=4):
    mapped = []
    for s, v in pts:
        x = scale_x(s, x0, x1, max(step for step, _ in pts) if pts else 1)
        y = scale_y(v, ymin, ymax, y0, y1)
        mapped.append((x, y, s, v))
    if len(mapped) >= 2:
        draw.line([(x, y) for x, y, _, _ in mapped], fill=color, width=width)
    for x, y, s, v in mapped:
        draw.ellipse((x-5, y-5, x+5, y+5), fill=color)

# Figure 1: Pass@1 primary comparison
img = Image.new('RGB', (1500, 900), 'white')
d = ImageDraw.Draw(img)
plot = (110, 120, 980, 760)
comparison_max_step = max(r['step'] for r in rows if r['run_family'] in PRIMARY_COMPARE_FAMILIES)
comparison_ticks = [0,50,100,150,200,250,300,350,400,450,500,550,600]
xt = [(scale_x(v, plot[0], plot[2], comparison_max_step), v) for v in comparison_ticks if v <= comparison_max_step]
yt = [(scale_y(v, 0, 0.9, plot[1], plot[3]), f'{v:.1f}') for v in [0.0,0.2,0.4,0.6,0.8]]
draw_axes(d, *plot, xt, yt, 'Pass@1', 'Pass@1 across paper comparison regimes', 'Fixed-round vs ScalingRL, with and without tags, plus the 3-agent executor-reviewer ablation')
# baseline
base_y = scale_y(single_agent_baseline['Pass@1'], 0, 0.9, plot[1], plot[3])
d.line((plot[0], base_y, plot[2], base_y), fill=COLORS['baseline'], width=3)
d.text((plot[2]-120, base_y-20), 'single-agent baseline', fill=COLORS['baseline'], font=SMALL)
for fam in PRIMARY_COMPARE_FAMILIES:
    pts = [(s, v) for s, v in series(fam, 'Pass@1')]
    mapped = [(scale_x(s, plot[0], plot[2], comparison_max_step), scale_y(v, 0, 0.9, plot[1], plot[3])) for s, v in pts]
    if len(mapped) >= 2:
        d.line(mapped, fill=COLORS[fam], width=4)
    for x, y in mapped:
        d.ellipse((x-5, y-5, x+5, y+5), fill=COLORS[fam])
draw_legend(d, [(LABELS[f], COLORS[f]) for f in PRIMARY_COMPARE_FAMILIES], 1020, 160)
img.save(FIG / 'fig_pass1_comparison.png')

# Figure 2: Failure comparison, two panels
img = Image.new('RGB', (1500, 1100), 'white')
d = ImageDraw.Draw(img)
top = (110, 120, 980, 470)
bot = (110, 600, 980, 950)
xt_top = [(scale_x(v, top[0], top[2], comparison_max_step), v) for v in comparison_ticks if v <= comparison_max_step]
yt01_top = [(scale_y(v, 0, 1.0, top[1], top[3]), f'{v:.1f}') for v in [0,0.25,0.5,0.75,1.0]]
draw_axes(d, *top, xt_top, yt01_top, 'Rate', 'Executor format violations across paper comparison regimes')
for fam in PRIMARY_COMPARE_FAMILIES:
    pts = series(fam, 'ExecutorNativeFormatViolations')
    mapped = [(scale_x(s, top[0], top[2], comparison_max_step), scale_y(v, 0, 1.0, top[1], top[3])) for s, v in pts]
    if len(mapped) >= 2:
        d.line(mapped, fill=COLORS[fam], width=4)
    for x, y in mapped:
        d.ellipse((x-5, y-5, x+5, y+5), fill=COLORS[fam])
xt_bot = [(scale_x(v, bot[0], bot[2], comparison_max_step), v) for v in comparison_ticks if v <= comparison_max_step]
yt01_bot = [(scale_y(v, 0, 1.0, bot[1], bot[3]), f'{v:.1f}') for v in [0,0.25,0.5,0.75,1.0]]
draw_axes(d, *bot, xt_bot, yt01_bot, 'Rate', 'Planner invalid-format rate across paper comparison regimes')
for fam in PRIMARY_COMPARE_FAMILIES:
    pts = series(fam, 'PlannerInvalidFormatRate')
    mapped = [(scale_x(s, bot[0], bot[2], comparison_max_step), scale_y(v, 0, 1.0, bot[1], bot[3])) for s, v in pts]
    if len(mapped) >= 2:
        d.line(mapped, fill=COLORS[fam], width=4)
    for x, y in mapped:
        d.ellipse((x-5, y-5, x+5, y+5), fill=COLORS[fam])
draw_legend(d, [(LABELS[f], COLORS[f]) for f in PRIMARY_COMPARE_FAMILIES], 1020, 220)
img.save(FIG / 'fig_failure_comparison.png')

# Figure 3: Dense500 detail
img = Image.new('RGB', (1500, 1100), 'white')
d = ImageDraw.Draw(img)
top = (110, 120, 980, 470)
bot = (110, 600, 980, 950)
dense_max_step = max(r['step'] for r in rows if r['run_family'] == 'dense_scaling_2agent')
xt = [(scale_x(v, top[0], top[2], dense_max_step), v) for v in [50,100,150,200,250,300,350,400,450,500]]
yt_pass = [(scale_y(v, -0.2, 0.9, top[1], top[3]), f'{v:.1f}') for v in [-0.2,0.0,0.2,0.4,0.6,0.8]]
draw_axes(d, *top, xt, yt_pass, 'Metric', 'Dense no-tag ScalingRL run: performance and collapse')
for metric, color in [('Pass@1', COLORS['dense_scaling_2agent']), ('Avg@1', (31, 119, 180))]:
    pts = series('dense_scaling_2agent', metric)
    mapped = [(scale_x(s, top[0], top[2], dense_max_step), scale_y(v, -0.2, 0.9, top[1], top[3])) for s, v in pts]
    if len(mapped) >= 2:
        d.line(mapped, fill=color, width=4)
    for x, y in mapped:
        d.ellipse((x-5, y-5, x+5, y+5), fill=color)
d.text((1020, 170), 'Pass@1', fill=COLORS['dense_scaling_2agent'], font=SMALL)
d.text((1020, 200), 'Avg@1', fill=(31,119,180), font=SMALL)
xt2 = [(scale_x(v, bot[0], bot[2], dense_max_step), v) for v in [50,100,150,200,250,300,350,400,450,500]]
yt2 = [(scale_y(v, 0, 1.0, bot[1], bot[3]), f'{v:.1f}') for v in [0,0.25,0.5,0.75,1.0]]
draw_axes(d, *bot, xt2, yt2, 'Rate', 'Dense no-tag ScalingRL run: failure metrics')
for metric, color in [('ExecutorNativeFormatViolations', (214,39,40)), ('InvalidActionTerminationRate', (255,127,14)), ('PlannerInvalidFormatRate', (148,103,189))]:
    pts = series('dense_scaling_2agent', metric)
    mapped = [(scale_x(s, bot[0], bot[2], dense_max_step), scale_y(v, 0, 1.0, bot[1], bot[3])) for s, v in pts]
    if len(mapped) >= 2:
        d.line(mapped, fill=color, width=4)
    for x, y in mapped:
        d.ellipse((x-5, y-5, x+5, y+5), fill=color)
d.text((1020, 660), 'Exec format violations', fill=(214,39,40), font=SMALL)
d.text((1020, 690), 'Invalid action terminations', fill=(255,127,14), font=SMALL)
d.text((1020, 720), 'Planner invalid format', fill=(148,103,189), font=SMALL)
img.save(FIG / 'fig_dense500_detail.png')

# Figure 4: Tag removal and reviewer ablation
img = Image.new('RGB', (1500, 900), 'white')
d = ImageDraw.Draw(img)
plot = (110, 120, 980, 760)
xt = [(scale_x(v, plot[0], plot[2], comparison_max_step), v) for v in comparison_ticks if v <= comparison_max_step]
yt = [(scale_y(v, 0, 1.0, plot[1], plot[3]), f'{v:.1f}') for v in [0,0.25,0.5,0.75,1.0]]
draw_axes(d, *plot, xt, yt, 'Pass@1', 'Tag removal and reviewer ablation')
tag_compare_families = ['fixed_round_2agent_tagged','fixed_round_2agent_no_tags','dense_scaling_2agent','three_agent_executor_reviewer_scaling']
for fam in tag_compare_families:
    pts = series(fam, 'Pass@1')
    mapped = [(scale_x(s, plot[0], plot[2], comparison_max_step), scale_y(v, 0, 1.0, plot[1], plot[3])) for s, v in pts]
    if len(mapped) >= 2:
        d.line(mapped, fill=COLORS[fam], width=4)
    for x, y in mapped:
        d.ellipse((x-5, y-5, x+5, y+5), fill=COLORS[fam])
draw_legend(d, [(LABELS[f], COLORS[f]) for f in tag_compare_families], 1020, 180)
img.save(FIG / 'fig_tagged_vs_notags_scaling.png')

# ---------- Main markdown ----------
def rel(p):
    return str(p)

# summary stats
families = {}
for r in rows:
    fam = r['run_family']
    if fam == 'single_agent_baseline':
        continue
    families.setdefault(fam, []).append(r)
family_summary = {}
for fam, rr in families.items():
    best = max(rr, key=lambda x: -999 if x['Pass@1'] is None else x['Pass@1'])
    final = max(rr, key=lambda x: x['step'])
    family_summary[fam] = {'best': best, 'final': final}

main_md = REPORT / 'PAPER_HANDOFF_DEEP_RESEARCH.md'
with main_md.open('w') as f:
    f.write('# Multi-Agent BabyAI Collapse Investigation: Deep Research Handoff\n\n')
    f.write('Updated on 2026-04-03. This packet is designed so ChatGPT Deep Research can inspect the repo, reuse the exact source files, and draft a paper that pivots from “multi-agent improvement” to “collapse investigation”.\n\n')
    f.write('## Original AgentGym-RL Reference\n')
    f.write(f'- README: `{README}`\n')
    f.write(f'- Paper link from README: [{ARXIV_URL}]({ARXIV_URL})\n')
    f.write('- README states the original paper frames ScalingInter-RL as a curriculum for stable long-horizon RL training; this handoff compares that claim against the local BabyAI multi-agent runs.\n\n')
    f.write('## Recommended Paper Reframe\n')
    f.write('- The strongest defensible story is **not** that a final multi-agent checkpoint robustly outperformed the single-agent baseline.\n')
    f.write('- The strongest defensible story is that multi-agent decomposition produced **intermediate checkpoints that sometimes surpassed the single-agent baseline**, but the training process exhibited **multiple distinct collapse modes** depending on prompt regime and curriculum.\n')
    f.write('- The paper can therefore be framed as a collapse-study / stability-study of multi-agent RL on BabyAI, with concrete failure taxonomies and ablations.\n\n')
    f.write('## Source Inventory\n')
    f.write(f'- Run catalog: `{catalog_tsv}` and `{catalog_json}`\n')
    f.write(f'- Source manifest: `{manifest_tsv}`\n')
    f.write(f'- Selected traces: `{trace_md}` and `{trace_json}`\n')
    f.write(f'- Existing March 21 collapse report: `{MAVINO_DIR / "MAVINO_COLLAPSE_REPORT.md"}`\n')
    f.write(f'- Existing plain-split ScalingRL report: `{PLAIN100_DIR / "DEEP_RESEARCH_ANALYSIS.md"}`\n')
    f.write(f'- Existing dense-curriculum report: `{DENSE500_DIR / "SUMMARY.md"}`\n')
    f.write(f'- New 3-agent reviewer report: `{THREE_AGENT_DIR / "summary.md"}` and `{THREE_AGENT_DIR / "selected_trace_examples.md"}`\n\n')
    f.write('## Experimental Regimes To Emphasize\n')
    f.write('1. **2-agent tagged prompts**\n')
    f.write('   - Fixed-round 2-agent tagged runs (historical composite evidence).\n')
    f.write('2. **2-agent no-tag fixed-round**\n')
    f.write('   - Persistent fixed-round 20-round run with no visible planner/executor tags.\n')
    f.write('3. **2-agent no-tag ScalingRL**\n')
    f.write('   - Plain-split retry v2 with `[6,13,20]`.\n')
    f.write('   - Dense-curriculum run with `[6,8,10,13,16,20]`.\n')
    f.write('4. **3-agent no-tag ScalingRL with executor reviewer**\n')
    f.write('   - Planner, executor, and executor-reviewer share the same policy and are trained end-to-end.\n\n')
    f.write('## Figures\n')
    f.write(f'![Pass@1 comparison](figures/fig_pass1_comparison.png)\n\n')
    f.write(f'![Failure comparison](figures/fig_failure_comparison.png)\n\n')
    f.write(f'![Dense500 detail](figures/fig_dense500_detail.png)\n\n')
    f.write(f'![Tagged vs no-tags scaling](figures/fig_tagged_vs_notags_scaling.png)\n\n')
    f.write('## High-Level Quantitative Summary\n')
    f.write(f'- Single-agent baseline: `Avg@1={single_agent_baseline["Avg@1"]:.6f}`, `Pass@1={single_agent_baseline["Pass@1"]:.6f}`.\n')
    for fam in PRIMARY_COMPARE_FAMILIES:
        s = family_summary[fam]
        f.write(f'- {LABELS[fam]}: best checkpoint `step {s["best"]["step"]}` with `Pass@1={s["best"]["Pass@1"]:.6f}`, final checkpoint `step {s["final"]["step"]}` with `Pass@1={s["final"]["Pass@1"]:.6f}`.\n')
    f.write('\n')
    f.write('## Scope Note\n')
    f.write('- The figures in this handoff omit the older 4-agent reviewer experiment and instead use the current paper comparison set.\n')
    f.write('- The current comparison uses: fixed-round tagged 2-agent, fixed-round no-tag 2-agent, no-tag ScalingRL 2-agent, and no-tag ScalingRL 3-agent with executor reviewer.\n')
    f.write(f'- The newly added fixed-round no-tag run is sourced from `{FIXED2_NOTAG_DIR / "checkpoint_metrics.tsv"}` and `{FIXED2_NOTAG_DIR / "summary.md"}`.\n\n')
    f.write('## Supported Claims\n')
    f.write('1. **Prompt tags created a distinct recursive scaffold-copying collapse mode.**\n')
    f.write(f'   - Evidence: `{MAVINO_DIR / "MAVINO_COLLAPSE_REPORT.md"}` and `{MAVINO_DIR / "representative_trace_examples.md"}`.\n')
    f.write('   - Earliest visible leak at `step 55`; terminal recursive contamination by `step 98-100`.\n')
    f.write('2. **Removing visible tags removed that specific failure mode.**\n')
    f.write(f'   - Evidence: `{PLAIN100_DIR / "DEEP_RESEARCH_ANALYSIS.md"}` and `{DENSE500_DIR / "SUMMARY.md"}`.\n')
    f.write('   - In the no-tag runs, `PlannerTagOnlyRate` stays at or near zero, and the late collapse is no longer driven by bracketed scaffold leakage.\n')
    f.write('3. **Removing tags alone is not sufficient to stabilize fixed-round 2-agent training.**\n')
    f.write(f'   - Evidence: `{FIXED2_NOTAG_DIR / "checkpoint_metrics.tsv"}`.\n')
    f.write('   - The new fixed-round no-tag run peaks at `step 100` with `Pass@1=0.677778`, then fully collapses from `step 400` onward with `PlannerTagOnlyRate=1.0` and `ExecutorNativeFormatViolations=1.0`.\n')
    f.write('4. **No-tag multi-agent ScalingRL runs can produce checkpoints that beat the single-agent baseline, but the gain is not robust through training.**\n')
    f.write('   - Coarse no-tag ScalingRL `[6,13,20]`: best `step 200`, `Pass@1=0.811111`.\n')
    f.write('   - Dense no-tag ScalingRL `[6,8,10,13,16,20]`: best `step 300`, `Pass@1=0.822222`.\n')
    f.write('   - Both runs collapse later (`450-500`).\n')
    f.write('5. **Adding an executor reviewer delays but does not eliminate collapse.**\n')
    f.write(f'   - Evidence: `{THREE_AGENT_DIR / "checkpoint_metrics.tsv"}` and `{THREE_AGENT_DIR / "selected_trace_examples.md"}`.\n')
    f.write('   - The 3-agent reviewer run peaks at `step 100` (`Pass@1=0.722222`), recovers again at `step 350` (`Pass@1=0.711111`), but still enters full collapse by `step 450` with both planner and executor metrics saturated.\n')
    f.write('6. **The dominant late no-tag collapse in ScalingRL remains an interaction failure between schema control and role outputs.**\n')
    f.write('   - In the 2-agent no-tag runs, late failure is mostly executor-side.\n')
    f.write('   - In the 3-agent reviewer run, the reviewer itself becomes part of the collapse: reviewer schema leaks into executor outputs, and malformed reviewer outputs trigger retries or terminations.\n\n')
    f.write('## Important Caveats\n')
    f.write('- The fixed-round 2-agent curve is a **composite evidence pool**, not a single uninterrupted run. Use it to describe the historical debugging trajectory, not as a single clean learning curve.\n')
    f.write('- The fixed-round 2-agent no-tag run is now present, but it is still only one run; treat it as an ablation datapoint, not a complete stability proof.\n')
    f.write('- The strongest paper claims should therefore focus on **failure modes and stability**, not on a single winner number.\n\n')
    f.write('## Failure Taxonomy\n')
    f.write('### A. Tagged prompt scaffold collapse\n')
    f.write('- Visible role headers such as `[Planner Turn]` and `[Executor Turn]` become copyable tokens.\n')
    f.write('- Planner leaks prompt scaffolding first.\n')
    f.write('- Executor begins copying leaked scaffolding instead of producing BabyAI-native `Thought:` / `Action:` responses.\n')
    f.write('- Best artifact: `tagged_step55_first_header_leak` and `tagged_step100_recursive_scaffold` in `selected_trace_packets.md`.\n\n')
    f.write('### B. Fixed-round planner verbosity/fallback collapse\n')
    f.write('- In the early fixed-round 2-agent regime, planner outputs drift long before total collapse.\n')
    f.write('- The critical onset is `too_long` planner outputs around `step 45`, which trigger generic fallback context.\n')
    f.write('- Best artifact: `fixed_round_step45_planner_too_long` in `selected_trace_packets.md`.\n\n')
    f.write('### C. No-tag ScalingRL late executor collapse\n')
    f.write('- Removing tags does not eliminate collapse; it changes its form.\n')
    f.write('- Mid-run performance is strong, but late in training the executor either invents invalid actions or stops emitting the required schema.\n')
    f.write('- Coarse no-tag run: degradation begins after `300`, with total schema collapse at `450-500`.\n')
    f.write('- Dense no-tag run: best checkpoint at `300`, same terminal executor-format collapse by `450-500`.\n\n')
    f.write('### D. Fixed-round no-tag late total collapse\n')
    f.write('- The newly added fixed-round no-tag run shows that removing visible tags alone is not enough.\n')
    f.write('- This run is healthy through `step 350`, then flips into total failure at `step 400` with `Pass@1=0.0`, `ExecutorNativeFormatViolations=1.0`, `PlannerInvalidFormatRate=1.0`, and `PlannerTagOnlyRate=1.0`.\n')
    f.write('- This matters because it shows a no-tag fixed-round regime can still collapse without the original tagged scaffold-copy mechanism.\n\n')
    f.write('### E. 3-agent reviewer schema interference collapse\n')
    f.write('- The executor reviewer helps some early checkpoints, but it introduces a new route for schema contamination.\n')
    f.write('- At `step 150`, the reviewer already misfires: it requests retry even when deterministic executor validation can still extract a valid action.\n')
    f.write('- At `step 350`, the run regains strong task success, but reviewer schema (`Verdict:` / `Reason:`) starts leaking into executor outputs.\n')
    f.write('- At `step 400`, planner invalid-format rates rise sharply while the reviewer still often passes executor outputs.\n')
    f.write('- By `step 450`, both planner and reviewer channels have collapsed into tag-only or garbage outputs, and the executor fails with `invalid_format` on every sampled case.\n\n')
    f.write('## Collapse Questions\n')
    f.write('### When does collapse begin?\n')
    f.write('- For the 3-agent reviewer run, the **first warning signs** appear by `step 150`, where reviewer outputs become malformed and can trigger retries or termination even when the executor action is otherwise valid.\n')
    f.write('- The run then **recovers** through `step 250-350`.\n')
    f.write('- The **terminal collapse onset** begins at `step 400`, where `PlannerInvalidFormatRate` jumps to `0.700000` and trace examples show planner schema contamination.\n')
    f.write('- The run becomes **fully collapsed** at `step 450`, where `Pass@1=0.0`, `ExecutorNativeFormatViolations=1.0`, `PlannerInvalidFormatRate=1.0`, and `PlannerTagOnlyRate=1.0`.\n\n')
    f.write('### How do we detect that collapse has started?\n')
    f.write('- Watch for a combination of metrics and traces:\n')
    f.write('  - rising `PlannerInvalidFormatRate`\n')
    f.write('  - rising `PlannerTagOnlyRate`\n')
    f.write('  - rising `ExecutorNativeFormatViolations`\n')
    f.write('  - reviewer outputs that stop following their own `Verdict/Reason` schema\n')
    f.write('  - executor outputs that begin to include reviewer schema or punctuation spam\n')
    f.write('- In the 3-agent run, `step 350` already shows reviewer-schema leakage inside the executor output even though task success is still high.\n\n')
    f.write('### Can the run recover after instability begins?\n')
    f.write('- **Yes, partially.** The 3-agent run dips badly at `150-200` and then recovers to strong checkpoints at `300-350`.\n')
    f.write('- **No, not after terminal collapse has fully set in.** Once the run reaches the `450+` regime, there is no evidence of spontaneous recovery in the remaining training horizon.\n\n')
    f.write('### How do we prevent it?\n')
    f.write('- The current evidence suggests that prompt simplification alone is not enough.\n')
    f.write('- The most promising direction is stronger schema control on every role, especially on the reviewer, plus explicit monitoring for reviewer-schema leakage into the executor channel.\n')
    f.write('- A practical prevention stack from these runs would include: bounded retries, low-variance reviewer prompts, early-stop or checkpoint selection based on collapse metrics, and curriculum schedules that avoid pushing far past the last stable checkpoint.\n\n')
    f.write('## Trace Guide For Deep Research\n')
    f.write('- Tagged prompt leak onset: `selected_trace_packets.md` -> `tagged_step55_first_header_leak`\n')
    f.write('- Tagged terminal recursive contamination: `selected_trace_packets.md` -> `tagged_step100_recursive_scaffold`\n')
    f.write('- Fixed-round verbosity onset: `selected_trace_packets.md` -> `fixed_round_step45_planner_too_long`\n')
    f.write('- Fixed-round stable example after stabilization: `selected_trace_packets.md` -> `fixed_round_step100_v2_stable`\n')
    f.write('- No-tag coarse scaling peak: `selected_trace_packets.md` -> `plain_split_200_valid`\n')
    f.write('- No-tag coarse scaling late invalid-action drift: `selected_trace_packets.md` -> `plain_split_400_invalid_action`\n')
    f.write('- No-tag coarse scaling terminal collapse: `selected_trace_packets.md` -> `plain_split_450_invalid_format`\n')
    f.write('- Dense no-tag peak: `selected_trace_packets.md` -> `dense500_step300_peak_valid_example`\n')
    f.write('- Dense no-tag transition: `selected_trace_packets.md` -> `dense500_step400_transition_example`\n')
    f.write('- Dense no-tag retry exhaustion: `selected_trace_packets.md` -> `dense500_step450_retry_exhaustion_example`\n')
    f.write('- Dense no-tag terminal collapse: `selected_trace_packets.md` -> `dense500_step500_terminal_collapse_example`\n\n')
    f.write('- 3-agent reviewer peak: `selected_trace_packets.md` -> `three_agent_step100_healthy`\n')
    f.write('- 3-agent reviewer early instability: `selected_trace_packets.md` -> `three_agent_step150_reviewer_false_retry`\n')
    f.write('- 3-agent reviewer schema leakage: `selected_trace_packets.md` -> `three_agent_step350_schema_leak`\n')
    f.write('- 3-agent reviewer collapse onset: `selected_trace_packets.md` -> `three_agent_step400_onset`\n')
    f.write('- 3-agent reviewer terminal collapse: `selected_trace_packets.md` -> `three_agent_step450_terminal`\n\n')
    f.write('## Suggested Paper Structure\n')
    f.write('1. Introduction: goal was to improve BabyAI long-horizon performance via multi-agent decomposition.\n')
    f.write('2. Setup: single-agent baseline, planner/executor design, executor-reviewer extension, ScalingInter-RL.\n')
    f.write('3. Main result: intermediate multi-agent checkpoints can outperform baseline, but training is unstable and reviewer-based corrections do not eliminate collapse.\n')
    f.write('4. Collapse taxonomy:\n')
    f.write('   - tagged scaffold-copying collapse\n')
    f.write('   - fixed-round planner verbosity/fallback collapse\n')
    f.write('   - no-tag late executor schema collapse\n')
    f.write('5. Ablation discussion:\n')
    f.write('   - what removing tags fixed\n')
    f.write('   - what ScalingRL improved\n')
    f.write('   - what the executor reviewer improved\n')
    f.write('   - why denser curricula and reviewer feedback still did not prevent terminal collapse\n')
    f.write('6. Conclusion: multi-agent RL can transiently help, but stable coordination and schema retention remain unresolved.\n\n')
    f.write('## Exact Files To Read First\n')
    f.write(f'1. `{main_md}`\n')
    f.write(f'2. `{catalog_tsv}`\n')
    f.write(f'3. `{trace_md}`\n')
    f.write(f'4. `{MAVINO_DIR / "MAVINO_COLLAPSE_REPORT.md"}`\n')
    f.write(f'5. `{PLAIN100_DIR / "DEEP_RESEARCH_ANALYSIS.md"}`\n')
    f.write(f'6. `{DENSE500_DIR / "SUMMARY.md"}`\n')
    f.write(f'7. `{THREE_AGENT_DIR / "summary.md"}`\n')
    f.write(f'8. `{THREE_AGENT_DIR / "selected_trace_examples.md"}`\n')
    f.write(f'9. `{PROMPT_ALIGN}`\n')

print(main_md)
