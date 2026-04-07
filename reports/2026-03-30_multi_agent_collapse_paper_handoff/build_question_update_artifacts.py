from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO = Path("/Users/mavinomichael/PycharmProjects/AgentGym-RL")
HANDOFF = REPO / "reports" / "2026-03-30_multi_agent_collapse_paper_handoff"
FIG_DIR = HANDOFF / "figures"
RUN_CATALOG = HANDOFF / "run_catalog.tsv"


REGIME_ANNOTATIONS = [
    {
        "run_family": "tagged_scaling_2agent",
        "label": "BabyAI 2-agent tagged ScalingRL",
        "environment": "BabyAI",
        "total_steps": 100,
        "warning_step": 55,
        "terminal_onset_step": 98,
        "full_collapse_step": 100,
        "recovery_start_step": None,
        "recovery_end_step": None,
        "recovery_observed": False,
        "dominant_failure_mode": "planner scaffold leakage into executor outputs",
        "evidence_traces": "tagged_step55_first_header_leak; tagged_step100_recursive_scaffold",
        "evidence_notes": "Trace packet shows planner copying role headers by step 55 and recursive scaffold contamination by step 100.",
    },
    {
        "run_family": "fixed_round_2agent_no_tags",
        "label": "BabyAI 2-agent fixed-round no tags",
        "environment": "BabyAI",
        "total_steps": 600,
        "warning_step": 400,
        "terminal_onset_step": 400,
        "full_collapse_step": 400,
        "recovery_start_step": None,
        "recovery_end_step": None,
        "recovery_observed": False,
        "dominant_failure_mode": "late total planner+executor format collapse",
        "evidence_traces": "fixed_round_no_tag_metrics_step400",
        "evidence_notes": "Metrics flip from healthy at step 350 to full invalid-format collapse at step 400.",
    },
    {
        "run_family": "plain_split_scaling_2agent",
        "label": "BabyAI 2-agent no-tag ScalingRL [6,13,20]",
        "environment": "BabyAI",
        "total_steps": 500,
        "warning_step": 400,
        "terminal_onset_step": 400,
        "full_collapse_step": 450,
        "recovery_start_step": None,
        "recovery_end_step": None,
        "recovery_observed": False,
        "dominant_failure_mode": "late executor invalid-action then invalid-format collapse",
        "evidence_traces": "plain_split_400_invalid_action; plain_split_450_invalid_format",
        "evidence_notes": "Metrics stay strong through step 300, degrade at step 400, and hit full executor-format collapse by step 450.",
    },
    {
        "run_family": "dense_scaling_2agent",
        "label": "BabyAI 2-agent no-tag ScalingRL [6,8,10,13,16,20]",
        "environment": "BabyAI",
        "total_steps": 500,
        "warning_step": 400,
        "terminal_onset_step": 400,
        "full_collapse_step": 450,
        "recovery_start_step": None,
        "recovery_end_step": None,
        "recovery_observed": False,
        "dominant_failure_mode": "late executor-side invalid-format saturation",
        "evidence_traces": "dense500_step400_transition_example; dense500_step450_retry_exhaustion_example; dense500_step500_terminal_collapse_example",
        "evidence_notes": "Step 300 is the peak, step 400 is the transition, and steps 450-500 are near-total executor-format failure with planner still valid.",
    },
    {
        "run_family": "three_agent_executor_reviewer_scaling",
        "label": "BabyAI 3-agent no-tag + reviewer",
        "environment": "BabyAI",
        "total_steps": 600,
        "warning_step": 150,
        "terminal_onset_step": 400,
        "full_collapse_step": 450,
        "recovery_start_step": 250,
        "recovery_end_step": 350,
        "recovery_observed": True,
        "dominant_failure_mode": "reviewer/executor schema disagreement, then three-channel contamination",
        "evidence_traces": "three_agent_step150_reviewer_false_retry; three_agent_step350_schema_leak; three_agent_step400_onset; three_agent_step450_terminal",
        "evidence_notes": "The run crashes at step 150, recovers through steps 250-350, then enters terminal planner+executor+reviewer collapse by step 450.",
    },
    {
        "run_family": "webarena_clean_2agent_scaling",
        "label": "WebArena 2-agent no-tag clean",
        "environment": "WebArena",
        "total_steps": 600,
        "warning_step": 2,
        "terminal_onset_step": 3,
        "full_collapse_step": 9,
        "recovery_start_step": None,
        "recovery_end_step": None,
        "recovery_observed": False,
        "dominant_failure_mode": "near-immediate planner tag-only collapse and executor invalid-format",
        "evidence_traces": "webarena_clean_step1_valid; webarena_clean_step3_tag_only_collapse; webarena_clean_step553_terminal",
        "evidence_notes": "Only step 1 is clearly healthy; traces show warning signs by step 2 and a fully collapsed batch by step 9.",
    },
]

SHORT_LABELS = {
    "tagged_scaling_2agent": "2A tagged",
    "fixed_round_2agent_no_tags": "2A fixed no-tag",
    "plain_split_scaling_2agent": "2A scale coarse",
    "dense_scaling_2agent": "2A scale dense",
    "three_agent_executor_reviewer_scaling": "3A reviewer",
    "webarena_clean_2agent_scaling": "WebArena 2A",
}


def ensure_dirs() -> None:
    HANDOFF.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_font(size: int, *, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE_FONT = load_font(34, bold=True)
LABEL_FONT = load_font(22)
SMALL_FONT = load_font(18)


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"NA", "None"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_run_catalog() -> list[dict]:
    with RUN_CATALOG.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        row["step"] = int(row["step"])
        for key in [
            "Avg@1",
            "Pass@1",
            "ExecutorNativeFormatViolations",
            "InvalidFormatTerminationRate",
            "InvalidActionTerminationRate",
            "PlannerInvalidFormatRate",
            "PlannerFallbackRate",
            "PlannerTagOnlyRate",
        ]:
            row[key] = to_float(row.get(key))
    return rows


def best_pass(rows: list[dict], *, min_step: int | None = None, max_step: int | None = None) -> tuple[int | None, float | None]:
    filtered = []
    for row in rows:
        step = row["step"]
        if min_step is not None and step < min_step:
            continue
        if max_step is not None and step > max_step:
            continue
        if row["Pass@1"] is None:
            continue
        filtered.append(row)
    if not filtered:
        return None, None
    best_row = max(filtered, key=lambda row: row["Pass@1"])
    return best_row["step"], best_row["Pass@1"]


def summarize_regimes(rows: list[dict]) -> list[dict]:
    by_family: dict[str, list[dict]] = {}
    for row in rows:
        by_family.setdefault(row["run_family"], []).append(row)

    summaries: list[dict] = []
    for annotation in REGIME_ANNOTATIONS:
        family_rows = sorted(by_family.get(annotation["run_family"], []), key=lambda row: row["step"])
        overall_best_step, overall_best_pass = best_pass(family_rows)
        pre_warning_best_step, pre_warning_best_pass = best_pass(
            family_rows,
            max_step=annotation["warning_step"] - 1 if annotation["warning_step"] is not None else None,
        )
        post_warning_best_step, post_warning_best_pass = best_pass(
            family_rows,
            min_step=annotation["warning_step"],
            max_step=annotation["full_collapse_step"] - 1 if annotation["full_collapse_step"] is not None else None,
        )
        final_row = family_rows[-1] if family_rows else None

        summaries.append(
            {
                **annotation,
                "overall_best_step": overall_best_step,
                "overall_best_pass": overall_best_pass,
                "pre_warning_best_step": pre_warning_best_step,
                "pre_warning_best_pass": pre_warning_best_pass,
                "post_warning_best_step": post_warning_best_step,
                "post_warning_best_pass": post_warning_best_pass,
                "final_step": final_row["step"] if final_row else None,
                "final_pass": final_row["Pass@1"] if final_row else None,
                "final_executor_invalid_format": final_row["ExecutorNativeFormatViolations"] if final_row else None,
                "final_planner_invalid_format": final_row["PlannerInvalidFormatRate"] if final_row else None,
                "final_planner_tag_only": final_row["PlannerTagOnlyRate"] if final_row else None,
            }
        )
    return summaries


def write_summary_files(summary_rows: list[dict]) -> None:
    summary_json = HANDOFF / "collapse_regime_summary.json"
    summary_tsv = HANDOFF / "collapse_regime_summary.tsv"
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    fieldnames = [
        "run_family",
        "label",
        "environment",
        "total_steps",
        "warning_step",
        "terminal_onset_step",
        "full_collapse_step",
        "recovery_start_step",
        "recovery_end_step",
        "recovery_observed",
        "overall_best_step",
        "overall_best_pass",
        "pre_warning_best_step",
        "pre_warning_best_pass",
        "post_warning_best_step",
        "post_warning_best_pass",
        "final_step",
        "final_pass",
        "final_executor_invalid_format",
        "final_planner_invalid_format",
        "final_planner_tag_only",
        "dominant_failure_mode",
        "evidence_traces",
        "evidence_notes",
    ]
    with summary_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)


def build_collapse_timeline(summary_rows: list[dict]) -> None:
    width, height = 1800, 900
    left, right = 470, 1680
    top, bottom = 120, 780
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    colors = {"BabyAI": "#1f77b4", "WebArena": "#d62728"}

    def x_pos(percent: float) -> float:
        return left + (percent / 100.0) * (right - left)

    draw.text((60, 35), "Collapse timing across BabyAI and WebArena regimes", fill="black", font=TITLE_FONT)
    draw.text((60, 78), "Warning = first clear instability, onset = start of terminal regime, full = fully collapsed sampled/eval state", fill="#555555", font=SMALL_FONT)

    row_gap = (bottom - top) / max(1, len(summary_rows) - 1)
    for tick in range(0, 101, 10):
        x = x_pos(float(tick))
        draw.line((x, top - 10, x, bottom + 20), fill="#e6e6e6", width=1)
        draw.text((x - 14, bottom + 30), str(tick), fill="black", font=SMALL_FONT)
    draw.text((left + 280, bottom + 62), "Training horizon consumed before warning / collapse (%)", fill="black", font=LABEL_FONT)

    legend_y = bottom + 12
    draw.ellipse((60, legend_y, 84, legend_y + 24), fill="#ff9f1c", outline="#ff9f1c")
    draw.text((94, legend_y - 2), "First warning", fill="black", font=SMALL_FONT)
    tri_x = 250
    draw.polygon([(tri_x + 12, legend_y), (tri_x, legend_y + 24), (tri_x + 24, legend_y + 24)], fill="#1f77b4")
    draw.text((284, legend_y - 2), "Terminal onset", fill="black", font=SMALL_FONT)
    x0 = 470
    draw.line((x0, legend_y, x0 + 24, legend_y + 24), fill="#111111", width=4)
    draw.line((x0 + 24, legend_y, x0, legend_y + 24), fill="#111111", width=4)
    draw.text((506, legend_y - 2), "Full collapse", fill="black", font=SMALL_FONT)
    draw.rectangle((685, legend_y, 715, legend_y + 24), fill="#9bd18b", outline="#9bd18b")
    draw.text((726, legend_y - 2), "Observed recovery band", fill="black", font=SMALL_FONT)

    for idx, row in enumerate(summary_rows):
        y = top + idx * row_gap
        env_color = colors[row["environment"]]
        draw.line((left, y, right, y), fill="#cfcfcf", width=3)
        draw.text((40, y - 28), row["label"], fill="black", font=LABEL_FONT)
        draw.text((40, y - 2), row["environment"], fill=env_color, font=SMALL_FONT)

        if row["recovery_observed"] and row["recovery_start_step"] and row["recovery_end_step"]:
            start_pct = 100.0 * row["recovery_start_step"] / row["total_steps"]
            end_pct = 100.0 * row["recovery_end_step"] / row["total_steps"]
            draw.rectangle((x_pos(start_pct), y - 14, x_pos(end_pct), y + 14), fill="#9bd18b", outline="#5c9f4f")
            draw.text((x_pos(end_pct) + 8, y - 11), f"recovery {row['recovery_start_step']}-{row['recovery_end_step']}", fill="#2d6a1f", font=SMALL_FONT)

        warning_pct = 100.0 * row["warning_step"] / row["total_steps"]
        onset_pct = 100.0 * row["terminal_onset_step"] / row["total_steps"]
        full_pct = 100.0 * row["full_collapse_step"] / row["total_steps"]

        wx = x_pos(warning_pct)
        ox = x_pos(onset_pct)
        fx = x_pos(full_pct)

        draw.ellipse((wx - 10, y - 10, wx + 10, y + 10), fill="#ff9f1c", outline="#b56b00")
        draw.polygon([(ox, y - 12), (ox - 12, y + 10), (ox + 12, y + 10)], fill=env_color)
        draw.line((fx - 10, y - 10, fx + 10, y + 10), fill="#111111", width=4)
        draw.line((fx + 10, y - 10, fx - 10, y + 10), fill="#111111", width=4)

        draw.text((wx + 14, y - 28), f"warn {row['warning_step']}", fill="#8a5a00", font=SMALL_FONT)
        draw.text((ox + 14, y - 3), f"onset {row['terminal_onset_step']}", fill=env_color, font=SMALL_FONT)
        draw.text((fx + 14, y + 18), f"full {row['full_collapse_step']}", fill="#111111", font=SMALL_FONT)

    image.save(FIG_DIR / "fig_collapse_timeline.png")


def build_recovery_summary(summary_rows: list[dict]) -> None:
    width, height = 1900, 950
    left, right = 110, 1770
    top, bottom = 120, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((60, 35), "Recovery after early warning is rare and disappears after full collapse", fill="black", font=TITLE_FONT)
    draw.text((60, 78), "Bars compare best checkpoint before the first warning, best checkpoint after warning, and the final checkpoint", fill="#555555", font=SMALL_FONT)

    n = len(summary_rows)
    group_width = (right - left) / max(1, n)
    bar_width = group_width * 0.18
    baseline_y = bottom

    def y_pos(value: float) -> float:
        return baseline_y - (value / 0.9) * (bottom - top)

    for tick in [0.0, 0.2, 0.4, 0.6, 0.8]:
        y = y_pos(tick)
        draw.line((left - 10, y, right, y), fill="#e6e6e6", width=1)
        draw.text((25, y - 10), f"{tick:.1f}", fill="black", font=SMALL_FONT)
    draw.text((15, top - 32), "Pass@1", fill="black", font=LABEL_FONT)

    colors = {
        "pre": "#4c78a8",
        "post": "#59a14f",
        "final": "#e15759",
    }

    legend_x = 1110
    legend_y = 72
    for idx, (name, color) in enumerate([("Best before warning", colors["pre"]), ("Best after warning", colors["post"]), ("Final checkpoint", colors["final"])]):
        y = legend_y + idx * 28
        draw.rectangle((legend_x, y, legend_x + 24, y + 18), fill=color, outline=color)
        draw.text((legend_x + 36, y - 2), name, fill="black", font=SMALL_FONT)

    for idx, row in enumerate(summary_rows):
        center = left + idx * group_width + group_width * 0.5
        pre_value = row["pre_warning_best_pass"] or 0.0
        post_value = row["post_warning_best_pass"] or 0.0
        final_value = row["final_pass"] or 0.0

        bars = [
            (center - bar_width * 1.4, pre_value, colors["pre"]),
            (center - bar_width * 0.2, post_value, colors["post"]),
            (center + bar_width, final_value, colors["final"]),
        ]
        for x, value, color in bars:
            draw.rectangle((x, y_pos(value), x + bar_width, baseline_y), fill=color, outline=color)

        label = SHORT_LABELS.get(row["run_family"], row["label"])
        draw.text((center - group_width * 0.18, bottom + 15), label, fill="black", font=SMALL_FONT)
        if row["recovery_observed"]:
            draw.text((center - group_width * 0.22, y_pos(max(pre_value, post_value)) - 28), "partial recovery", fill="#2d6a1f", font=SMALL_FONT)
        elif row["environment"] == "WebArena":
            draw.text((center - group_width * 0.12, y_pos(max(pre_value, post_value) + 0.03) - 10), "no recovery", fill="#7a1c1c", font=SMALL_FONT)

    image.save(FIG_DIR / "fig_recovery_summary.png")


def main() -> None:
    ensure_dirs()
    rows = load_run_catalog()
    summary_rows = summarize_regimes(rows)
    write_summary_files(summary_rows)
    build_collapse_timeline(summary_rows)
    build_recovery_summary(summary_rows)
    print(HANDOFF / "collapse_regime_summary.tsv")


if __name__ == "__main__":
    main()
