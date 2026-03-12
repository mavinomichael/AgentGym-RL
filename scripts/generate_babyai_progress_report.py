#!/usr/bin/env python3
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "reports" / "babyai_multi_agent_diagnostics_2026-03-09"
SEARCH_ROOTS = [
    ROOT,
    REPO_ROOT / "reports" / "babyai_multi_agent_diagnostics_2026-03-10",
    REPO_ROOT / "reports" / "babyai_multi_agent_diagnostics_2026-03-11",
]
LOG_ROOT = Path("/mnt/data/logs")


@dataclass
class RunMetrics:
    name: str
    log_name: str
    log_path: Path
    avg_at_1: float
    pass_at_1: float
    executor_native_format_violations: float
    invalid_format_termination_rate: float
    invalid_action_termination_rate: float
    planner_invalid_format_rate: float
    planner_fallback_rate: float
    planner_tag_only_rate: float
    note: str


def parse_metric(text: str, name: str) -> float:
    match = re.search(rf"{re.escape(name)}:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
    if match is None:
        return float("nan")
    return float(match.group(1))


def resolve_log_path(log_name: str) -> Path:
    direct_candidates = []
    for base in SEARCH_ROOTS:
        direct_candidates.append(base / log_name)
    direct_candidates.append(LOG_ROOT / log_name)
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    recursive_roots = [base for base in SEARCH_ROOTS if base.exists()] + [LOG_ROOT]
    for base in recursive_roots:
        if not base.exists():
            continue
        matches = sorted(base.glob(f"**/{log_name}"))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Unable to locate log file: {log_name}")


def parse_run(name: str, log_name: str, note: str) -> RunMetrics:
    log_path = resolve_log_path(log_name)
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    return RunMetrics(
        name=name,
        log_name=log_name,
        log_path=log_path,
        avg_at_1=parse_metric(text, "Avg@1"),
        pass_at_1=parse_metric(text, "Pass@1"),
        executor_native_format_violations=parse_metric(text, "ExecutorNativeFormatViolations"),
        invalid_format_termination_rate=parse_metric(text, "InvalidFormatTerminationRate"),
        invalid_action_termination_rate=parse_metric(text, "InvalidActionTerminationRate"),
        planner_invalid_format_rate=parse_metric(text, "PlannerInvalidFormatRate"),
        planner_fallback_rate=parse_metric(text, "PlannerFallbackRate"),
        planner_tag_only_rate=parse_metric(text, "PlannerTagOnlyRate"),
        note=note,
    )


def fmt(value: float) -> str:
    return "NA" if math.isnan(value) else f"{value:.6f}"


def main() -> int:
    runs = [
        parse_run(
            "700-step failed eval",
            "eval_babyai_multi_clean.log",
            "Initial multi-agent training collapsed; executor outputs were not reaching BabyAI in valid form.",
        ),
        parse_run(
            "Base model eval",
            "eval_babyai_base_trace.log",
            "Reference checkpoint showing the task and runtime can work before RL drift.",
        ),
        parse_run(
            "Retrain step-100",
            "eval_stage100_retry.log",
            "Early retrain improved over total collapse but still had major executor format failures.",
        ),
        parse_run(
            "Sanity step-50",
            "eval_stage50_sanity.log",
            "Checkpoint trained before planner-history cleanup; planner fallback remained the main failure mode.",
        ),
        parse_run(
            "Sanity step-15",
            "eval_stage15_sanity.log",
            "Checkpoint trained after planner stabilization and prompt-history cleanup.",
        ),
        parse_run(
            "Diagnostic step-100 v2",
            "eval_step100_diagnostic.log",
            "Fresh 100-step diagnostic run after Ray temp fixes, planner retries, lower planner token budget, and planner-weighted KL anchoring.",
        ),
    ]

    resume_candidates = [
        (
            "Resume step-236",
            "eval_stage236_resume.log",
            "Long resumed run from the stabilized step-15 checkpoint. Late in training, planner validation collapsed again, forcing fallback on nearly every turn.",
        ),
        (
            "Resume step-350",
            "eval_stage350_resume.log",
            "Continuation run from the resumed checkpoint. By step 350, both planner and executor had collapsed again and the eval fell to the floor reward on every item.",
        ),
        (
            "Resume step-235",
            "eval_stage235_resume.log",
            "Long resumed run from the stabilized step-15 checkpoint with end-only evaluation at step 235.",
        ),
    ]
    for run_name, log_name, note in resume_candidates:
        if any(path.exists() for path in (ROOT / log_name, LOG_ROOT / log_name)):
            runs.append(parse_run(run_name, log_name, note))

    report_path = ROOT / "babyai_eval_comparison_report.txt"
    png_path = ROOT / "babyai_eval_comparison.png"
    ROOT.mkdir(parents=True, exist_ok=True)

    base = next(run for run in runs if run.name == "Base model eval")
    latest = runs[-1]

    with report_path.open("w", encoding="utf-8") as f:
        f.write("BabyAI Multi-Agent Progress Report\n")
        f.write(f"Date: {date.today().isoformat()}\n\n")
        f.write("Context\n")
        f.write("- This report combines historical eval logs with the latest resumed checkpoints when available.\n")
        f.write("- Later evals were run with the stabilized multi-agent runtime. Historical numbers are preserved from their original logs.\n\n")
        f.write("Key Takeaways\n")
        f.write(f"- Base model reference: Avg@1={fmt(base.avg_at_1)}, Pass@1={fmt(base.pass_at_1)}.\n")
        f.write(f"- Initial 700-step RL run failed completely: Avg@1={fmt(runs[0].avg_at_1)}, Pass@1={fmt(runs[0].pass_at_1)}.\n")
        f.write(f"- Step-100 retrain partially recovered performance but still had executor format failures: Avg@1={fmt(runs[2].avg_at_1)}, Pass@1={fmt(runs[2].pass_at_1)}, ExecutorNativeFormatViolations={fmt(runs[2].executor_native_format_violations)}.\n")
        f.write(f"- Step-50 sanity eval exposed the planner-side failure clearly: Avg@1={fmt(runs[3].avg_at_1)}, Pass@1={fmt(runs[3].pass_at_1)}, PlannerInvalidFormatRate={fmt(runs[3].planner_invalid_format_rate)}, PlannerFallbackRate={fmt(runs[3].planner_fallback_rate)}.\n")
        f.write(f"- Diagnostic step-100 v2 completed cleanly with near-zero planner/executor failures, but performance remained below the base-model reference: Avg@1={fmt(runs[5].avg_at_1)}, Pass@1={fmt(runs[5].pass_at_1)}, PlannerInvalidFormatRate={fmt(runs[5].planner_invalid_format_rate)}, InvalidActionTerminationRate={fmt(runs[5].invalid_action_termination_rate)}.\n")
        f.write(
            f"- Latest evaluated checkpoint ({latest.name}) reports Avg@1={fmt(latest.avg_at_1)}, "
            f"Pass@1={fmt(latest.pass_at_1)}, ExecutorNativeFormatViolations={fmt(latest.executor_native_format_violations)}, "
            f"PlannerInvalidFormatRate={fmt(latest.planner_invalid_format_rate)}.\n\n"
        )

        f.write("Run Metrics\n")
        f.write(
            "Run\tAvg@1\tPass@1\tExecutorNativeFormatViolations\tInvalidFormatTerminationRate\tInvalidActionTerminationRate\tPlannerInvalidFormatRate\tPlannerFallbackRate\tPlannerTagOnlyRate\n"
        )
        for run in runs:
            f.write(
                f"{run.name}\t{fmt(run.avg_at_1)}\t{fmt(run.pass_at_1)}\t"
                f"{fmt(run.executor_native_format_violations)}\t{fmt(run.invalid_format_termination_rate)}\t"
                f"{fmt(run.invalid_action_termination_rate)}\t{fmt(run.planner_invalid_format_rate)}\t"
                f"{fmt(run.planner_fallback_rate)}\t{fmt(run.planner_tag_only_rate)}\n"
            )

        f.write("\nInterpretation By Run\n")
        for run in runs:
            f.write(f"- {run.name}: {run.note} (source: {run.log_path})\n")

        f.write("\nDelta vs Base Model\n")
        for run in runs:
            if run.name == "Base model eval":
                continue
            f.write(f"\n{run.name}\n")
            f.write(f"- Avg@1 delta: {fmt(run.avg_at_1 - base.avg_at_1) if not math.isnan(run.avg_at_1) else 'NA'}\n")
            f.write(f"- Pass@1 delta: {fmt(run.pass_at_1 - base.pass_at_1) if not math.isnan(run.pass_at_1) else 'NA'}\n")
            if not math.isnan(run.executor_native_format_violations):
                f.write(
                    "- ExecutorNativeFormatViolations: "
                    f"{fmt(run.executor_native_format_violations)}\n"
                )
            if not math.isnan(run.planner_invalid_format_rate):
                f.write(
                    "- PlannerInvalidFormatRate: "
                    f"{fmt(run.planner_invalid_format_rate)}\n"
                )
                f.write(
                    "- PlannerFallbackRate: "
                    f"{fmt(run.planner_fallback_rate)}\n"
                )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = []
        for run in runs:
            lowered = run.name.lower()
            if "700" in lowered:
                labels.append("700")
            elif "base" in lowered:
                labels.append("Base")
            elif "diagnostic" in lowered and "100" in lowered:
                labels.append("100v2")
            elif "100" in lowered:
                labels.append("100")
            elif "350" in lowered:
                labels.append("350")
            elif "50" in lowered:
                labels.append("50")
            elif "15" in lowered:
                labels.append("15")
            elif "236" in lowered:
                labels.append("236")
            elif "235" in lowered:
                labels.append("235")
            else:
                labels.append(run.name.replace(" eval", ""))
        x = np.arange(len(runs))
        width = 0.35

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)

        avg_vals = [0.0 if math.isnan(run.avg_at_1) else run.avg_at_1 for run in runs]
        pass_vals = [0.0 if math.isnan(run.pass_at_1) else run.pass_at_1 for run in runs]
        axes[0].bar(x - width / 2, avg_vals, width=width, label="Avg@1", color="#2563eb")
        axes[0].bar(x + width / 2, pass_vals, width=width, label="Pass@1", color="#16a34a")
        axes[0].axhline(base.avg_at_1, color="#2563eb", linestyle="--", linewidth=1, alpha=0.5)
        axes[0].axhline(base.pass_at_1, color="#16a34a", linestyle="--", linewidth=1, alpha=0.5)
        perf_min = min(avg_vals + pass_vals)
        perf_lower = min(-0.25, math.floor((perf_min - 0.05) * 10) / 10) if perf_min < 0 else 0
        axes[0].set_ylim(perf_lower, 1)
        axes[0].axhline(0, color="#111827", linewidth=1, alpha=0.4)
        axes[0].set_ylabel("Score")
        axes[0].set_title("Performance Across Runs")
        axes[0].legend(loc="upper right")
        axes[0].grid(axis="y", alpha=0.25)

        exec_fmt = [0.0 if math.isnan(run.executor_native_format_violations) else run.executor_native_format_violations for run in runs]
        invalid_fmt = [0.0 if math.isnan(run.invalid_format_termination_rate) else run.invalid_format_termination_rate for run in runs]
        invalid_action = [0.0 if math.isnan(run.invalid_action_termination_rate) else run.invalid_action_termination_rate for run in runs]
        axes[1].bar(x - width, exec_fmt, width=width, label="ExecutorNativeFormatViolations", color="#dc2626")
        axes[1].bar(x, invalid_fmt, width=width, label="InvalidFormatTerminationRate", color="#f59e0b")
        axes[1].bar(x + width, invalid_action, width=width, label="InvalidActionTerminationRate", color="#7c3aed")
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Rate")
        axes[1].set_title("Executor Failure Modes")
        axes[1].legend(loc="upper right")
        axes[1].grid(axis="y", alpha=0.25)

        planner_invalid = [0.0 if math.isnan(run.planner_invalid_format_rate) else run.planner_invalid_format_rate for run in runs]
        planner_fallback = [0.0 if math.isnan(run.planner_fallback_rate) else run.planner_fallback_rate for run in runs]
        planner_tag_only = [0.0 if math.isnan(run.planner_tag_only_rate) else run.planner_tag_only_rate for run in runs]
        axes[2].plot(x, planner_invalid, marker="o", linewidth=2, label="PlannerInvalidFormatRate", color="#b91c1c")
        axes[2].plot(x, planner_fallback, marker="o", linewidth=2, label="PlannerFallbackRate", color="#ea580c")
        axes[2].plot(x, planner_tag_only, marker="o", linewidth=2, label="PlannerTagOnlyRate", color="#0891b2")
        axes[2].set_ylim(0, 1)
        axes[2].set_ylabel("Rate")
        axes[2].set_title("Planner Failure Modes")
        axes[2].legend(loc="upper right")
        axes[2].grid(axis="y", alpha=0.25)

        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha="right")

        fig.suptitle("BabyAI Multi-Agent Training Progress and Failure Modes", fontsize=16)
        fig.savefig(png_path, dpi=220)
        plt.close(fig)
    except Exception as exc:
        _generate_png_with_pillow(png_path, runs, base)
        print(f"Matplotlib unavailable, used Pillow fallback for PNG: {exc}")

    return 0


def _generate_png_with_pillow(png_path: Path, runs: list[RunMetrics], base: RunMetrics) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1800, 1320
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def load_font(size: int):
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/HelveticaNeue.ttc",
        ]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    title_font = load_font(30)
    section_font = load_font(22)
    body_font = load_font(18)
    small_font = load_font(15)

    def section_title(text: str, x: int, y: int) -> None:
        draw.text((x, y), text, fill="black", font=section_font)

    def draw_chart_box(x0: int, y0: int, x1: int, y1: int, title: str) -> tuple[int, int, int, int]:
        draw.rectangle((x0, y0, x1, y1), outline="#888888", width=2)
        section_title(title, x0 + 10, y0 + 10)
        return x0 + 80, y0 + 52, x1 - 28, y1 - 44

    def scale(value: float, top: int, bottom: int, y_min: float = 0.0, y_max: float = 1.0) -> int:
        if y_max <= y_min:
            y_max = y_min + 1.0
        clipped = max(y_min, min(y_max, value))
        ratio = (clipped - y_min) / (y_max - y_min)
        return bottom - int(ratio * (bottom - top))

    def draw_grouped_bars(box, labels, series, y_min: float = 0.0, y_max: float = 1.0):
        left, top, right, bottom = box
        for i in range(6):
            y = top + int((bottom - top) * i / 5)
            draw.line((left, y, right, y), fill="#e5e7eb", width=1)
            tick_value = y_max - ((y_max - y_min) * i / 5)
            draw.text((left - 52, y - 10), f"{tick_value:.1f}", fill="black", font=small_font)
        group_width = (right - left) / max(1, len(labels))
        bar_width = max(10, int(group_width / (len(series) + 1)))
        zero_y = scale(0.0, top, bottom, y_min=y_min, y_max=y_max)
        if y_min < 0 < y_max:
            draw.line((left, zero_y, right, zero_y), fill="#111827", width=1)
        for idx, label in enumerate(labels):
            gx = left + idx * group_width + 18
            for offset, (_, color, values) in enumerate(series):
                value = values[idx]
                bx0 = gx + offset * bar_width
                bx1 = bx0 + bar_width - 4
                by0 = scale(value, top, bottom, y_min=y_min, y_max=y_max)
                if value >= 0:
                    draw.rectangle((bx0, by0, bx1, zero_y if y_min < 0 else bottom), fill=color, outline=color)
                else:
                    draw.rectangle((bx0, zero_y, bx1, by0), fill=color, outline=color)
            draw.text((gx - 8, bottom + 10), label, fill="black", font=small_font)

    def draw_lines(box, labels, series):
        left, top, right, bottom = box
        for i in range(6):
            y = top + int((bottom - top) * i / 5)
            draw.line((left, y, right, y), fill="#e5e7eb", width=1)
            draw.text((left - 48, y - 10), f"{1 - i * 0.2:.1f}", fill="black", font=small_font)
        step = (right - left) / max(1, len(labels) - 1)
        for _, color, values in series:
            points = []
            for idx, value in enumerate(values):
                px = left + int(idx * step)
                py = scale(value, top, bottom)
                points.append((px, py))
            draw.line(points, fill=color, width=3)
            for px, py in points:
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)
        for idx, label in enumerate(labels):
            px = left + int(idx * step)
            draw.text((px - 18, bottom + 10), label, fill="black", font=small_font)

    labels = []
    for run in runs:
        lowered = run.name.lower()
        if "700" in lowered:
            labels.append("700")
        elif "base" in lowered:
            labels.append("Base")
        elif "100" in lowered:
            labels.append("100")
        elif "350" in lowered:
            labels.append("350")
        elif "50" in lowered:
            labels.append("50")
        elif "15" in lowered:
            labels.append("15")
        elif "236" in lowered:
            labels.append("236")
        elif "235" in lowered:
            labels.append("235")
        else:
            labels.append(run.name.replace(" eval", ""))
    avg_vals = [0.0 if math.isnan(run.avg_at_1) else run.avg_at_1 for run in runs]
    pass_vals = [0.0 if math.isnan(run.pass_at_1) else run.pass_at_1 for run in runs]
    exec_fmt = [0.0 if math.isnan(run.executor_native_format_violations) else run.executor_native_format_violations for run in runs]
    invalid_fmt = [0.0 if math.isnan(run.invalid_format_termination_rate) else run.invalid_format_termination_rate for run in runs]
    invalid_action = [0.0 if math.isnan(run.invalid_action_termination_rate) else run.invalid_action_termination_rate for run in runs]
    planner_invalid = [0.0 if math.isnan(run.planner_invalid_format_rate) else run.planner_invalid_format_rate for run in runs]
    planner_fallback = [0.0 if math.isnan(run.planner_fallback_rate) else run.planner_fallback_rate for run in runs]
    planner_tag_only = [0.0 if math.isnan(run.planner_tag_only_rate) else run.planner_tag_only_rate for run in runs]

    draw.text((40, 24), "BabyAI Multi-Agent Training Progress and Failure Modes", fill="black", font=title_font)
    draw.text((40, 55), f"Base reference: Avg@1={base.avg_at_1:.3f}, Pass@1={base.pass_at_1:.3f}", fill="black", font=body_font)
    draw.text((40, 82), "Runs: " + ", ".join(labels), fill="black", font=body_font)

    perf_box = draw_chart_box(40, 130, 1760, 455, "Performance")
    perf_min = min(avg_vals + pass_vals)
    perf_lower = min(-0.25, math.floor((perf_min - 0.05) * 10) / 10) if perf_min < 0 else 0.0
    draw_grouped_bars(
        perf_box,
        labels,
        [
            ("Avg@1", "#2563eb", avg_vals),
            ("Pass@1", "#16a34a", pass_vals),
        ],
        y_min=perf_lower,
        y_max=1.0,
    )

    exec_box = draw_chart_box(40, 495, 1760, 820, "Executor Failure Modes")
    draw_grouped_bars(
        exec_box,
        labels,
        [
            ("ExecutorNativeFormatViolations", "#dc2626", exec_fmt),
            ("InvalidFormatTerminationRate", "#f59e0b", invalid_fmt),
            ("InvalidActionTerminationRate", "#7c3aed", invalid_action),
        ],
    )

    planner_box = draw_chart_box(40, 860, 1760, 1185, "Planner Failure Modes")
    draw_lines(
        planner_box,
        labels,
        [
            ("PlannerInvalidFormatRate", "#b91c1c", planner_invalid),
            ("PlannerFallbackRate", "#ea580c", planner_fallback),
            ("PlannerTagOnlyRate", "#0891b2", planner_tag_only),
        ],
    )

    legend_y = 1210
    legend_items = [
        ("Avg@1", "#2563eb"),
        ("Pass@1", "#16a34a"),
        ("ExecutorNativeFormatViolations", "#dc2626"),
        ("InvalidFormatTerminationRate", "#f59e0b"),
        ("InvalidActionTerminationRate", "#7c3aed"),
        ("PlannerInvalidFormatRate", "#b91c1c"),
        ("PlannerFallbackRate", "#ea580c"),
        ("PlannerTagOnlyRate", "#0891b2"),
    ]
    x = 40
    for label, color in legend_items:
        draw.rectangle((x, legend_y, x + 14, legend_y + 14), fill=color, outline=color)
        draw.text((x + 22, legend_y - 2), label, fill="black", font=small_font)
        x += 225
        if x > 1550:
            x = 40
            legend_y += 24

    image.save(png_path)


if __name__ == "__main__":
    raise SystemExit(main())
