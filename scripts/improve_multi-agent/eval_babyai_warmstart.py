#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "AgentGym-RL"

import sys

for candidate in (PKG_ROOT, REPO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from verl.improve_multi_agent.protocol import validate_executor_json, validate_planner_json


EXECUTOR_ACTIONS_RE = re.compile(r"^\d+:\s+(.*)$", re.MULTILINE)
PLANNER_ACTIONS_RE = re.compile(r"^-\s+(.*)$", re.MULTILINE)
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_json_blob(text: str) -> str:
    text = text.strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        match = JSON_RE.search(text)
        if not match:
            return text
        return match.group(0)


def _extract_legal_actions(sample: Dict[str, Any], stage: str) -> List[str]:
    metadata = sample.get("metadata") or {}
    if metadata.get("available_actions"):
        return list(metadata["available_actions"])
    prompt = sample["conversations"][0]["value"]
    if stage == "executor":
        return [match.group(1).strip() for match in EXECUTOR_ACTIONS_RE.finditer(prompt)]
    return [match.group(1).strip() for match in PLANNER_ACTIONS_RE.finditer(prompt)]


def _build_generation_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


def _generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    rendered = _build_generation_prompt(tokenizer, prompt)
    batch = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, batch["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def evaluate_executor(samples: List[Dict[str, Any]], model, tokenizer, max_examples: int, max_new_tokens: int) -> Dict[str, Any]:
    metrics = {
        "total": 0,
        "json_valid": 0,
        "legal_action": 0,
        "exact_action_id": 0,
        "exact_json": 0,
        "reason_nonempty": 0,
    }
    examples = []
    for sample in samples[:max_examples]:
        prompt = sample["conversations"][0]["value"]
        target_blob = sample["conversations"][1]["value"]
        target = json.loads(target_blob)
        legal_actions = _extract_legal_actions(sample, "executor")
        prediction = _generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        validation = validate_executor_json(prediction, legal_actions)
        predicted_blob = _extract_json_blob(prediction)
        predicted_json = None
        try:
            predicted_json = json.loads(predicted_blob)
        except json.JSONDecodeError:
            predicted_json = None
        metrics["total"] += 1
        metrics["json_valid"] += int(validation.valid)
        metrics["legal_action"] += int(validation.valid and validation.action in legal_actions)
        metrics["reason_nonempty"] += int(validation.valid and bool(validation.decision.reason.strip()))
        metrics["exact_action_id"] += int(
            bool(predicted_json)
            and predicted_json.get("action_id") == target.get("action_id")
        )
        metrics["exact_json"] += int(predicted_json == target)
        if len(examples) < 5:
            examples.append(
                {
                    "prompt": prompt,
                    "prediction": prediction,
                    "target": target,
                    "validation_reason": validation.reason,
                }
            )
    return _finalize_metrics(metrics, examples)


def evaluate_planner(samples: List[Dict[str, Any]], model, tokenizer, max_examples: int, max_new_tokens: int) -> Dict[str, Any]:
    metrics = {
        "total": 0,
        "json_valid": 0,
        "exact_subgoal": 0,
        "exact_action_hint": 0,
        "exact_json": 0,
        "executable_hint": 0,
    }
    examples = []
    for sample in samples[:max_examples]:
        prompt = sample["conversations"][0]["value"]
        target_blob = sample["conversations"][1]["value"]
        target = json.loads(target_blob)
        legal_actions = _extract_legal_actions(sample, "planner")
        prediction = _generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        validation = validate_planner_json(prediction, legal_actions)
        predicted_blob = _extract_json_blob(prediction)
        try:
            predicted_json = json.loads(predicted_blob)
        except json.JSONDecodeError:
            predicted_json = None
        metrics["total"] += 1
        metrics["json_valid"] += int(validation.valid)
        metrics["exact_subgoal"] += int(
            bool(predicted_json) and predicted_json.get("subgoal_id") == target.get("subgoal_id")
        )
        metrics["exact_action_hint"] += int(
            bool(predicted_json) and predicted_json.get("action_hint") == target.get("action_hint")
        )
        metrics["exact_json"] += int(predicted_json == target)
        metrics["executable_hint"] += int(validation.executable_hint)
        if len(examples) < 5:
            examples.append(
                {
                    "prompt": prompt,
                    "prediction": prediction,
                    "target": target,
                    "validation_reason": validation.reason,
                }
            )
    return _finalize_metrics(metrics, examples)


def _finalize_metrics(metrics: Dict[str, Any], examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = max(1, int(metrics["total"]))
    rates = {
        f"{key}_rate": round(value / total, 4)
        for key, value in metrics.items()
        if key != "total"
    }
    return {
        "metrics": {**metrics, **rates},
        "examples": examples,
    }


def _write_markdown(path: Path, stage: str, result: Dict[str, Any], dataset_path: Path, checkpoint_path: Path) -> None:
    lines = [
        f"# BabyAI {stage.title()} Warm-Start Eval",
        "",
        f"- dataset: `{dataset_path}`",
        f"- checkpoint: `{checkpoint_path}`",
        "",
        "## Metrics",
    ]
    for key, value in result["metrics"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Sample Predictions"])
    for idx, example in enumerate(result["examples"], start=1):
        lines.extend(
            [
                f"### Example {idx}",
                "",
                "**Target**",
                "```json",
                json.dumps(example["target"], ensure_ascii=True, indent=2),
                "```",
                "",
                "**Prediction**",
                "```text",
                example["prediction"],
                "```",
                "",
                f"- validation_reason: `{example['validation_reason']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["executor", "planner"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = _load_dataset(args.dataset)
    if not samples:
        raise SystemExit(f"Dataset is empty: {args.dataset}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if args.stage == "executor":
        result = evaluate_executor(samples, model, tokenizer, args.max_examples, args.max_new_tokens)
    else:
        result = evaluate_planner(samples, model, tokenizer, args.max_examples, args.max_new_tokens)

    summary = {
        "stage": args.stage,
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        **result,
    }
    json_path = args.output_dir / f"{args.stage}_warmstart_eval.json"
    md_path = args.output_dir / f"{args.stage}_warmstart_eval.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    _write_markdown(md_path, args.stage, result, args.dataset, args.checkpoint)
    print(json.dumps(summary["metrics"], ensure_ascii=True, indent=2))
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
