from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from .protocol import (
    ExecutorDecision,
    PlannerMessage,
    build_executor_prompt,
    build_planner_prompt,
    extract_available_actions,
    extract_front_distance,
    extract_visible_objects,
    render_babyai_action_payload,
    safe_json_dumps,
    validate_executor_json,
    validate_planner_json,
)
from .rewarding import detect_babyai_milestones, detect_subgoal_completion

DEFAULT_FAILURE_REASONS = {
    "reset": "reset_error",
    "observe": "observe_error",
    "available_actions": "missing_legal_actions",
    "planner_generate": "planner_error",
    "planner_validate": "planner_invalid_json",
    "executor_generate": "executor_error",
    "executor_validate": "executor_invalid_json",
    "env_step": "env_step_error",
}

TIMEOUT_FAILURE_REASONS = {
    "reset": "reset_timeout",
    "observe": "observe_timeout",
    "available_actions": "available_actions_timeout",
    "planner_generate": "planner_timeout",
    "planner_validate": "planner_validate_timeout",
    "executor_generate": "executor_timeout",
    "executor_validate": "executor_validate_timeout",
    "env_step": "env_step_timeout",
}


@dataclass
class BoundaryEvent:
    name: str
    status: str
    started_at_unix: float
    ended_at_unix: float
    duration_ms: float
    detail: str = ""


@dataclass
class StepRecord:
    step_index: int
    observation_excerpt: str
    legal_actions: list[str]
    previous_action: str
    planner_mode: str
    executor_mode: str
    planner_prompt: str = ""
    planner_raw_output: str = ""
    planner_validation_reason: str = ""
    planner_message_json: str = ""
    executor_prompt: str = ""
    executor_raw_output: str = ""
    executor_validation_reason: str = ""
    executor_decision_json: str = ""
    action_payload: str = ""
    env_reward: Optional[float] = None
    env_done: Optional[bool] = None
    env_state_excerpt: str = ""
    boundary_events: list[BoundaryEvent] = field(default_factory=list)


@dataclass
class EpisodeRecord:
    status: str
    episode_id: str
    item_id: int
    planner_mode: str
    executor_mode: str
    completed_env_steps: int
    first_transition_completed: bool
    latest_success_boundary: str
    failure_type: Optional[str]
    failure_detail: str
    started_at_unix: float
    ended_at_unix: float
    duration_ms: float
    max_steps: int
    planner_interval: int
    terminal_state_excerpt: str
    step_records: list[StepRecord]
    boundary_events: list[BoundaryEvent]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


class PlannerAgent(Protocol):
    mode: str

    def generate(self, observation: str, legal_actions: Sequence[str], previous_action: str) -> str:
        ...


class ExecutorAgent(Protocol):
    mode: str

    def generate(
        self,
        observation: str,
        planner_message: PlannerMessage,
        legal_actions: Sequence[str],
        previous_action: str,
    ) -> str:
        ...


def _clip(text: Any, limit: int = 240) -> str:
    value = str(text)
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _now() -> float:
    return time.time()


def _run_with_timeout(fn: Callable[[], Any], timeout_seconds: float) -> tuple[str, Any]:
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def _target() -> None:
        try:
            result_queue.put(("ok", fn()))
        except Exception as exc:  # pragma: no cover - exercised via harness tests
            result_queue.put(("error", exc))

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        return "timeout", TimeoutError(f"Exceeded timeout of {timeout_seconds:.1f}s")
    if result_queue.empty():
        return "error", RuntimeError("Boundary completed without a result payload")
    return result_queue.get_nowait()


def _call_boundary(
    *,
    name: str,
    timeout_seconds: float,
    trace: list[BoundaryEvent],
    step_trace: Optional[list[BoundaryEvent]],
    latest_success_boundary: list[str],
    fn: Callable[[], Any],
) -> tuple[Optional[Any], Optional[str], Optional[str]]:
    started = _now()
    status, payload = _run_with_timeout(fn, timeout_seconds)
    ended = _now()
    detail = ""
    if status == "error":
        detail = repr(payload)
    elif status == "timeout":
        detail = str(payload)
    event = BoundaryEvent(
        name=name,
        status=status,
        started_at_unix=started,
        ended_at_unix=ended,
        duration_ms=round((ended - started) * 1000.0, 3),
        detail=detail,
    )
    trace.append(event)
    if step_trace is not None:
        step_trace.append(event)
    if status == "ok":
        latest_success_boundary[0] = name
        return payload, None, None
    if status == "timeout":
        return None, TIMEOUT_FAILURE_REASONS.get(name, f"{name}_timeout"), detail
    return None, DEFAULT_FAILURE_REASONS.get(name, f"{name}_error"), detail


def _build_planner_message_from_action(action: str, observation: str) -> PlannerMessage:
    normalized = action.lower().strip()
    target = {"object_type": "none", "color": "none", "location_hint": "unknown"}
    visible_objects = extract_visible_objects(observation)
    if visible_objects:
        first_visible = visible_objects[0].split()
        if len(first_visible) == 2:
            target["color"], target["object_type"] = first_visible
    if "pick up" in normalized:
        subgoal_id = "pickup"
        action_hint = "pickup"
    elif "toggle" in normalized:
        subgoal_id = "open"
        action_hint = "toggle"
    elif normalized == "drop":
        subgoal_id = "drop"
        action_hint = "drop"
    elif normalized == "turn left":
        subgoal_id = "explore"
        action_hint = "turn_left"
        target = {"object_type": "none", "color": "none", "location_hint": "left"}
    elif normalized == "turn right":
        subgoal_id = "explore"
        action_hint = "turn_right"
        target = {"object_type": "none", "color": "none", "location_hint": "right"}
    elif normalized == "move forward":
        subgoal_id = "approach"
        action_hint = "move_forward"
        target = {"object_type": "none", "color": "none", "location_hint": "front"}
    elif normalized.startswith("go to"):
        subgoal_id = "approach"
        action_hint = "move_forward"
    else:
        subgoal_id = "finish"
        action_hint = "done"
    location_hint = "unknown"
    observation_lower = observation.lower()
    if "to your left" in observation_lower:
        location_hint = "left"
    elif "to your right" in observation_lower:
        location_hint = "right"
    elif extract_front_distance(observation) not in (None, 0):
        location_hint = "front"
    target["location_hint"] = location_hint
    return PlannerMessage(
        schema_version="v1",
        subgoal_id=subgoal_id,
        target=target,
        action_hint=action_hint,
        success_check="world state moves toward target",
        confidence=0.75,
    )


class ScriptedPlannerAgent:
    mode = "scripted"

    def generate(self, observation: str, legal_actions: Sequence[str], previous_action: str) -> str:
        del previous_action
        preferred_action = None
        for candidate in legal_actions:
            lowered = candidate.lower()
            if "pick up" in lowered:
                preferred_action = candidate
                break
            if lowered.startswith("go to"):
                preferred_action = candidate
            elif lowered == "toggle" and preferred_action is None:
                preferred_action = candidate
        if preferred_action is None:
            if "move forward" in legal_actions and extract_front_distance(observation) not in (None, 0):
                preferred_action = "move forward"
            elif "turn left" in legal_actions and "to your left" in observation.lower():
                preferred_action = "turn left"
            elif "turn right" in legal_actions and "to your right" in observation.lower():
                preferred_action = "turn right"
            else:
                preferred_action = legal_actions[0]
        message = _build_planner_message_from_action(preferred_action, observation)
        if _normalize_action_hint_to_legal(message.action_hint, legal_actions) is None and message.action_hint != "done":
            if "turn left" in legal_actions and message.target.get("location_hint") == "left":
                message = PlannerMessage(
                    schema_version=message.schema_version,
                    subgoal_id="explore",
                    target={"object_type": "none", "color": "none", "location_hint": "left"},
                    action_hint="turn_left",
                    success_check="observation changes after scouting",
                    confidence=message.confidence,
                )
            elif "turn right" in legal_actions and message.target.get("location_hint") == "right":
                message = PlannerMessage(
                    schema_version=message.schema_version,
                    subgoal_id="explore",
                    target={"object_type": "none", "color": "none", "location_hint": "right"},
                    action_hint="turn_right",
                    success_check="observation changes after scouting",
                    confidence=message.confidence,
                )
            elif "move forward" in legal_actions:
                message = PlannerMessage(
                    schema_version=message.schema_version,
                    subgoal_id="explore",
                    target={"object_type": "none", "color": "none", "location_hint": "front"},
                    action_hint="move_forward",
                    success_check="world state changes after scouting",
                    confidence=message.confidence,
                )
            elif any("pick up" in action.lower() for action in legal_actions):
                message = PlannerMessage(
                    schema_version=message.schema_version,
                    subgoal_id="pickup",
                    target=message.target,
                    action_hint="pickup",
                    success_check=message.success_check,
                    confidence=message.confidence,
                )
            elif "toggle" in [action.lower() for action in legal_actions]:
                message = PlannerMessage(
                    schema_version=message.schema_version,
                    subgoal_id="open",
                    target=message.target,
                    action_hint="toggle",
                    success_check=message.success_check,
                    confidence=message.confidence,
                )
            else:
                message = PlannerMessage(
                    schema_version=message.schema_version,
                    subgoal_id="finish",
                    target={"object_type": "none", "color": "none", "location_hint": "unknown"},
                    action_hint="done",
                    success_check="no more legal progress available",
                    confidence=0.0,
                )
        return message.to_json()


class ScriptedExecutorAgent:
    mode = "scripted"

    def generate(
        self,
        observation: str,
        planner_message: PlannerMessage,
        legal_actions: Sequence[str],
        previous_action: str,
    ) -> str:
        del observation, previous_action
        action_index = _choose_action_id(planner_message, legal_actions)
        decision = ExecutorDecision(
            reason="Follow the structured subgoal.",
            action_id=action_index,
        )
        return decision.to_json()


class _SharedTransformersRunner:
    def __init__(self) -> None:
        self._active_checkpoint: Optional[str] = None
        self._tokenizer = None
        self._model = None

    def _load(self, checkpoint: str) -> None:
        if self._active_checkpoint == checkpoint and self._model is not None and self._tokenizer is not None:
            return
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on local install
            raise RuntimeError("transformers and torch are required for model-backed rollout harness mode") from exc

        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        self._active_checkpoint = checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, checkpoint: str, prompt: str, max_new_tokens: int) -> str:
        self._load(checkpoint)
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on local install
            raise RuntimeError("torch is required for model-backed rollout harness mode") from exc
        rendered = _build_generation_prompt(self._tokenizer, prompt)
        batch = self._tokenizer(rendered, return_tensors="pt")
        if hasattr(self._model, "device"):
            batch = {key: value.to(self._model.device) for key, value in batch.items()}
        with torch.no_grad():
            output = self._model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0, batch["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


def _build_generation_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


class ModelPlannerAgent:
    mode = "model"

    def __init__(self, checkpoint: str, max_new_tokens: int, runner: _SharedTransformersRunner):
        self.checkpoint = checkpoint
        self.max_new_tokens = max_new_tokens
        self.runner = runner

    def generate(self, observation: str, legal_actions: Sequence[str], previous_action: str) -> str:
        prompt = build_planner_prompt(observation, previous_action=previous_action, legal_actions=legal_actions)
        return self.runner.generate(self.checkpoint, prompt, self.max_new_tokens)


class ModelExecutorAgent:
    mode = "model"

    def __init__(self, checkpoint: str, max_new_tokens: int, runner: _SharedTransformersRunner):
        self.checkpoint = checkpoint
        self.max_new_tokens = max_new_tokens
        self.runner = runner

    def generate(
        self,
        observation: str,
        planner_message: PlannerMessage,
        legal_actions: Sequence[str],
        previous_action: str,
    ) -> str:
        prompt = build_executor_prompt(
            observation=observation,
            planner_message=planner_message,
            legal_actions=legal_actions,
            previous_action=previous_action,
        )
        return self.runner.generate(self.checkpoint, prompt, self.max_new_tokens)


def _normalize_action_hint_to_legal(action_hint: str, legal_actions: Sequence[str]) -> Optional[int]:
    mappings = [
        ("turn_left", lambda action: action.lower() == "turn left"),
        ("turn_right", lambda action: action.lower() == "turn right"),
        ("move_forward", lambda action: action.lower() == "move forward"),
        ("pickup", lambda action: "pick up" in action.lower()),
        ("toggle", lambda action: "toggle" in action.lower()),
        ("drop", lambda action: action.lower() == "drop"),
        ("done", lambda action: action.lower().startswith("done")),
    ]
    for hint, predicate in mappings:
        if action_hint != hint:
            continue
        for idx, action in enumerate(legal_actions):
            if predicate(action):
                return idx
    return None


def _choose_action_id(planner_message: PlannerMessage, legal_actions: Sequence[str]) -> int:
    direct_match = _normalize_action_hint_to_legal(planner_message.action_hint, legal_actions)
    if direct_match is not None:
        return direct_match
    target_object = planner_message.target.get("object_type", "none")
    target_color = planner_message.target.get("color", "none")
    if target_object != "none":
        for idx, action in enumerate(legal_actions):
            lowered = action.lower()
            if target_object in lowered and (target_color == "none" or target_color in lowered):
                return idx
    for idx, action in enumerate(legal_actions):
        if action.lower() != "check available actions":
            return idx
    return 0


def build_planner_agent(
    *,
    mode: str,
    checkpoint: Optional[str],
    max_new_tokens: int,
    runner: Optional[_SharedTransformersRunner] = None,
) -> PlannerAgent:
    if mode == "scripted":
        return ScriptedPlannerAgent()
    if not checkpoint:
        raise ValueError("planner checkpoint is required when planner mode is 'model'")
    return ModelPlannerAgent(checkpoint=checkpoint, max_new_tokens=max_new_tokens, runner=runner or _SharedTransformersRunner())


def build_executor_agent(
    *,
    mode: str,
    checkpoint: Optional[str],
    max_new_tokens: int,
    runner: Optional[_SharedTransformersRunner] = None,
) -> ExecutorAgent:
    if mode == "scripted":
        return ScriptedExecutorAgent()
    if not checkpoint:
        raise ValueError("executor checkpoint is required when executor mode is 'model'")
    return ModelExecutorAgent(checkpoint=checkpoint, max_new_tokens=max_new_tokens, runner=runner or _SharedTransformersRunner())


def build_babyai_env_client(env_server_url: str, timeout_seconds: float):
    from agentenv.envs import BabyAIEnvClient

    return BabyAIEnvClient(env_server_base=env_server_url, data_len=1, timeout=int(timeout_seconds))


def run_babyai_rollout_episode(
    *,
    item_id: int,
    env_client: Any,
    planner_agent: PlannerAgent,
    executor_agent: ExecutorAgent,
    max_steps: int = 6,
    planner_interval: int = 3,
    boundary_timeouts: Optional[Dict[str, float]] = None,
) -> EpisodeRecord:
    boundary_timeouts = boundary_timeouts or {}
    boundary_events: list[BoundaryEvent] = []
    step_records: list[StepRecord] = []
    latest_success_boundary = [""]
    started_at = _now()
    episode_id = str(uuid.uuid4())
    previous_action = "none"
    previous_score = 0.0
    current_plan: Optional[PlannerMessage] = None
    planner_steps_remaining = 0
    force_replan = True
    completed_env_steps = 0
    terminal_state_excerpt = ""
    terminal_failure_type: Optional[str] = None
    terminal_failure_detail = ""

    reset_result, failure_type, failure_detail = _call_boundary(
        name="reset",
        timeout_seconds=boundary_timeouts.get("reset", 20.0),
        trace=boundary_events,
        step_trace=None,
        latest_success_boundary=latest_success_boundary,
        fn=lambda: env_client.reset(item_id),
    )
    if failure_type:
        terminal_failure_type = failure_type
        terminal_failure_detail = failure_detail or ""
        ended_at = _now()
        return EpisodeRecord(
            status="failed",
            episode_id=episode_id,
            item_id=item_id,
            planner_mode=planner_agent.mode,
            executor_mode=executor_agent.mode,
            completed_env_steps=0,
            first_transition_completed=False,
            latest_success_boundary=latest_success_boundary[0],
            failure_type=terminal_failure_type,
            failure_detail=terminal_failure_detail,
            started_at_unix=started_at,
            ended_at_unix=ended_at,
            duration_ms=round((ended_at - started_at) * 1000.0, 3),
            max_steps=max_steps,
            planner_interval=planner_interval,
            terminal_state_excerpt="",
            step_records=step_records,
            boundary_events=boundary_events,
        )

    done = bool((reset_result or {}).get("done", False))
    for step_index in range(max_steps):
        if done:
            break

        step_boundary_events: list[BoundaryEvent] = []
        observe_result, failure_type, failure_detail = _call_boundary(
            name="observe",
            timeout_seconds=boundary_timeouts.get("observe", 15.0),
            trace=boundary_events,
            step_trace=step_boundary_events,
            latest_success_boundary=latest_success_boundary,
            fn=env_client.observe,
        )
        if failure_type:
            terminal_failure_type = failure_type
            terminal_failure_detail = failure_detail or ""
            terminal_state_excerpt = failure_detail or ""
            break
        observation = str(observe_result)

        legal_actions, failure_type, failure_detail = _call_boundary(
            name="available_actions",
            timeout_seconds=boundary_timeouts.get("available_actions", 2.0),
            trace=boundary_events,
            step_trace=step_boundary_events,
            latest_success_boundary=latest_success_boundary,
            fn=lambda: extract_available_actions(observation),
        )
        legal_actions = list(legal_actions or [])
        if failure_type:
            terminal_failure_type = failure_type
            terminal_failure_detail = failure_detail or ""
            terminal_state_excerpt = failure_detail or ""
            break
        if not legal_actions:
            terminal_failure_type = "missing_legal_actions"
            terminal_failure_detail = "No legal actions were extracted from the current observation."
            terminal_state_excerpt = _clip(observation)
            boundary_events.append(
                BoundaryEvent(
                    name="available_actions",
                    status="error",
                    started_at_unix=_now(),
                    ended_at_unix=_now(),
                    duration_ms=0.0,
                    detail=terminal_failure_detail,
                )
            )
            break

        step_record = StepRecord(
            step_index=step_index,
            observation_excerpt=_clip(observation),
            legal_actions=legal_actions,
            previous_action=previous_action,
            planner_mode=planner_agent.mode,
            executor_mode=executor_agent.mode,
            boundary_events=step_boundary_events,
        )

        if force_replan or planner_steps_remaining <= 0 or current_plan is None:
            step_record.planner_prompt = build_planner_prompt(
                observation,
                previous_action=previous_action,
                legal_actions=legal_actions,
            )
            planner_raw, failure_type, failure_detail = _call_boundary(
                name="planner_generate",
                timeout_seconds=boundary_timeouts.get("planner_generate", 45.0),
                trace=boundary_events,
                step_trace=step_boundary_events,
                latest_success_boundary=latest_success_boundary,
                fn=lambda: planner_agent.generate(observation, legal_actions, previous_action),
            )
            if failure_type:
                terminal_failure_type = failure_type
                terminal_failure_detail = failure_detail or ""
                terminal_state_excerpt = failure_detail or ""
                step_records.append(step_record)
                break
            step_record.planner_raw_output = str(planner_raw)
            planner_validation, failure_type, failure_detail = _call_boundary(
                name="planner_validate",
                timeout_seconds=boundary_timeouts.get("planner_validate", 2.0),
                trace=boundary_events,
                step_trace=step_boundary_events,
                latest_success_boundary=latest_success_boundary,
                fn=lambda: validate_planner_json(str(planner_raw), legal_actions),
            )
            if failure_type:
                terminal_failure_type = failure_type
                terminal_failure_detail = failure_detail or ""
                terminal_state_excerpt = failure_detail or ""
                step_records.append(step_record)
                break
            step_record.planner_validation_reason = planner_validation.reason
            if not planner_validation.valid:
                terminal_failure_type = "planner_invalid_json"
                terminal_failure_detail = planner_validation.reason
                terminal_state_excerpt = _clip(planner_raw)
                step_records.append(step_record)
                break
            current_plan = planner_validation.message
            planner_steps_remaining = planner_interval
            force_replan = False
            step_record.planner_message_json = current_plan.to_json()

        assert current_plan is not None
        step_record.executor_prompt = build_executor_prompt(
            observation=observation,
            planner_message=current_plan,
            legal_actions=legal_actions,
            previous_action=previous_action,
        )
        executor_raw, failure_type, failure_detail = _call_boundary(
            name="executor_generate",
            timeout_seconds=boundary_timeouts.get("executor_generate", 45.0),
            trace=boundary_events,
            step_trace=step_boundary_events,
            latest_success_boundary=latest_success_boundary,
            fn=lambda: executor_agent.generate(observation, current_plan, legal_actions, previous_action),
        )
        if failure_type:
            terminal_failure_type = failure_type
            terminal_failure_detail = failure_detail or ""
            terminal_state_excerpt = failure_detail or ""
            step_records.append(step_record)
            break
        step_record.executor_raw_output = str(executor_raw)
        executor_validation, failure_type, failure_detail = _call_boundary(
            name="executor_validate",
            timeout_seconds=boundary_timeouts.get("executor_validate", 2.0),
            trace=boundary_events,
            step_trace=step_boundary_events,
            latest_success_boundary=latest_success_boundary,
            fn=lambda: validate_executor_json(str(executor_raw), legal_actions),
        )
        if failure_type:
            terminal_failure_type = failure_type
            terminal_failure_detail = failure_detail or ""
            terminal_state_excerpt = failure_detail or ""
            step_records.append(step_record)
            break
        step_record.executor_validation_reason = executor_validation.reason
        if not executor_validation.valid or executor_validation.decision is None:
            terminal_failure_type = (
                "illegal_action_id" if executor_validation.reason == "action_id_out_of_range" else "executor_invalid_json"
            )
            terminal_failure_detail = executor_validation.reason
            terminal_state_excerpt = _clip(executor_raw)
            step_records.append(step_record)
            break

        decision = executor_validation.decision
        step_record.executor_decision_json = decision.to_json()
        action_payload = render_babyai_action_payload(decision, legal_actions)
        step_record.action_payload = action_payload
        step_output, failure_type, failure_detail = _call_boundary(
            name="env_step",
            timeout_seconds=boundary_timeouts.get("env_step", 30.0),
            trace=boundary_events,
            step_trace=step_boundary_events,
            latest_success_boundary=latest_success_boundary,
            fn=lambda: env_client.step(action_payload),
        )
        if failure_type:
            if failure_type == "env_step_timeout":
                terminal_failure_type = "env_step_timeout"
            else:
                terminal_failure_type = "env_step_error"
            terminal_failure_detail = failure_detail or ""
            terminal_state_excerpt = failure_detail or ""
            step_records.append(step_record)
            break

        completed_env_steps += 1
        previous_action = executor_validation.action or previous_action
        reward = float(step_output.reward)
        done = bool(step_output.done)
        next_observation = str(step_output.state)
        step_record.env_reward = reward
        step_record.env_done = done
        step_record.env_state_excerpt = _clip(next_observation)

        milestones = detect_babyai_milestones(
            observation,
            next_observation,
            current_plan,
            previous_score=previous_score,
            current_score=reward,
            valid_action_streak=completed_env_steps,
        )
        subgoal_success = detect_subgoal_completion(
            current_plan,
            observation,
            next_observation,
            previous_action,
            milestones,
        )
        planner_steps_remaining -= 1
        if subgoal_success or any(
            milestones[key] > 0.0
            for key in ("inventory_changed", "door_opened", "positive_task_delta", "target_object_visible")
        ):
            force_replan = True
        previous_score = reward
        terminal_state_excerpt = _clip(next_observation)
        step_records.append(step_record)

    if not terminal_state_excerpt and reset_result is not None:
        terminal_state_excerpt = _clip(json.dumps(reset_result, ensure_ascii=True))

    status = "passed" if completed_env_steps > 0 and terminal_failure_type is None else "failed"
    ended_at = _now()
    return EpisodeRecord(
        status=status,
        episode_id=episode_id,
        item_id=item_id,
        planner_mode=planner_agent.mode,
        executor_mode=executor_agent.mode,
        completed_env_steps=completed_env_steps,
        first_transition_completed=completed_env_steps > 0,
        latest_success_boundary=latest_success_boundary[0],
        failure_type=terminal_failure_type,
        failure_detail=terminal_failure_detail,
        started_at_unix=started_at,
        ended_at_unix=ended_at,
        duration_ms=round((ended_at - started_at) * 1000.0, 3),
        max_steps=max_steps,
        planner_interval=planner_interval,
        terminal_state_excerpt=terminal_state_excerpt,
        step_records=step_records,
        boundary_events=boundary_events,
    )


def write_episode_artifacts(record: EpisodeRecord, *, summary_path: Path, trace_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(record.to_dict(), ensure_ascii=True, indent=2), encoding="utf-8")
    with trace_path.open("w", encoding="utf-8") as handle:
        for event in record.boundary_events:
            handle.write(safe_json_dumps({"type": "boundary", **asdict(event)}) + "\n")
        for step in record.step_records:
            payload = asdict(step)
            payload["type"] = "step"
            handle.write(safe_json_dumps(payload) + "\n")
        handle.write(
            safe_json_dumps(
                {
                    "type": "terminal",
                    "status": record.status,
                    "failure_type": record.failure_type,
                    "completed_env_steps": record.completed_env_steps,
                    "first_transition_completed": record.first_transition_completed,
                }
            )
            + "\n"
        )


def aggregate_episode_records(records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    episodes = [record.to_dict() for record in records]
    episodes_requested = len(records)
    episodes_passed = sum(1 for record in records if record.status == "passed")
    first_transition_count = sum(1 for record in records if record.first_transition_completed)
    return {
        "status": "passed" if episodes_requested > 0 and episodes_passed == episodes_requested else "failed",
        "episodes_requested": episodes_requested,
        "episodes_passed": episodes_passed,
        "first_transition_completed": episodes_requested > 0 and first_transition_count == episodes_requested,
        "first_transition_completed_count": first_transition_count,
        "completed_env_steps_total": sum(record.completed_env_steps for record in records),
        "episodes": episodes,
    }


def write_aggregate_summary(records: Sequence[EpisodeRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(aggregate_episode_records(records), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def ensure_harness_passed(summary_path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(summary_path)
    if not path.exists():
        raise RuntimeError(f"Rollout harness summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed" or not payload.get("first_transition_completed"):
        raise RuntimeError(
            "Rollout harness gate failed: "
            f"status={payload.get('status')} first_transition_completed={payload.get('first_transition_completed')} "
            f"path={path}"
        )
    return payload


__all__ = [
    "BoundaryEvent",
    "EpisodeRecord",
    "ExecutorAgent",
    "PlannerAgent",
    "ScriptedExecutorAgent",
    "ScriptedPlannerAgent",
    "_SharedTransformersRunner",
    "aggregate_episode_records",
    "build_babyai_env_client",
    "build_executor_agent",
    "build_planner_agent",
    "ensure_harness_passed",
    "run_babyai_rollout_episode",
    "write_aggregate_summary",
    "write_episode_artifacts",
]
