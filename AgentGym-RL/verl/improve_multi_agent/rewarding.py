from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List

from .protocol import PlannerMessage, extract_carrying_item, extract_front_distance, extract_visible_objects

PLANNER_VALID_JSON_REWARD = 0.02
PLANNER_EXECUTABLE_HINT_REWARD = 0.02
PLANNER_SUBGOAL_SUCCESS_REWARD = 0.05
PLANNER_INVALID_JSON_PENALTY = -0.05
PLANNER_CONTRADICTION_PENALTY = -0.05
EXECUTOR_VALID_JSON_REWARD = 0.02
EXECUTOR_LEGAL_ACTION_REWARD = 0.02
EXECUTOR_STATE_CHANGE_REWARD = 0.02
EXECUTOR_SUBGOAL_SUCCESS_REWARD = 0.05
EXECUTOR_INVALID_OUTPUT_PENALTY = -0.2
EXECUTOR_ILLEGAL_ACTION_PENALTY = -0.2
MILESTONE_REWARD = 0.03


@dataclass(frozen=True)
class PlannerRewardBreakdown:
    total: float
    valid_json: float
    executable_hint: float
    subgoal_success: float
    contradiction_penalty: float
    invalid_json_penalty: float


@dataclass(frozen=True)
class ExecutorRewardBreakdown:
    total: float
    valid_json: float
    legal_action: float
    state_change: float
    milestone: float
    subgoal_success: float
    invalid_output_penalty: float
    illegal_action_penalty: float


def _extract_open_doors(observation: str) -> set[str]:
    tokens = set()
    lowered = (observation or "").lower()
    for color in ("red", "green", "blue", "yellow", "grey", "purple"):
        if f"{color} open door" in lowered or f"{color} opened door" in lowered:
            tokens.add(color)
    return tokens


def detect_babyai_milestones(
    previous_observation: str,
    current_observation: str,
    planner_message: PlannerMessage,
    previous_score: float,
    current_score: float,
    valid_action_streak: int,
) -> Dict[str, float]:
    previous_visible = set(extract_visible_objects(previous_observation))
    current_visible = set(extract_visible_objects(current_observation))
    previous_carry = extract_carrying_item(previous_observation)
    current_carry = extract_carrying_item(current_observation)
    previous_open = _extract_open_doors(previous_observation)
    current_open = _extract_open_doors(current_observation)

    room_transition = 0.0
    if previous_visible and current_visible and previous_visible != current_visible:
        room_transition = 1.0

    target_label = planner_message.target["color"]
    object_type = planner_message.target["object_type"]
    target_object_visible = 0.0
    if object_type != "none":
        target_token = f"{target_label} {object_type}".strip()
        if target_token in current_visible:
            target_object_visible = 1.0

    inventory_changed = 1.0 if previous_carry != current_carry else 0.0
    door_opened = 1.0 if len(current_open - previous_open) > 0 else 0.0
    positive_task_delta = 1.0 if current_score > previous_score else 0.0
    valid_streak = 1.0 if valid_action_streak >= 3 else 0.0

    return {
        "room_transition": room_transition,
        "target_object_visible": target_object_visible,
        "inventory_changed": inventory_changed,
        "door_opened": door_opened,
        "positive_task_delta": positive_task_delta,
        "valid_action_streak": valid_streak,
    }


def detect_subgoal_completion(
    planner_message: PlannerMessage,
    previous_observation: str,
    current_observation: str,
    executed_action: str,
    milestones: Dict[str, float],
) -> bool:
    subgoal = planner_message.subgoal_id
    current_carry = extract_carrying_item(current_observation)
    target_type = planner_message.target["object_type"]
    target_color = planner_message.target["color"]
    target_token = f"{target_color} {target_type}".strip()
    front_distance = extract_front_distance(current_observation)

    if subgoal == "explore":
        return bool(milestones["room_transition"] or milestones["target_object_visible"])
    if subgoal == "approach":
        if target_type == "none":
            return False
        visible = target_token in set(extract_visible_objects(current_observation))
        close_enough = front_distance is not None and front_distance <= 1
        return visible and close_enough
    if subgoal == "pickup":
        return target_type != "none" and target_token in current_carry
    if subgoal in {"open", "unlock"}:
        return bool(milestones["door_opened"])
    if subgoal == "drop":
        return "drop" in executed_action.lower() and current_carry == "none"
    if subgoal == "finish":
        return bool(milestones["positive_task_delta"])
    return False


def compute_planner_reward(
    *,
    valid_json: bool,
    executable_hint: bool,
    subgoal_success: bool,
    contradiction: bool,
) -> PlannerRewardBreakdown:
    valid_json_reward = PLANNER_VALID_JSON_REWARD if valid_json else 0.0
    executable_reward = PLANNER_EXECUTABLE_HINT_REWARD if executable_hint else 0.0
    subgoal_reward = PLANNER_SUBGOAL_SUCCESS_REWARD if subgoal_success else 0.0
    contradiction_penalty = PLANNER_CONTRADICTION_PENALTY if contradiction else 0.0
    invalid_penalty = 0.0 if valid_json else PLANNER_INVALID_JSON_PENALTY
    total = valid_json_reward + executable_reward + subgoal_reward + contradiction_penalty + invalid_penalty
    return PlannerRewardBreakdown(
        total=total,
        valid_json=valid_json_reward,
        executable_hint=executable_reward,
        subgoal_success=subgoal_reward,
        contradiction_penalty=contradiction_penalty,
        invalid_json_penalty=invalid_penalty,
    )


def compute_executor_reward(
    *,
    valid_json: bool,
    legal_action: bool,
    observation_changed: bool,
    milestone_hits: Dict[str, float],
    subgoal_success: bool,
) -> ExecutorRewardBreakdown:
    valid_json_reward = EXECUTOR_VALID_JSON_REWARD if valid_json else 0.0
    legal_action_reward = EXECUTOR_LEGAL_ACTION_REWARD if legal_action else 0.0
    state_change_reward = EXECUTOR_STATE_CHANGE_REWARD if observation_changed else 0.0
    milestone_reward = MILESTONE_REWARD * sum(float(v) for v in milestone_hits.values())
    subgoal_reward = EXECUTOR_SUBGOAL_SUCCESS_REWARD if subgoal_success else 0.0
    invalid_penalty = 0.0 if valid_json else EXECUTOR_INVALID_OUTPUT_PENALTY
    illegal_penalty = 0.0 if legal_action else EXECUTOR_ILLEGAL_ACTION_PENALTY
    total = (
        valid_json_reward
        + legal_action_reward
        + state_change_reward
        + milestone_reward
        + subgoal_reward
        + invalid_penalty
        + illegal_penalty
    )
    return ExecutorRewardBreakdown(
        total=total,
        valid_json=valid_json_reward,
        legal_action=legal_action_reward,
        state_change=state_change_reward,
        milestone=milestone_reward,
        subgoal_success=subgoal_reward,
        invalid_output_penalty=invalid_penalty,
        illegal_action_penalty=illegal_penalty,
    )


def trailing_median(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(statistics.median(values))
