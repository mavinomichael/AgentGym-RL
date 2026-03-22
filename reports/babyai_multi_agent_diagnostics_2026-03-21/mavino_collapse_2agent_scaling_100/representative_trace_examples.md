# Representative Trace Examples

## first_header_leak_step55

- `training_step`: 55
- `item_id`: 1722
- `round`: 1

**planner_prompt**

```text
[Planner Turn]
[Environment Observation]
Your goal: go to the red ball
In front of you in this room, you can see several objects: The room has walls around you. You are facing a wall 2 steps away. You are not carrying anything.
Available actions: ["turn left", "turn right", "move forward", "check available actions"]

You are the Planner.

The original task instruction, action surface, and final response format remain unchanged.
Help the Executor produce the next valid single-agent response.

Output rules:
- Output exactly one short guidance phrase.
- Output 2 to 6 words only.
- Start with a verb.
- Focus on the next useful direction, object, or constraint.
- Stay grounded in the observation only.
- Do not emit the final environment response.
- Do not output the exact environment action.
- Do not explain.
- Do not use role labels, Thought:, or Action:.

Output only the guidance phrase.
```

**observation_excerpt**

```text
Your goal: go to the red ball
In front of you in this room, you can see several objects: The room has walls around you. You are facing a wall 2 steps away. You are not carrying anything.
Available actions: ["turn left", "turn right", "move forward", "check available actions"]
```

**planner_raw_output**

```text
[Planner] Check room for ball.
```

**planner_message**

```text
Check room for ball
```

**executor_prompt**

```text
[Executor Turn]
[Environment Observation]
Your goal: go to the red ball
In front of you in this room, you can see several objects: The room has walls around you. You are facing a wall 2 steps away. You are not carrying anything.
Available actions: ["turn left", "turn right", "move forward", "check available actions"]

[Latest Planner Message]
Check room for ball

You are the Executor.
Produce the next environment response exactly as the original single-agent agent would.
The original task instruction, action surface, and response format remain unchanged.
Use the planner message as guidance only; if it conflicts with the observation, follow the observation and the original task instruction.
Format requirement: Use the BabyAI native format with a Thought section followed by a single Action line.
Do not prepend role labels like Planner: or Executor:.

BabyAI reminder:
- Keep the original single-agent structure:
Thought:
<brief thought>

Action:
<one action>
- Output exactly one Action line.
```

**executor_raw_output**

```text
Thought:
Observe room for ball.

Action:
turn right
```

## healthy_step50_success

- `training_step`: 50
- `item_id`: 7515
- `round`: 1
- `planner_validation_valid`: True
- `planner_validation_reason`: `ok`
- `planner_rewrite_used`: False
- `executor_native_format_valid`: True
- `executor_action_valid`: True
- `validation_reason`: `ok`

**planner_raw_output**

```text
Approach the green closed door.
```

**planner_rewritten_output**

```text

```

**planner_message**

```text
Approach the green closed door
```

**executor_raw_output**

```text
Thought:
Toggle and go through door.

Action:
toggle and go through green closed door 1
```

**executor_normalized_output**

```text
Thought:
Toggle and go through door.

Action:
toggle and go through green closed door 1
```

**available_actions**

```text
["turn left", "turn right", "move forward", "pickup purple box 1", "toggle and go through green closed door 1", "go to purple box 1", "go to green closed door 1", "check available actions"]
```

## warning_step90_prompt_leak

- `training_step`: 90
- `item_id`: 940
- `round`: 1
- `planner_validation_valid`: False
- `planner_validation_reason`: `too_long`
- `planner_rewrite_used`: True
- `executor_native_format_valid`: False
- `executor_action_valid`: False
- `validation_reason`: `invalid_format`

**planner_raw_output**

```text
[Planner]
[Executor Check]
[Environment Observation]
Your goal: pick
```

**planner_rewritten_output**

```text
Check options now
```

**planner_message**

```text
Check options now
```

**executor_raw_output**

```text
Thought:
[Executor Turn]
[Environment Observation]
Your goal: pick up the green ball
In front of you in this room, you can see several objects: The room has walls around you. You are facing a wall 1 steps away. You are not carrying anything.
Available actions: ["turn left", "turn right", "check available actions"]

[Planner Guidance]
[Executor Check
```

**executor_normalized_output**

```text
Thought:
[Executor Turn]
[Environment Observation]
Your goal: pick up the green ball
In front of you in this room, you can see several objects: The room has walls around you. You are facing a wall 1 steps away. You are not carrying anything.
Available actions: ["turn left", "turn right", "check available actions"]

[Planner Guidance]
[Executor Check
```

**available_actions**

```text
["turn left", "turn right", "check available actions"]
```

## hard_collapse_step94_role_contamination

- `training_step`: 94
- `item_id`: 1750
- `round`: 2
- `planner_validation_valid`: False
- `planner_validation_reason`: `too_long`
- `planner_rewrite_used`: True
- `executor_native_format_valid`: False
- `executor_action_valid`: False
- `validation_reason`: `invalid_format`

**planner_raw_output**

```text
[Planner Review] [Executor Check door for ball.]

[Executor Check]

```

**planner_rewritten_output**

```text
Face red door
```

**planner_message**

```text
Face red door
```

**executor_raw_output**

```text
[Executor Review] No ball. Check door.

Action:
Go to red closed door 1
```

**executor_normalized_output**

```text
[Executor Review] No ball. Check door.

Action:
Go to red closed door 1
```

**available_actions**

```text
["turn left", "turn right", "move forward", "toggle and go through red closed door 1", "go to red closed door 1", "check available actions"]
```

## terminal_step98_recursive_scaffold

- `training_step`: 98
- `item_id`: 2698
- `round`: 1
- `planner_validation_valid`: False
- `planner_validation_reason`: `too_long`
- `planner_rewrite_used`: True
- `executor_native_format_valid`: False
- `executor_action_valid`: False
- `validation_reason`: `invalid_format`

**planner_raw_output**

```text
[Planner]
[Executor Turn]
[Environment Observation]
[Executor Turn]

```

**planner_rewritten_output**

```text
Approach purple ball
```

**planner_message**

```text
Approach purple ball
```

**executor_raw_output**

```text
[Executor Turn]
Thought:
[Executor Turn]
[Environment Observation]
Your goal: pick up the box
In front of you in this room, you can see several objects: There is a purple ball 1 right in front of you 2 steps away.  There is a purple ball 2 0 steps in front of you and 2 steps to your left.  There is a
```

**executor_normalized_output**

```text
[Executor Turn]
Thought:
[Executor Turn]
[Environment Observation]
Your goal: pick up the box
In front of you in this room, you can see several objects: There is a purple ball 1 right in front of you 2 steps away.  There is a purple ball 2 0 steps in front of you and 2 steps to your left.  There is a
```

**available_actions**

```text
["turn left", "turn right", "move forward", "pickup purple ball 1", "pickup purple ball 2", "pickup purple key 1", "pickup blue box 1", "pickup green key 1", "pickup green key 2", "go to purple ball 1", "go to purple ball 2", "go to purple key 1", "go to blue box 1", "go to green key 1", "go to green key 2", "check available actions"]
```

## terminal_step100_recursive_scaffold

- `training_step`: 100
- `item_id`: 1488
- `round`: 1
- `planner_validation_valid`: False
- `planner_validation_reason`: `too_long`
- `planner_rewrite_used`: True
- `executor_native_format_valid`: False
- `executor_action_valid`: False
- `validation_reason`: `invalid_format`

**planner_raw_output**

```text
[Planner]
[Executor Turn]
[Environment Observation]
[Executor Response]

```

**planner_rewritten_output**

```text
Approach blue ball
```

**planner_message**

```text
Approach blue ball
```

**executor_raw_output**

```text
[Executor Turn]
Thought:
[Executor Turn]
[Environment Observation]
[Executor Response]
[Reviewers Turn]
[Executor Response]
Thought:
[Executor Turn]
[Environment Observation]
[Executor Response]
[Reviewers Turn]
[Executor Response]
Thought:
[Executor Turn]
[Environment Observation]
[Executor Response]
[Reviewers Turn]
[Executor Response]
Thought:
[Executor Turn]
[
```

**executor_normalized_output**

```text
[Executor Turn]
Thought:
[Executor Turn]
[Environment Observation]
[Executor Response]
[Reviewers Turn]
[Executor Response]
Thought:
[Executor Turn]
[Environment Observation]
[Executor Response]
[Reviewers Turn]
[Executor Response]
Thought:
[Executor Turn]
[Environment Observation]
[Executor Response]
[Reviewers Turn]
[Executor Response]
Thought:
[Executor Turn]
[
```

**available_actions**

```text
["turn left", "turn right", "pickup blue ball 1", "go to blue ball 1", "check available actions"]
```

