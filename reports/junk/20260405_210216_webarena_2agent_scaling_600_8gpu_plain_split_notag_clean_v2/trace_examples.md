# Trace-focused examples

- Remote trace directory: `/home/mavinomichael/agentgym_runs/logs/webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2/trace_train`
- Local copied trace directory: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260405_210216_webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2/artifacts/extracted/trace_train`
- Trace sample count: `3983`
- Planner fallback used: `3955/3983`
- Planner-valid traces: `28/3983`
- Executor native-format valid traces: `24/3983`
- Final action valid traces: `24/3983`

## Selected non-collapse example

- Training step: `1`
- Rank: `0`
- Item id: `330`
- Round: `1`
- Planner valid: `True` (`ok`)
- Planner fallback used: `False`
- Executor native format valid: `True`
- Final action valid: `True` (`ok`)

Planner message:
```text
Check the navigation menu for any option related to order history or customer account
```
Executor first-pass output:
```text
In summary, the next action I will perform is ```click [1277]```.

This action is chosen because the 'My Account' link in the navigation menu is the most likely option to provide access to order history or customer account information.
```

## Selected planner-collapse example

- Training step: `1`
- Rank: `0`
- Item id: `330`
- Round: `4`
- Planner valid: `False` (`not_intent_phrase`)
- Planner fallback used: `True`
- Executor native format valid: `True`
- Final action valid: `True` (`ok`)

Planner message:
```text
Planner guidance unavailable. Infer the next step from the observation only.
```
Executor first-pass output:
```text
In summary, the next action I will perform is ```click [9191]```.

This action is chosen because the 'Proceed to Checkout' button is the most relevant action to take next, as it will allow us to move forward with the purchase process and potentially find more detailed information about the order total for March 2023.
```

## Selected executor-format collapse example

- Training step: `1`
- Rank: `0`
- Item id: `330`
- Round: `7`
- Planner valid: `True` (`ok`)
- Planner fallback used: `False`
- Executor native format valid: `False`
- Final action valid: `False` (`invalid_format`)

Planner message:
```text
Wait for the page to load and then check the order details for the total spent
```
Executor first-pass output:
```text
In summary, the next action I will perform is ```wait [3]```.

This action is chosen because the page is currently loading, indicated by the 'Loading...' message. Waiting for a short period will allow the page to finish loading, after which we can check the order details for the total spent.
```

## Selected invalid-final-action example

- Training step: `1`
- Rank: `0`
- Item id: `330`
- Round: `7`
- Planner valid: `True` (`ok`)
- Planner fallback used: `False`
- Executor native format valid: `False`
- Final action valid: `False` (`invalid_format`)

Planner message:
```text
Wait for the page to load and then check the order details for the total spent
```
Executor first-pass output:
```text
In summary, the next action I will perform is ```wait [3]```.

This action is chosen because the page is currently loading, indicated by the 'Loading...' message. Waiting for a short period will allow the page to finish loading, after which we can check the order details for the total spent.
```

