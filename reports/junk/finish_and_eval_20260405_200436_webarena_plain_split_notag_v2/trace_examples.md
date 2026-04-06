# Trace-focused examples

- Remote trace directory: `/home/mavinomichael/agentgym_runs/logs/webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2/trace_train`
- Local trace mirror: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/finish_and_eval_20260405_200436_webarena_plain_split_notag_v2/artifacts_full/trace_train`
- Total traced packets: `3983`
- `planner_validation_reason=tag_only`: `3954`
- `planner_validation_reason=ok`: `28`

## Earliest collapse trace

- Earliest traced `tag_only` event appears at `training_step 2` on item `372`.
- Source: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/finish_and_eval_20260405_200436_webarena_plain_split_notag_v2/artifacts_full/trace_train/executor_payload_trace_rank6.jsonl:1`

## Non-collapse examples

### Valid example 1

- Source: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/finish_and_eval_20260405_200436_webarena_plain_split_notag_v2/artifacts_full/trace_train/executor_payload_trace_rank0.jsonl:1`
- `training_step`: `1`
- `item_id`: `330`
- `planner_validation_reason`: `ok`
- `executor_native_format_valid`: `True`
- `executor_action_valid`: `True`

**observation_excerpt**
```text
OBSERVATION:
Tab 0 (current): One Stop Market

[1] RootWebArea 'One Stop Market' focused: True
	[1277] link 'My Account'
	[1275] link 'My Wish List'
	[1279] link 'Sign Out'
	[1282] StaticText 'Welcome, Emma Lopez!'
	[1188] link 'Skip to Content'
	[1197] link 'store logo'
		[1285] img 'one_stop_market_logo'
	[1286] link '\ue611 My Cart 1 1items'
	[2497] StaticText 'Search'
	[1655] combobox '\ue615 Search' autocomplete: both hasPopup: listbox required: False expanded: False
	[2500] link 'Advanced Search'
	[1658] button 'Search' disabled: True
	[1066] tablist '' multiselectable: False orientation: horizontal
		[1068] tabpanel ''
			[10] menu '' orientation: vertical
				[761] menuitem '\ue622 Beauty & Personal Care' hasPopup: menu
				[826] menuitem '\ue622 Sports & Outdoors' hasPopup: menu
	...[truncated]
```

**planner_raw_output**
```text
Check the navigation menu for any option related to order history or customer account.
```

**executor_raw_output**
```text
In summary, the next action I will perform is ```click [1277]```.

This action is chosen because the 'My Account' link in the navigation menu is the most likely option to provide access to order history or customer account information.
```

### Valid example 2

- Source: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/finish_and_eval_20260405_200436_webarena_plain_split_notag_v2/artifacts_full/trace_train/executor_payload_trace_rank0.jsonl:2`
- `training_step`: `1`
- `item_id`: `330`
- `planner_validation_reason`: `ok`
- `executor_native_format_valid`: `True`
- `executor_action_valid`: `True`

**observation_excerpt**
```text
OBSERVATION:
Tab 0 (current): My Account

[2704] RootWebArea 'My Account' focused: True
	[4078] link 'My Account'
	[4076] link 'My Wish List'
	[4080] link 'Sign Out'
	[4083] StaticText 'Welcome, Emma Lopez!'
	[3907] link 'Skip to Content'
	[3916] link 'store logo'
		[4086] img 'one_stop_market_logo'
	[4087] link '\ue611 My Cart 1 1items'
	[5067] StaticText 'Search'
	[4341] combobox '\ue615 Search' autocomplete: both hasPopup: listbox required: False expanded: False
	[5070] link 'Advanced Search'
	[4344] button 'Search' disabled: True
	[3769] tablist '' multiselectable: False orientation: horizontal
		[3771] tabpanel ''
			[2713] menu '' orientation: vertical
				[3464] menuitem '\ue622 Beauty & Personal Care' hasPopup: menu
				[3529] menuitem '\ue622 Sports & Outdoors' hasPopup: menu
				...[truncated]
```

**planner_raw_output**
```text
Click on 'My Orders' to view order history.
```

**executor_raw_output**
```text
In summary, the next action I will perform is ```click [4311]```.

This action is chosen because it leads to the order history, which may contain the necessary information to determine the amount spent in March 2023.
```

### Valid example 3

- Source: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/finish_and_eval_20260405_200436_webarena_plain_split_notag_v2/artifacts_full/trace_train/executor_payload_trace_rank0.jsonl:3`
- `training_step`: `1`
- `item_id`: `330`
- `planner_validation_reason`: `ok`
- `executor_native_format_valid`: `True`
- `executor_action_valid`: `True`

**observation_excerpt**
```text
OBSERVATION:
Tab 0 (current): My Orders

[5090] RootWebArea 'My Orders' focused: True
	[6395] link 'My Account'
	[6393] link 'My Wish List'
	[6397] link 'Sign Out'
	[6400] StaticText 'Welcome, Emma Lopez!'
	[6293] link 'Skip to Content'
	[6302] link 'store logo'
		[6403] img 'one_stop_market_logo'
	[6404] link '\ue611 My Cart 1 1items'
	[7469] StaticText 'Search'
	[6738] combobox '\ue615 Search' autocomplete: both hasPopup: listbox required: False expanded: False
	[7472] link 'Advanced Search'
	[6741] button 'Search' disabled: True
	[6155] tablist '' multiselectable: False orientation: horizontal
		[6157] tabpanel ''
			[5099] menu '' orientation: vertical
				[5850] menuitem '\ue622 Beauty & Personal Care' hasPopup: menu
				[5915] menuitem '\ue622 Sports & Outdoors' hasPopup: menu
				[5...[truncated]
```

**planner_raw_output**
```text
Find and click on an order from March 2023 to view its details.
```

**executor_raw_output**
```text
In summary, the next action I will perform is ```click [6447]```.

This action is chosen because the order '000000180' from 3/11/23 has a status of 'Complete', indicating it was paid for. Clicking on it will allow us to view the details and calculate the amount spent in March 2023.
```

## Collapse examples

### Collapse example 1

- Source: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/finish_and_eval_20260405_200436_webarena_plain_split_notag_v2/artifacts_full/trace_train/executor_payload_trace_rank0.jsonl:8`
- `training_step`: `3`
- `item_id`: `35`
- `planner_validation_reason`: `tag_only`
- `executor_native_format_valid`: `False`
- `executor_action_valid`: `False`

**observation_excerpt**
```text
OBSERVATION:
Tab 0 (current): OpenStreetMap

[1] RootWebArea 'OpenStreetMap' focused: True
	[40] heading 'OpenStreetMap logo OpenStreetMap'
		[68] link 'OpenStreetMap logo OpenStreetMap'
			[71] img 'OpenStreetMap logo'
	[489] link 'Edit'
	[490] button ''
	[606] link 'History'
	[607] link 'Export'
	[608] link 'GPS Traces'
	[609] link 'User Diaries'
	[610] link 'Communities'
	[611] link 'Copyright'
	[612] link 'Help'
	[613] link 'Donate'
	[614] link 'About'
	[75] button 'Select Language'
	[503] link 'Log In'
	[504] link 'Sign Up'
	[10] textbox 'Search' focused: True required: False
	[652] button 'Where is this?'
	[623] button 'Go'
	[621] link 'Find directions between two points'
	[592] button 'Close'
	[593] heading 'Welcome to OpenStreetMap!'
	[626] StaticText 'OpenStreetMap is a map of the...[truncated]
```

**planner_raw_output**
```text
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

**executor_raw_output**
```text
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

### Collapse example 2

- Source: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/finish_and_eval_20260405_200436_webarena_plain_split_notag_v2/artifacts_full/trace_train/executor_payload_trace_rank0.jsonl:9`
- `training_step`: `4`
- `item_id`: `49`
- `planner_validation_reason`: `tag_only`
- `executor_native_format_valid`: `False`
- `executor_action_valid`: `False`

**observation_excerpt**
```text
OBSERVATION:
Tab 0 (current): One Stop Market

[1] RootWebArea 'One Stop Market' focused: True
	[1257] link 'My Account'
	[1255] link 'My Wish List'
	[1259] link 'Sign Out'
	[1262] StaticText 'Welcome, Emma Lopez!'
	[1168] link 'Skip to Content'
	[1177] link 'store logo'
		[1265] img 'one_stop_market_logo'
	[1266] link '\ue611 My Cart'
	[2430] StaticText 'Search'
	[1588] combobox '\ue615 Search' autocomplete: both hasPopup: listbox required: False expanded: False
	[2433] link 'Advanced Search'
	[1591] button 'Search' disabled: True
	[1066] tablist '' multiselectable: False orientation: horizontal
		[1068] tabpanel ''
			[10] menu '' orientation: vertical
				[761] menuitem '\ue622 Beauty & Personal Care' hasPopup: menu
				[826] menuitem '\ue622 Sports & Outdoors' hasPopup: menu
				[836] ...[truncated]
```

**planner_raw_output**
```text
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

**executor_raw_output**
```text
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

### Collapse example 3

- Source: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/finish_and_eval_20260405_200436_webarena_plain_split_notag_v2/artifacts_full/trace_train/executor_payload_trace_rank0.jsonl:10`
- `training_step`: `6`
- `item_id`: `766`
- `planner_validation_reason`: `tag_only`
- `executor_native_format_valid`: `False`
- `executor_action_valid`: `False`

**observation_excerpt**
```text
OBSERVATION:
Tab 0 (current): OpenStreetMap

[1] RootWebArea 'OpenStreetMap' focused: True
	[40] heading 'OpenStreetMap logo OpenStreetMap'
		[68] link 'OpenStreetMap logo OpenStreetMap'
			[71] img 'OpenStreetMap logo'
	[489] link 'Edit'
	[490] button ''
	[606] link 'History'
	[607] link 'Export'
	[608] link 'GPS Traces'
	[609] link 'User Diaries'
	[610] link 'Communities'
	[611] link 'Copyright'
	[612] link 'Help'
	[613] link 'Donate'
	[614] link 'About'
	[75] button 'Select Language'
	[503] link 'Log In'
	[504] link 'Sign Up'
	[10] textbox 'Search' focused: True required: False
	[652] button 'Where is this?'
	[623] button 'Go'
	[621] link 'Find directions between two points'
	[592] button 'Close'
	[593] heading 'Welcome to OpenStreetMap!'
	[626] StaticText 'OpenStreetMap is a map of the...[truncated]
```

**planner_raw_output**
```text
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

**executor_raw_output**
```text
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

### Collapse example 4

- Source: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/finish_and_eval_20260405_200436_webarena_plain_split_notag_v2/artifacts_full/trace_train/executor_payload_trace_rank0.jsonl:11`
- `training_step`: `7`
- `item_id`: `256`
- `planner_validation_reason`: `tag_only`
- `executor_native_format_valid`: `False`
- `executor_action_valid`: `False`

**observation_excerpt**
```text
OBSERVATION:
Tab 0 (current): OpenStreetMap

[1] RootWebArea 'OpenStreetMap' focused: True
	[40] heading 'OpenStreetMap logo OpenStreetMap'
		[68] link 'OpenStreetMap logo OpenStreetMap'
			[71] img 'OpenStreetMap logo'
	[489] link 'Edit'
	[490] button ''
	[606] link 'History'
	[607] link 'Export'
	[608] link 'GPS Traces'
	[609] link 'User Diaries'
	[610] link 'Communities'
	[611] link 'Copyright'
	[612] link 'Help'
	[613] link 'Donate'
	[614] link 'About'
	[75] button 'Select Language'
	[503] link 'Log In'
	[504] link 'Sign Up'
	[10] textbox 'Search' focused: True required: False
	[652] button 'Where is this?'
	[623] button 'Go'
	[621] link 'Find directions between two points'
	[592] button 'Close'
	[593] heading 'Welcome to OpenStreetMap!'
	[626] StaticText 'OpenStreetMap is a map of the...[truncated]
```

**planner_raw_output**
```text
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

**executor_raw_output**
```text
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```
