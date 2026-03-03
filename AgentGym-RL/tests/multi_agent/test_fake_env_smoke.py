from conftest import load_multi_agent_module


planner_executor = load_multi_agent_module(
    "verl.multi_agent.workers.rollout.agent_vllm_rollout.planner_executor",
    "verl/multi_agent/workers/rollout/agent_vllm_rollout/planner_executor.py",
)


class FakeEnv:
    def __init__(self):
        self.actions = []
        self.done = False

    def step(self, action):
        self.actions.append(action)
        self.done = len(self.actions) >= 2
        return {"observation": f"obs-{len(self.actions)}", "reward": float(len(self.actions)), "done": self.done}


def test_max_rounds_counts_environment_actions_not_speaker_turns():
    env = FakeEnv()
    max_rounds = 2
    rounds = 0
    observation = "obs-0"
    while rounds < max_rounds:
        planner_prompt = planner_executor.build_planner_turn_prompt(observation)
        planner_message = f"Planner: derived from {planner_prompt}"
        executor_prompt = planner_executor.build_executor_turn_prompt(observation, planner_message)
        executor_message = f"Executor: action from {executor_prompt}"
        result = env.step(planner_executor.strip_speaker_prefix(executor_message, "Executor"))
        observation = result["observation"]
        rounds += 1
        if result["done"]:
            break
    assert len(env.actions) == 2
    assert rounds == 2
