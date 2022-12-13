"""Script for running PenboxDemoScenario 1

Usage
-----

$ python demo [-ai] [-h] env_name
"""

import os.path as osp

import nasim
from nasim.agents.dqn_agent import DQNAgent
from nasim.agents.keyboard_agent import run_keyboard_agent, run_generative_keyboard_agent

DQN_POLICY_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)),
    "agents",
    "policies"
)
DQN_POLICIES = {
    "tiny": osp.join(DQN_POLICY_DIR, "dqn_tiny.pt"),
    "small": osp.join(DQN_POLICY_DIR, "dqn_small.pt")
}

##Set the scenario
scenario = 15

if __name__ == "__main__":
    if scenario == 1:
        env = nasim.load("scenarios/benchmark/penbox_scenario1.yaml")
    if scenario == 2:
        env = nasim.load("scenarios/benchmark/penbox_scenario2.yaml")
    if scenario == 3:
        env = nasim.load("scenarios/benchmark/penbox_scenario3.yaml")
    if scenario == 14:
        env = nasim.load("scenarios/benchmark/penbox_traint_4_static.yaml")
    if scenario == 15:
        env = nasim.load("scenarios/benchmark/penbox_eval_1.yaml")

    line_break = f"\n{'-' * 60}"
    print(line_break)
    print(f"Running Demo on Penbox environment")
    print("Player controlled")
    ret, steps, goal = run_generative_keyboard_agent(env, "readable")
    print(line_break)
    print("Episode Complete")
    print(line_break)
    if goal:
        print("Goal accomplished. Sensitive data retrieved!")
    #MT-T2
    if env.honeypot_hit:
        print("Caught by Honeypot!")
    print(f"Final Score={ret}")
    print(f"Steps taken={steps}")

