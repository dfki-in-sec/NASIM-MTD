"""Script for running Moving Target Demo

Usage
-----

$ python mt-demo
"""

#MT-T

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

if __name__ == "__main__":

    env = nasim.load("nasim/scenarios/benchmark/mt-szenario.yaml") #6,4, (59), (49)
    #env = nasim.load("nasim/scenarios/benchmark/mt-szenario-simple.yaml") #6,4,18

    line_break = f"\n{'-' * 60}"
    print(line_break)
    print(f"Running Demo on MT environment")
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

