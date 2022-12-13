"""Script for running PenboxDemoScenario 1

Usage
-----

$ python demo [-ai] [-h] env_name
"""

import os.path as osp

import nasim
from nasim.agents.debug_agent import debug_agent


if __name__ == "__main__":
    env = nasim.load("scenarios/benchmark/penbox_eval_2.yaml")
    line_break = f"\n{'-' * 60}"
    print(line_break)
    print(f"Running Demo on Penbox environment")
    print("Player controlled")
    ret, steps, goal = debug_agent(env, "readable")
    print(line_break)
    print("Episode Complete")
    print(line_break)
    if goal:
        print("Goal accomplished. Sensitive data retrieved!")
    print(f"Final Score={ret}")
    print(f"Steps taken={steps}")
