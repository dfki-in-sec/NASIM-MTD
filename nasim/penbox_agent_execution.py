import os.path as osp

import nasim
from nasim.agents.Penbox_in_Nasim import penbox_agent

env = nasim.load("scenarios/benchmark/s1_rotation/s1_r_a_0.yaml")
line_break = f"\n{'-' * 60}"
print(line_break)
print(f"Running Demo on Penbox environment")
print("Player controlled")
ret, steps, goal = penbox_agent(env, "readable")
print(line_break)
print("Episode Complete")
print(line_break)
if goal:
    print("Goal accomplished. Sensitive data retrieved!")
print(f"Final Score={ret}")
print(f"Steps taken={steps}")

envs = ["s1_r_a_0.yaml", "s1_r_a_1.yaml", "s1_r_a_2.yaml", "s1_r_a_3.yaml", "s1_r_a_4.yaml", "s1_r_a_5.yaml",
        "s1_r_a_6.yaml", "s1_r_a_7.yaml", "s1_r_a_8.yaml", "s1_r_a_9.yaml", "s1_r_a_10.yaml", "s1_r_a_11.yaml",
        "s1_r_a_12.yaml", "s1_r_a_13.yaml", "s1_r_a_14.yaml", "s1_r_a_15.yaml", "s1_r_a_16.yaml", "s1_r_a_17.yaml",
        "s1_r_a_18.yaml", "s1_r_a_19.yaml", "s1_r_a_20.yaml", "s1_r_a_21.yaml", "s1_r_a_22.yaml", "s1_r_a_23.yaml"]

results = []
for idx, env in enumerate(envs):
    env = nasim.load("./scenarios/benchmark/s1_rotation/" + env, flat_actions=True, flat_obs=True)
    ret, steps, goal = penbox_agent(env, "readable")
    print(f"Final Score={ret}")
    print(f"Steps taken={steps}")
    results.append([idx,ret, steps, goal])
with open("results1.txt", "a") as f:
    print(results, file=f)