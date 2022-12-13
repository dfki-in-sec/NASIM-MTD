"""Script for running Moving Target Demo - Agents

Usage
-----

$ python mt-demo
"""

#MT-T3

import os.path as osp

import nasim
from nasim.agents.mt_careful_agent import run_careful_agent
from nasim.agents.mt_standard_agent import run_standard_agent
from nasim.agents.mt_aggressiv_agent import run_aggressiv_agent

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

    env = nasim.load("nasim/scenarios/benchmark/mt-szenario.yaml", flat_obs=False, flat_actions=False) #6,4, (59), (49)
    #env = nasim.load("nasim/scenarios/benchmark/mt-szenario-simple.yaml", flat_obs=False, flat_actions=False) #6,4,18

    line_break = f"\n{'-' * 60}"
    print(line_break)
    print(f"Running Demo2 on MT environment")
    #steps, reward, goal = run_careful_agent(env)
    steps, reward, goal = run_standard_agent(env)
    #steps, reward, goal = run_aggressiv_agent(env)    
    print(line_break)
    print("Episode Complete")
    print(line_break)
    if goal:
        print("Goal accomplished. Sensitive data retrieved!")
    #MT-T2
    if env.honeypot_hit:
        print("Caught by Honeypot!")
    print(f"Final Score={reward}")
    print(f"Steps taken={steps}")

