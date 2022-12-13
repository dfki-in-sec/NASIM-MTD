"""A aggressiv agent that chooses random exploit

For MT-T

For env:
- flat_actions=False
- flat_obs=False

"""

#MT-T3

from nasim.scenarios.utils import EXPLOIT_CREDENTIALS_NEEDED
import numpy as np

import nasim
from nasim.envs.action import Exploit, PrivilegeEscalation
from nasim.envs.host_vector import HostVector

import random

from time import sleep

LINE_BREAK = "-"*60

"""Get Index of correct Action type

return -1 if action type not found

Parameters
----------
env : Environment
    The environment
type : String
    Name of Action Type

Returns
-------
idx
    index of action type

"""
def get_action_type_index(env, type):
    for i, atype in enumerate(env.action_space.action_types):
        if type == atype.__name__:
            return i
    return -1

"""Generate Next Action
- only for one subnet

Parameters
----------
env : Environment
    The environment the action is executed
agent_obs : Observation
    the observation that the agent has accumulated
actions_arr : Array[Action]
    Array with actions that should be executed next
phase : int
    Phase of agent - to choose next action

Returns
-------
Action
    The next Action to make
Array[Action] 
    Updated actions_arr
Int
    Next Phase

"""
def next_action(env, agent_obs, actions_arr, phase, access, new_compromised):
    penbox = (1,0) #? find penbox?

    # Wiretapping if compromised
    if access == 2:
        idx = get_action_type_index(env, "Wiretapping")
        action = env.action_space.get_action([idx, new_compromised[0]-1, new_compromised[1], 0, 0])
        return action, actions_arr, phase

    # use queued action
    if len(actions_arr) != 0:
        action = actions_arr.pop(random.randint(0, len(actions_arr)-1))
        return action, actions_arr, phase
    
    # create the next new actions
    if phase == 0:
        # init foodhold
        exploits = env.scenario.exploits
        found = False
        for host_idx in range(agent_obs.obs_shape[0]-1):
            host_obs_vec = HostVector(agent_obs.tensor[host_idx])
            if host_obs_vec.address == (penbox[0], penbox[1]):
                found = True
                if host_obs_vec.access < exploits['e_init']["access"]:
                    actions_arr.append(Exploit(name='e_init', target=penbox, **exploits['e_init'])) 
            break
        if not found:
            actions_arr.append(Exploit(name='e_init', target=penbox, **exploits['e_init']))
        # subnetscan scan
        idx = get_action_type_index(env, "SubnetScan")
        actions_arr.append(env.action_space.get_action([idx, penbox[0]-1, penbox[1], 0, 0]))
        action = actions_arr.pop(0)
        return action, actions_arr, 1
    if phase == 1:
        exploits = env.scenario.exploits
        exploit_names = list(exploits.keys())

        privescs = env.scenario.privescs
        privescs_names = list(privescs.keys())

        idx = random.randint(0, len(exploit_names)+len(privescs_names)-1)
        # to prevent RecurisonError
        if access == -1: # if -1 already one recursion
            idx = new_compromised[0]
            new_compromised = (idx+1, new_compromised[1])
            if idx == len(exploit_names)+len(privescs_names):
                return next_action(env, agent_obs, actions_arr, 0, 0, None)

        if idx < len(exploit_names):
            #Exploit
            e_name = exploit_names[idx]
            for host_idx in range(agent_obs.obs_shape[0]-1):
                host_obs_vec = HostVector(agent_obs.tensor[host_idx])
                if host_obs_vec.address != (0,0):
                    if host_obs_vec.access < exploits[e_name]["access"]: # only if host isn't already exploited -> doesn't work with MT, leads to Recursion error and nothing more to do -> MT detection?
                        actions_arr.append(Exploit(name=e_name, target=host_obs_vec.address, **exploits[e_name]))

            if len(actions_arr) > 0:
                action = actions_arr.pop(random.randint(0, len(actions_arr)-1))
                return action, actions_arr, phase
            else: #no actions with this exploit
                idx = random.randint(len(exploit_names), len(exploit_names)+len(privescs_names)-1)
                #return next_action(env, agent_obs, actions_arr, phase, 0, new_compromised)
        if idx >= len(exploit_names):
            # PrivEscalation
            pe_name = privescs_names[idx-len(exploit_names)]
            for host_idx in range(agent_obs.obs_shape[0]-1):
                host_obs_vec = HostVector(agent_obs.tensor[host_idx])
                if host_obs_vec.address != (0,0):
                    if host_obs_vec.access < privescs[pe_name]["access"]: # only if host isn't already exploited
                        actions_arr.append(PrivilegeEscalation(name=pe_name, target=host_obs_vec.address, **privescs[pe_name]))

            if len(actions_arr) > 0:
                action = actions_arr.pop(random.randint(0, len(actions_arr)-1))
                return action, actions_arr, phase
            else: #no actions with this privesc
                if new_compromised == None:
                    new_compromised = (0,0)
                return next_action(env, agent_obs, actions_arr, phase, -1, new_compromised)


"""
Update the Observations from the Agent

Parameters
----------
agent_obs : Observation
    the observation that the agent has accumulated
new_obs : Observation
    the observation, that are returned from newest action

Returns
-------
Observation
    the updated observation from the agent
"""
def update_agent_obs(agent_obs, new_obs, with_acc): #? for flat_obs
    for row in range(agent_obs.obs_shape[0]):
        if not with_acc: #MT-T5
            host_obs_vec = HostVector(agent_obs.tensor[row])
            acc_idx = host_obs_vec._access_idx
            for column in range(agent_obs.obs_shape[1]):
                if new_obs[row][column] != 0 and column != acc_idx:
                    agent_obs.tensor[row][column] = new_obs[row][column]
        else:
            for column in range(agent_obs.obs_shape[1]):
                if new_obs[row][column] != 0:
                    agent_obs.tensor[row][column] = new_obs[row][column]
    return agent_obs

"""Check if last Observation shows newly accessed host

Parameters
----------
obs : Observation
    the last observation

Returns
-------
Address
    Address of newly accessed host
    if no new compromise -> is None
Access
    the amount of access
    if no new compromise -> is 0
"""
def check_new_access(obs):
    for host_idx in range(obs.shape[0]-1):
        host_obs_vec = HostVector(obs[host_idx])
        if host_obs_vec.access != 0 and not host_obs_vec.is_running_os("Penbox"):
            return host_obs_vec.access, host_obs_vec.address
    return 0, None

"""
Reset the Observations from the Agent

#MT-T5 (prev MT-T4)

Parameters
----------
env : Environment
    the environment with the state
agent_obs : Observation
    the observation that the agent has accumulated

Returns
-------
Observation
    the setback observation from the agent - only (1,0)=Penbox will stay
"""
def setback_agent_obs(env, agent_obs):
    new_obs = env.current_state.get_initial_observation(env.fully_obs)
    for row in range(agent_obs.obs_shape[0]):
        host_obs_vec = HostVector(agent_obs.tensor[row])
        if host_obs_vec.address == (1,0): #? find penbox
            for column in range(agent_obs.obs_shape[1]):
                new_obs.tensor[row][column] = agent_obs.tensor[row][column]
    return new_obs

"""Check if attacked Host has been discovered

#MT-T5

Discovered defines if it is an empty host - so MT

Parameters
----------
env : Environment
    the environment with the state
action : action
    the last done action

Returns
-------
discovered : int
    returns if host is discovered or not
"""
def check_if_discovered(env, action):
    discovered = env.current_state.host_discovered(action.target)
    if discovered == 1.0:
        return True
    else: 
        return False

"""
Update successful exploits/privescs

#MT-T5

Parameters
----------
total_success : int[]
    counter for access level 1 and 2
action : Action
    the last action
obs : tensor
    the observation from current action
agent_obs : Observation
    the observation that the agent has accumulated

Returns
-------
total_success
    updated counter

"""
def update_total_success(total_success, action, obs, agent_obs):
    if action.is_exploit() or action.is_privilege_escalation():
        success = obs[agent_obs.aux_row][agent_obs._success_idx]
        if success:
            for row in range(agent_obs.obs_shape[0]):
                host_obs_vec = HostVector(agent_obs.tensor[row])
                if host_obs_vec.address == action.target:
                    access = obs[row][host_obs_vec._access_idx]
                    total_success[int(access)-1] += 1
                    return total_success
    return total_success

def run_aggressiv_agent(env, step_limit=1e6, verbose=True, render_mode="readable"):
    if verbose:
        print(LINE_BREAK)
        print("STARTING EPISODE")
        print(LINE_BREAK)
        #print(f"t: Reward")

    env.reset()
    agent_obs = env.current_state.get_initial_observation(env.fully_obs)
    total_reward = 0
    done = False
    t = 0
    a = 0
    actions_arr = []
    phase = 0
    new_compromised = None
    access = 0

    total_success = [0,-1]

    if env.scenario.step_limit != None:
        step_limit = env.scenario.step_limit

    while not done and t < step_limit:
        action, actions_arr, phase = next_action(env, agent_obs, actions_arr, phase, access, new_compromised)
        obs, r, done, _ = env.step(action)
        total_reward += r
        # do not update access if scan #MT-T5
        if action.is_scan():
            with_acc = False
        else:
            with_acc = True
        agent_obs = update_agent_obs(agent_obs, obs, with_acc)
        access, new_compromised = check_new_access(obs)

        #update total times exploit/privesc was successful
        total_success = update_total_success(total_success, action, obs, agent_obs)

        if action.is_process_scan():
            access = 0
            new_compromised = None
      
        if verbose:
            print("Step: " + str(t) + " - Phase: " + str(phase))
            print(action)
            print(LINE_BREAK)
            #env.render(render_mode, obs)
            #env.render(render_mode, agent_obs)
            #print(LINE_BREAK)
            sleep(1)
        if (t+1) % 100 == 0 and verbose:
            print(f"{t}: {total_reward}")
        t += 1

        #MT-T5
        discovered = check_if_discovered(env, action)
        if not discovered:
            if verbose:
                print("Moving Target detected!")
            discovered = True
            actions_arr = []
            phase = 0
            agent_obs = setback_agent_obs(env, agent_obs)

    if done and verbose:
        print(LINE_BREAK)
        print("EPISODE FINISHED")
        print(LINE_BREAK)
        print(f"Total steps = {t}")
        print(f"Total reward = {total_reward}")
    elif verbose:
        print(LINE_BREAK)
        print("STEP LIMIT REACHED")
        print(LINE_BREAK)

    if done:
        done = env.goal_reached()

    return t, total_reward, done, total_success


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str,
                        help="benchmark scenario name")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="random seed")
    parser.add_argument("-r", "--runs", type=int, default=1,
                        help="number of runs to perform (default=1)")
    parser.add_argument("-o", "--partially_obs", action="store_true",
                        help="Partially Observable Mode")
    parser.add_argument("-p", "--param_actions", action="store_true",
                        help="Use Parameterised action space")
    parser.add_argument("-f", "--box_obs", action="store_true",
                        help="Use 2D observation space")
    args = parser.parse_args()

    seed = args.seed
    run_steps = []
    run_rewards = []
    run_goals = 0
    run_honeypots = 0 #MT-T2
    for i in range(args.runs):
        env = nasim.make_benchmark(args.env_name,
                                   seed,
                                   not args.partially_obs,
                                   not args.param_actions,
                                   not args.box_obs)
        steps, reward, done = run_aggressiv_agent(env, verbose=False)
        run_steps.append(steps)
        run_rewards.append(reward)
        run_goals += int(done)
        run_honeypots += int(env.honeypot_hit) #MT-T2
        seed += 1

        if args.runs > 1:
            print(f"Run {i}:")
            print(f"\tSteps = {steps}")
            print(f"\tReward = {reward}")
            print(f"\tGoal reached = {done}")
            print(f"\tHoneypots attacked = {env.honeypot_hit}") #MT-T2

    run_steps = np.array(run_steps)
    run_rewards = np.array(run_rewards)

    print(LINE_BREAK)
    print("Aggressiv Agent Runs Complete")
    print(LINE_BREAK)
    print(f"Mean steps = {run_steps.mean():.2f} +/- {run_steps.std():.2f}")
    print(f"Mean rewards = {run_rewards.mean():.2f} "
          f"+/- {run_rewards.std():.2f}")
    true_goals = run_goals - run_honeypots #MT-T2
    print(f"Goals reached = {true_goals} / {args.runs}") #MT-T2
