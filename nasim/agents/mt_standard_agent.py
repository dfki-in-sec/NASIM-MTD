"""A standard agent that chooses random host

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

from time import sleep
import random

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

"""Get Ranked Exploits

Ranking based on:
- cost (higher worse)
- access (higher better)
- credentials needed (higher worse)

Parameters
----------
exploits : exploit{}
    dictionary with all possible exploits

Returns
-------
ranked_exploits
    exploits ranked

"""
def get_ranked_exploits(exploits):
    ranked_exploits = {}
    for key, exploit in exploits.items():
        if key != 'e_init':
            ranked_exploits[key] = exploit['cost'] + (exploit['credentials_needed']*3) - exploit['access']
    ranked_exploits = dict(sorted(ranked_exploits.items(), key=lambda item: item[1]))
    return ranked_exploits

"""Get Ranked Exploits

Ranking based on:
- cost (higher worse)
- access (higher better)
- credentials needed (higher worse)
- process needed (higher worse)

Parameters
----------
privescs : privescs{}
    dictionary with all possible privescs

Returns
-------
ranked_privescs
    privescs ranked

"""
def get_ranked_privescs(privescs):
    ranked_privescs = {}
    for key, privesc in privescs.items():
        if key != 'e_init':
            ranked_privescs[key] = privesc['cost'] - (privesc['credentials_tofind']) - privesc['access']
            if privesc['process'] != None:
                ranked_privescs[key] += 1
    ranked_privescs = dict(sorted(ranked_privescs.items(), key=lambda item: item[1]))
    return ranked_privescs

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
curr_host : int
    host_idx for current host choosen

Returns
-------
Action
    The next Action to make
Array[Action] 
    Updated actions_arr
Int
    Next Phase

"""
def next_action(env, agent_obs, actions_arr, phase, curr_host):
    penbox = (1,0) #? find penbox?
    used_subnet = 2 #? find correct subnet?

    # use queued action
    if len(actions_arr) != 0:
        action = actions_arr.pop(0)
        return action, actions_arr, phase, curr_host
    
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
        return action, actions_arr, 1, 0
    if phase == 1:
        compromised = True
        possible_addresse_idx = []
        for host_idx in range(agent_obs.obs_shape[0]-1):
            host_obs_vec = HostVector(agent_obs.tensor[host_idx])
            if host_obs_vec.address != (0,0) and host_obs_vec.address != penbox:
                possible_addresse_idx.append(host_idx)
        while compromised: # only if host isn't already compromised
            host_idx = random.choice(possible_addresse_idx)
            host_obs_vec = HostVector(agent_obs.tensor[host_idx])
            if (host_obs_vec.address != (0,0)) and host_obs_vec.access != 2.0:
                compromised = False
            else:
                possible_addresse_idx.remove(host_idx)
        #scans
        idx = get_action_type_index(env, "ServiceScan")
        actions_arr.append(env.action_space.get_action([idx, host_obs_vec.address[0]-1, host_obs_vec.address[1], 0, 0]))
        idx = get_action_type_index(env, "OSScan")
        actions_arr.append(env.action_space.get_action([idx, host_obs_vec.address[0]-1, host_obs_vec.address[1], 0, 0]))  
        idx = get_action_type_index(env, "VulScan")
        actions_arr.append(env.action_space.get_action([idx, host_obs_vec.address[0]-1, host_obs_vec.address[1], 0, 0]))

        if host_obs_vec.access == 1:
            idx = get_action_type_index(env, "ProcessScan")
            actions_arr.append(env.action_space.get_action([idx, host_obs_vec.address[0]-1, host_obs_vec.address[1], 0, 0]))

        action = actions_arr.pop(0)
        return action, actions_arr, 2, host_idx
    if phase == 2:
        host_obs_vec = HostVector(agent_obs.tensor[curr_host])

        if host_obs_vec.access == 0:
            # exploits
            exploits = env.scenario.exploits
            possible_exploits = []
            for e_name in get_ranked_exploits(exploits).keys():
                if (host_obs_vec.access < exploits[e_name]["access"]) and \
                        (host_obs_vec.is_running_service(exploits[e_name]['service'])) and \
                            (exploits[e_name]['os'] is None or host_obs_vec.is_running_os(exploits[e_name]['os'])) and \
                                (exploits[e_name]['vul'] == None or host_obs_vec.is_running_vul(exploits[e_name]['vul'])) and \
                                    (exploits[e_name]['credentials_needed'] == 0 or host_obs_vec.got_credentials(exploits[e_name]['credentials_needed'])):
                    possible_exploits.append(e_name)
            
            if len(possible_exploits) > 0:
                e_name = possible_exploits[0]
                target = host_obs_vec.address
                actions_arr.append(Exploit(name=e_name, target=target, **exploits[e_name]))
                action = actions_arr.pop(0)
                return action, actions_arr, 2, curr_host
            else: #if no exploit found, look for new host
                #print("NO Exploit")
                return next_action(env, agent_obs, actions_arr, 1, 0)

        elif host_obs_vec.access == 1:
            #Privilege Escalation
            privescs = env.scenario.privescs
            possible_privescs = []
            for pe_name in get_ranked_privescs(privescs).keys():
                if (host_obs_vec.access < privescs[pe_name]["access"]) and \
                            (privescs[pe_name]['os'] is None or host_obs_vec.is_running_os(privescs[pe_name]['os'])) and \
                                (privescs[pe_name]['process'] == 0 or host_obs_vec.is_running_process(privescs[pe_name]['process'])):
                    possible_privescs.append(pe_name)

            if len(possible_privescs) > 0:
                e_name = possible_privescs[0]
                target = host_obs_vec.address
                actions_arr.append(PrivilegeEscalation(name=pe_name, target=target, **privescs[pe_name]))
                action = actions_arr.pop(0)
                return action, actions_arr, 2, curr_host
            else: #if no privescalation found, look for new host
                #print("NO PrivEsc")
                return next_action(env, agent_obs, actions_arr, 1, 0)

        else:
            # Wiretapping and new host in next round
            idx = get_action_type_index(env, "Wiretapping")
            action = env.action_space.get_action([idx, host_obs_vec.address[0]-1, host_obs_vec.address[1], 0, 0])
            return action, actions_arr, 1, 0


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

"""
Reset the Observations from the Agent

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
        if host_obs_vec.address == (1,0):  #? find penbox
            for column in range(agent_obs.obs_shape[1]):
                new_obs.tensor[row][column] = agent_obs.tensor[row][column]
    return new_obs

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

def run_standard_agent(env, step_limit=1e6, verbose=True, render_mode="readable"):
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
    curr_host = 0

    total_success = [0,-1]

    if env.scenario.step_limit != None:
        step_limit = env.scenario.step_limit

    while not done and t < step_limit:
        action, actions_arr, phase, curr_host = next_action(env, agent_obs, actions_arr, phase, curr_host)
        obs, r, done, _ = env.step(action)
        total_reward += r
        # do not update access if scan #MT-T5
        if action.is_scan():
            with_acc = False
        else:
            with_acc = True
        agent_obs = update_agent_obs(agent_obs, obs, with_acc)

        #update total times exploit/privesc was successful
        total_success = update_total_success(total_success, action, obs, agent_obs)

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

        #MT detection
        success = obs[agent_obs.aux_row][agent_obs._success_idx]
        if not success:
            if verbose:
                print("Moving Target detected!")
            agent_obs = setback_agent_obs(env, agent_obs)
            phase = 0
            actions_arr = []
            curr_host = 0

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
        steps, reward, done = run_standard_agent(env, verbose=False)
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
    print("Standard Agent Runs Complete")
    print(LINE_BREAK)
    print(f"Mean steps = {run_steps.mean():.2f} +/- {run_steps.std():.2f}")
    print(f"Mean rewards = {run_rewards.mean():.2f} "
          f"+/- {run_rewards.std():.2f}")
    true_goals = run_goals - run_honeypots #MT-T2
    print(f"Goals reached = {true_goals} / {args.runs}") #MT-T2
