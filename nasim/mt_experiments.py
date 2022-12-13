"""Script for running Experiment
"""

#MT-T5

from nasim.envs import NASimEnv
from nasim.envs.host_vector import HostVector
from nasim.scenarios.mt_generator import MTScenarioGenerator

from nasim.agents.mt_careful_agent import run_careful_agent
from nasim.agents.mt_standard_agent import run_standard_agent
from nasim.agents.mt_aggressiv_agent import run_aggressiv_agent

import csv

from multiprocessing import Pool

def generate_scenario(num_hosts, num_services, **params):
    """Generate Scenario from network parameters.

    Parameters
    ----------
    num_hosts : int
        number of hosts to include in network (minimum is 3)
    num_services : int
        number of services to use in environment (minimum is 1)
    params : dict, optional
        generator params (see :class:`ScenarioGenertor` for full list)

    Returns
    -------
    Scenario
        a new scenario object
    """
    generator = MTScenarioGenerator()
    return generator.generate(num_hosts, num_services, **params)

"""
Update successful exploits/privescs in final state

#MT-T5

Parameters
----------
state_tensor : Tensor
    final state tensor used to count unique access

Returns
-------
unique_success
    unique success counter

"""
def calculate_unique_success(state_tensor):
    unique_success = [0, -1]
    rows = state_tensor.shape[0]
    for row in range(rows):
        host_vector = HostVector(state_tensor[row])
        access = host_vector.access
        if access != 0:
            unique_success[int(access)-1] += 1
    return unique_success

def generate_env(num_honeypots, num_hosts, movement_time, one_goal, seed):
    num_sensitive = 3

    num_hosts = num_hosts + num_honeypots + num_sensitive

    num_services = 10
    generate_dict={
        "num_os":1,
        "num_sensitive":num_sensitive, 
        "num_honeypots":num_honeypots, #iterieren
        "num_processes":10,
        "num_exploits":10,
        "num_privescs":10,
        "num_vulns":10,
        "num_creds":None, #weglassen
        "r_sensitive":1000,
        "r_honeypot":-1000,
        "exploit_cost":1,
        "exploit_probs":1.0,
        "privesc_cost":1,
        "privesc_probs":1.0,
        "service_scan_cost":1,
        "os_scan_cost":1,
        "subnet_scan_cost":1,
        "process_scan_cost":1,
        "vul_scan_cost":1,
        "wiretapping_cost":1,
        "uniform":True,
        "alpha_H":2.0,
        "alpha_V":2.0,
        "lambda_V":1.0,
        "base_host_value":1,
        "host_discovery_value":1,
        "seed":seed,
        "name":None,
        "step_limit":3000, #erst mal okay
        "movement_time":movement_time, #iteriert
        "addresses": 256
    }
    
    scenario = generate_scenario(num_hosts, num_services, **generate_dict)
    env_kwargs = {"fully_obs": False,
                  "flat_actions": False,
                  "flat_obs": False,
                  "one_goal": one_goal}
    #print(scenario.get_description())

    env =  NASimEnv(scenario, **env_kwargs)
    #print(env.network)
    return env

def run_one_agent(agent, env):
    steps, reward, goal, total_success = agent(env, verbose=False)  

    unique_success = calculate_unique_success(env.current_state.tensor) 

    if goal:
        won = 1
    if env.honeypot_hit:
        won = 0
    if not goal and not env.honeypot_hit:
        won = -1

    return won, steps, unique_success, total_success

def write_to_file(agent, seed, one_goal, num_hosts, num_honeypots, movement_time, won, steps, unique_success, total_success):
    path = "results/" + str(agent) + "_" + str(seed) + "_" + str(one_goal) + "_" + str(num_hosts) + "_" + str(num_honeypots) + "_" + str(movement_time) + ".csv"
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        if movement_time == None:
            movement_time = "-"
        writer.writerow([agent, seed, one_goal, num_hosts, num_honeypots, movement_time, won, steps, unique_success, total_success])

# MT-T6
def func_to_pool(agent, seed, one_goal, num_hosts, num_honeypots, movement_time):
    env = generate_env(num_honeypots, num_hosts, movement_time, one_goal, seed)  
    won, steps, unique_success, total_success = run_one_agent(agent, env)
    agent = agent.__name__.split("_")[1]
    # print(str(agent) + ", " + str(seed) + ", " + str(one_goal) + ", " + 
    #         str(num_hosts) + ", " + str(num_honeypots) + ", " + str(movement_time) + ", " + 
    #         str(won) + ", " + str(steps) + ", " + str(unique_success) + ", " + str(total_success))
    write_to_file(agent, seed, one_goal, num_hosts, num_honeypots, movement_time, won, steps, unique_success, total_success)

# MT-T6
def pool_func(params, tries):
    for t in range(tries): #so there will be no deadlock with file resource
        with Pool() as pool:
            L = pool.starmap(func_to_pool, params)
        print("Pool done " + str(t))
    
if __name__ == "__main__":

    num_honeypots_options = [0, 2, 4, 6, 8, 10] #iterate - 0-10 in 2er steps
    num_hosts_options = [10, 50] #10,50
    movement_time_options = [None, 25, 50, 75, 100] #iterate - None oder 25-100 in 25iger steps
    one_goal_options = [True, False] #Gewinnstrategie
    seed_options = [1234, 42, 24121997] #3

    agents = [run_careful_agent, run_standard_agent, run_aggressiv_agent]
    # = 1.080

    tries = 100 #100 erst mal
    # = 108.000

    params = []

    line_break = f"\n{'-' * 60}"
    print(line_break)
    print(f"Running Generation MT Experiment")

    for i in range(len(agents)):
        #print(line_break)
        #print("Agent " + str(i) + " " + str(agents[i].__name__))
        agent = agents[i]#.__name__.split("_")[1]
        for seed in seed_options:
            for one_goal in one_goal_options:
                for num_hosts in num_hosts_options:
                    for num_honeypots in num_honeypots_options:
                        for movement_time in movement_time_options:
    # MT-T6
                            t = (agent, seed, one_goal, num_hosts, num_honeypots, movement_time)
                            params.append(t) 
    pool_func(params, tries)       