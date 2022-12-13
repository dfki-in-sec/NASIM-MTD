"""Script for running Testing Generation

"""

#MT-T4

from nasim.envs import NASimEnv, host_vector
from nasim.scenarios.host import Host
from nasim.envs.host_vector import HostVector
from nasim.scenarios.mt_generator import MTScenarioGenerator

from nasim.agents.mt_careful_agent import run_careful_agent
from nasim.agents.mt_standard_agent import run_standard_agent
from nasim.agents.mt_aggressiv_agent import run_aggressiv_agent

from nasim.agents.keyboard_agent import run_keyboard_agent, run_generative_keyboard_agent

from time import sleep

ENDC = '\033[0m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'

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



if __name__ == "__main__":

    num_sensitive = 3
    num_honeypots = 0
    num_hosts = 10 + num_honeypots + num_sensitive 
    movement_time = 25

    one_goal = True

    num_services = 10

    generate_dict={
        "num_os":1,
        "num_sensitive":num_sensitive, 
        "num_honeypots":num_honeypots,
        "num_processes":10,
        "num_exploits":10,
        "num_privescs":10,
        "num_vulns":10,
        "num_creds":None,#
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
        "seed":1234,
        "name":None,
        "step_limit":3000,
        "movement_time":movement_time,
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
   
    agents = [run_careful_agent, run_standard_agent, run_aggressiv_agent]
    #agents = [run_aggressiv_agent]

    line_break = f"\n{'-' * 60}"
    print(line_break)
    print(f"Running Generation MT demo environment")
    print(line_break)

    for i in range(len(agents)):    
        print("\n")
        env = NASimEnv(scenario, **env_kwargs)

        #RUN AGENTS
        #print("Agent " + str(i) + " " + str(agents[i].__name__))
        #steps, reward, goal, total_success = agents[i](env, verbose=True)  

        #RUN KEYBOARD AGENT
        total_success = "-"
        reward, steps, goal = run_keyboard_agent(env) 

        unique_success = calculate_unique_success(env.current_state.tensor) 
        #print(line_break)
        #print("Episode Complete")
        print("\n")
        if goal:
            print(OKGREEN + "Goal accomplished. Sensitive data retrieved!" + ENDC)
            print("\n")
        #MT-T2
        if env.honeypot_hit:
            print(FAIL + "Caught by Honeypot!" + ENDC)
            print("\n")
        if not goal and not env.honeypot_hit:
            print(WARNING + "Too many Steps!" + ENDC)
            print("\n")
        
        print(f"Final Score={reward}")
        print(f"Steps taken={steps}")
        print(f"Unique Successes={unique_success}") #MT-T5
        print(f"Total Successes={total_success}") #MT-T5
        print(line_break)

        sleep(5)
