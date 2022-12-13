Paper Additions
===============

Main Files
----------

- evaluation_mt.ipynb
- mt_experiments.py
- results folder

Changes
-------

ADDED MT-T
- moving target
- randomly changes hosts in one subnet
- change occures after specific time (defined by cost)

ADDED MT-T2
- added extra honeypots (similar to sensitive hosts)
- simulation is done when honeypot is attacked (but not scanned)

ADDED MT-T3
- Careful agent
  - Observation for agent (updated)
  - choose action phases
    - 0 -> init + subnetscan
    - 1 -> service scan, os, vuln scan
    - >= 2 -> exploits (best first) + possible hosts in random order  
    - wiretapping everytime new host is compromised and process scan everytime new access level
    - >= 2+len(exploits) -> privilege escalations + process scan
- Standard Agent
  - choose action phases
    - 0 -> init + subnetscan
    - 1 -> choose random host + make scans (including process scan )
    - 2 -> dependend on access level do exploit or privesc or wiretapping
      - right now, choose best ranked exploit, that should work
    -> if compromised or no exploit/privesc, back to phase 1
- Aggressiv Agent
  - choose action phases
    - 0 -> init + subnetscan
    - 1 -> choose random exploit/privesc -> order of action random
    - wiretapping every time compromised host
- all agents aware of MT (agressiv does not notice)

ADDED MT-T4
Generator
- added initial Penbox with init-service/exploit + Penbox os - only for this box 
- added random sensitive host 
- added honeypots
- added credentials + vulns generation
Address space
- added 'empty hosts' in generator
- host discovery function - if empty host - will not be discovered (returns (0,0))
  - agents won't try to scan/exploit (0,0) hosts
  - but address space is available for MT

Added MT-T5
- vulnerbale host - just one needs to be visited to accomplish goal
- aggressive agent:
  - starts with phase 0 after a whole round (MT discovered)
  - remembers if exploit of host without service, then needs to scan again -> (checks in aggressiv agent, if action target discovered in state)
- Parameter
  - costs all the same
  - other parameter fixed
  - num_hosts is seperate of sensitive+honeypots - adjusted
- Userrights
  - after Mutation still Userrights -> attacker is not aware of this
  - total success = in agent after every exploit/privesc
  - unique success = outside agent at end looking at final state
  - both a size 2 array idx=0 means user and idx=1 means root 
- win-strategies parameterized 
  - 1 sensitive (one_goal) or all sensitive (!one_goal)
- Test series documented
  - Save in CSV: row=one test series, spalte=Variabls+Results
  - One Run:
    Agent, seed, one_goal, #Hosts, #Honeypots, MT, won?, steps, unique_success, total_success
    name,  num,  bool,     num,    num,       num, -1/0/1, num, int[2],         int[2]
  - csv-file for ever set parameterchoice

Added MT-T6
- added multiprocessing
  - one pool writes to each file once -> prevent any possible deadlocks
- went through all possibilities

Added MT-T7
- average steps + dict of won -> split in files by agnet/seed/one_goal
- create Python Notebook to evaluate data
- aggressiv=red, standard=blue, carefull=green (probability plot)
- boxplot for steps
- One_Goal and Num_Hosts Bar Plots
- bar chart in procent
- 3d hp+mt+agents plot

---------------------------------------------------------------------------------------------------------

**Status**: Stable release. No extra development is planned, but still being maintained (bug fixes, etc).


Network Attack Simulator
========================

|docs|

Network Attack Simulator (NASim) is a simulated computer network complete with vulnerabilities, scans and exploits designed to be used as a testing environment for AI agents and planning techniques applied to network penetration testing.


Installation
------------

The easiest way to install the latest version of NASim hosted on PyPi is via pip::

  $ pip install nasim


To install dependencies for running the DQN test agent (this is needed to run the demo) run::

  $ pip install nasim[dqn]


To get the latest bleeding edge version and install in development mode see the `Install docs <https://networkattacksimulator.readthedocs.io/en/latest/tutorials/installation.html>`_


Demo
----

To see NASim in action, you can run the provided demo to interact with an environment directly or see a pre-trained AI agent in action.

To run the `tiny` benchmark scenario demo in interactive mode run::

  $ python -m nasim.demo tiny


This will then run an interactive console where the user can see the current state and choose the next action to take. The goal of the scenario is to *compromise* every host with a non-zero value.

See `here <https://networkattacksimulator.readthedocs.io/en/latest/reference/scenarios/benchmark_scenarios.html>`_ for the full list of scenarios.

To run the `tiny` benchmark scenario demo using the pre-trained AI agent, first ensure the DQN dependencies are installed (see *Installation* section above), then run::

  $ python -m nasim.demo tiny -ai


**Note:** Currently you can only run the AI demo for the `tiny` scenario.


Documentation
-------------

The documentation is available at: https://networkattacksimulator.readthedocs.io/



Using with OpenAI gym
---------------------

NASim implements the `Open AI Gym <https://github.com/openai/gym>`_ environment interface and so can be used with any algorithm that is developed for that interface.

See `Starting NASim using OpenAI gym <https://networkattacksimulator.readthedocs.io/en/latest/tutorials/gym_load.html>`_.


Authors
-------

**Jonathon Schwartz** - Jonathon.schwartz@anu.edu.au


License
-------

`MIT`_ © 2020, Jonathon Schwartz

.. _MIT: LICENSE


What's new
----------

- 2021-3-15 (v 0.8.0) (MINOR release)

  + Added option of specifying a 'value' for each host when defining a custom network using the .YAML format (thanks @Joe-zsc for the suggestion).
  + Added the 'small-honeypot' scenario to included scenarios.

- 2020-12-24 (v 0.7.5) (MICRO release)

  + Added 'undefined error' to observation to fix issue with initial and later observations being indistinguishable.

- 2020-12-17 (v 0.7.4) (MICRO release)

  + Fixed issues with incorrect observation of host 'value' and 'discovery_value'. Now, when in partially observable mode, the agent will correctly only observe these values on the step that they are recieved.
  + Some other minor code formatting fixes

- 2020-09-23 (v 0.7.3) (MICRO release)

  + Fixed issue with scenario YAML files not being included with PyPi package
  + Added final policy visualisation option to DQN and Q-Learning agents

- 2020-09-20 (v 0.7.2) (MICRO release)

  + Fixed bug with 're-registering' Gym environments when reloading modules
  + Added example implementations of Tabular Q-Learning: `agents/ql_agent.py` and `agents/ql_replay.py`
  + Added `Agents` section to docs, along with other minor doc updates

- 2020-09-20 (v 0.7.1) (MICRO release)

  + Added some scripts for running random benchmarks and describing benchmark scenarios
  + Added some more docs (including for creating custom scenarios) and updated other docs

- 2020-09-20 (v 0.7.0) (MINOR release)

  + Implemented host based firewalls
  + Added priviledge escalation
  + Added a demo script, including a pre-trained agent for the 'tiny' scenario
  + Fix to upper bound calculation (factored in reward for discovering a host)

- 2020-08-02 (v 0.6.0) (MINOR release)

  + Implemented compatibility with gym.make()
  + Updated docs for loading and interactive with NASimEnv
  + Added extra functions to nasim.scenarios to make it easier to load scenarios seperately to a NASimEnv
  + Fixed bug to do with class attributes and creating different scenarios in same python session
  + Fixed up bruteforce agent and tests

- 2020-07-31 (v 0.5.0) (MINOR release)

  + First official release on PyPi
  + Cleaned up dependencies, setup.py, etc and some small fixes


.. |docs| image:: https://readthedocs.org/projects/networkattacksimulator/badge/?version=latest
    :target: https://networkattacksimulator.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
    :scale: 100%
