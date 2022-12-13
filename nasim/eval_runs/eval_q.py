#This File should Test multiple Paramater for each Enviromnent to look for the Best Parameter
#with Switching Envirnoments
"""An example Tabular, epsilon greedy Q-Learning Agent.

This agent does not use an Experience replay (see the 'ql_replay_agent.py')

It uses pytorch 1.5+ tensorboard library for logging (HINT: these dependencies
can be installed by running pip install nasim[dqn])

To run 'tiny' benchmark scenario with default settings, run the following from
the nasim/agents dir:

$ python ql_agent.py tiny

To see detailed results using tensorboard:

$ tensorboard --logdir runs/

To see available hyperparameters:

$ python ql_agent.py --help

Notes
-----

This is by no means a state of the art implementation of Tabular Q-Learning.
It is designed to be an example implementation that can be used as a reference
for building your own agents and for simple experimental comparisons.
"""
import os
import random
import numpy as np
from pprint import pprint
from multiprocessing import Pool

import nasim

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    from gym import error
    raise error.DependencyNotInstalled(
        f"{e}. (HINT: you can install tabular_q_learning_agent dependencies "
        "by running 'pip install nasim[dqn]'.)"
    )

class TabularQFunction:
    """Tabular Q-Function """

    def __init__(self, num_actions):
        self.q_func = dict()
        self.num_actions = num_actions

    def __call__(self, x):
        return self.forward(x)

    def set_q_func(self,dict):
        self.q_func = dict

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = str(x.astype(np.int))
        if x not in self.q_func:
            self.q_func[x] = np.zeros(self.num_actions, dtype=np.float32)
        return self.q_func[x]

    def forward_batch(self, x_batch):
        return np.asarray([self.forward(x) for x in x_batch])

    def update_batch(self, s_batch, a_batch, delta_batch):
        for s, a, delta in zip(s_batch, a_batch, delta_batch):
            q_vals = self.forward(s)
            q_vals[a] += delta

    def update(self, s, a, delta):
        q_vals = self.forward(s)
        q_vals[a] += delta

    def get_action(self, x):
        return int(self.forward(x).argmax())

    def display(self):
        pprint(self.q_func)


class TabularQLearningAgent:
    """A Tabular. epsilon greedy Q-Learning Agent using Experience Replay """

    def __init__(self,
                 env_list,
                 seed=None,
                 lr=0.001,
                 training_steps=10000,
                 final_epsilon=0.05,
                 exploration_steps=10000,
                 gamma=0.99,
                 verbose=True,
                 **kwargs):

        # This implementation only works for flat actions
        for env in env_list:
            assert env.flat_actions

        self.verbose = verbose
        if self.verbose:
            print("\nRunning Tabular Q-Learning with config:")
            pprint(locals())

        # set seeds
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # envirnment setup
        self.env_list = env_list
        #Set biggest enviroment
        self.env = env_list[0]
        # For Changing Envirnoment
        self.env_counter = 0


        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape
        # logger setup
        """ used for Parameter 
        self.logger = SummaryWriter(
            comment="" + str(lr) + "_" + str(final_epsilon) + "_" + str(exploration_steps) + "_" + str(gamma))
        """
        self.logger = SummaryWriter(
            comment= str(len(env_list)))
        # Training related attributes
        self.lr = lr
        self.exploration_steps = exploration_steps
        self.final_epsilon = final_epsilon
        self.epsilon_schedule = np.linspace(
            1.0, self.final_epsilon, self.exploration_steps
        )
        self.discount = gamma
        self.training_steps = training_steps
        self.steps_done = 0

        # Q-Function
        self.qfunc = TabularQFunction(self.num_actions)


    def save(self):
        np.save("runs/" + str(self.lr) + "_" + str(self.final_epsilon) + "_" + str(self.exploration_steps) + "_" + str(self.discount) + "",
                self.qfunc.q_func)

    def load(self,name):
        self.qfunc.set_q_func(np.load(name, allow_pickle=True).item())
        self.steps_done = self.training_steps


    def get_epsilon(self):
        if self.steps_done < self.exploration_steps:
            return self.epsilon_schedule[self.steps_done]
        return self.final_epsilon

    def get_egreedy_action(self, o, epsilon):
        if random.random() > epsilon:
            return self.qfunc.get_action(o)
        return random.randint(0, self.num_actions-1)

    def optimize(self, s, a, next_s, r, done):
        # get q_val for state and action performed in that state
        q_vals_raw = self.qfunc.forward(s)
        q_val = q_vals_raw[a]

        # get target q val = max val of next state
        target_q_val = self.qfunc.forward(next_s).max()
        target = r + self.discount * (1-done) * target_q_val

        # calculate error and update
        td_error = target - q_val
        td_delta = self.lr * td_error

        # optimize the model
        self.qfunc.update(s, a, td_delta)

        s_value = q_vals_raw.max()
        return td_error, s_value

    def train(self):
        if self.verbose:
            print("\nStarting training")

        num_episodes = 0
        training_steps_remaining = self.training_steps

        while self.steps_done < self.training_steps:
            ep_results = self.run_train_episode(training_steps_remaining)
            ep_return, ep_steps, goal = ep_results
            num_episodes += 1
            training_steps_remaining -= ep_steps

            self.logger.add_scalar("episode", num_episodes, self.steps_done)
            self.logger.add_scalar(
                "epsilon", self.get_epsilon(), self.steps_done
            )
            self.logger.add_scalar(
                "episode_return", ep_return, self.steps_done
            )
            self.logger.add_scalar(
                "episode_steps", ep_steps, self.steps_done
            )
            self.logger.add_scalar(
                "episode_goal_reached", int(goal), self.steps_done
            )

            if num_episodes % 10 == 0 and self.verbose:
                print(f"\nEpisode {num_episodes}:")
                print(f"\tsteps done = {self.steps_done} / "
                      f"{self.training_steps}")
                print(f"\treturn = {ep_return}")
                print(f"\tgoal = {goal}")

        self.logger.close()
        if self.verbose:
            print("Training complete")
            print(f"\nEpisode {num_episodes}:")
            print(f"\tsteps done = {self.steps_done} / {self.training_steps}")
            print(f"\treturn = {ep_return}")
            print(f"\tgoal = {goal}")

    def run_train_episode(self, step_limit):
        self.env = self.env_list[self.env_counter%len(self.env_list)]
        self.env.reset()
        s = self.env.reset()
        self.env_counter += 1

        done = False

        steps = 0
        episode_return = 0

        while not done and steps < step_limit:
            a = self.get_egreedy_action(s, self.get_epsilon())
            next_s, r, done, _ = self.env.step(a)
            self.steps_done += 1
            td_error, s_value = self.optimize(s, a, next_s, r, done)
            self.logger.add_scalar("td_error", td_error, self.steps_done)
            self.logger.add_scalar("s_value", s_value, self.steps_done)

            s = next_s
            episode_return += r
            steps += 1

        return episode_return, steps, self.env.goal_reached()

    def run_eval_episode(self,
                         env=None,
                         render=False,
                         eval_epsilon=0.05,
                         render_mode="readable"):
        if env is None:
            env = self.env
        env.reset()
        s = env.reset()
        done = False

        steps = 0
        episode_return = 0

        line_break = "="*60
        if render:
            print("\n" + line_break)
            print(f"Running EVALUATION using epsilon = {eval_epsilon:.4f}")
            print(line_break)
            env.render(render_mode)
            input("Initial state. Press enter to continue..")

        while not done:
            a = self.get_egreedy_action(s, eval_epsilon)
            next_s, r, done, _ = env.step(a)
            s = next_s
            episode_return += r
            steps += 1
            if render:
                print("\n" + line_break)
                print(f"Step {steps}")
                print(line_break)
                print(f"Action Performed = {env.action_space.get_action(a)}")
                env.render(render_mode)
                print(f"Reward = {r}")
                print(f"Done = {done}")
                input("Press enter to continue..")

                if done:
                    print("\n" + line_break)
                    print("EPISODE FINISHED")
                    print(line_break)
                    print(f"Goal reached = {env.goal_reached()}")
                    print(f"Total steps = {steps}")
                    print(f"Total reward = {episode_return}")

        return episode_return, steps, env.goal_reached()

if __name__ == "__main__":
    envs = ["s3_r_a_0.yaml", "s3_r_a_1.yaml", "s3_r_a_2.yaml", "s3_r_a_3.yaml", "s3_r_a_4.yaml", "s3_r_a_5.yaml",
            "s3_r_a_6.yaml", "s3_r_a_7.yaml", "s3_r_a_8.yaml", "s3_r_a_9.yaml", "s3_r_a_10.yaml", "s3_r_a_11.yaml",
            "s3_r_a_12.yaml", "s3_r_a_13.yaml", "s3_r_a_14.yaml", "s3_r_a_15.yaml", "s3_r_a_16.yaml", "s3_r_a_17.yaml",
            "s3_r_a_18.yaml", "s3_r_a_19.yaml", "s3_r_a_20.yaml", "s3_r_a_21.yaml", "s3_r_a_22.yaml", "s3_r_a_23.yaml"]

    env_list = []
    for env in envs:
        env_list.append(nasim.load("../scenarios/benchmark/s3_rotation/" + env, flat_actions=True, flat_obs=True))


    result = []
    directory="data"
    #for each agent
    for file in os.listdir(directory):
        if file.endswith('.npy'):
            filex =file.replace(".","_")
            para = filex.split("_",)
            ql_agent = TabularQLearningAgent(
                env_list=env_list,
                seed=None,
                lr=float(para[0]),
                training_steps=200000,  # default 10000  working well -> 200000
                final_epsilon=float(para[1]),  # default 0.05
                exploration_steps=int(para[2]),
                gamma=float(para[3]),
                verbose=False  # To get prints set True
            )
            ql_agent.load("data/"+str(file))
            reward = 0
            list_result = []
            for env in env_list:
                s = env.reset()
                #print("-------------------------"+env.name+"------------------------------------\n")
                done = False
                s_reward = 0
                a_counter = 0
                while not done:
                    a_counter += 1
                    a = ql_agent.get_egreedy_action(s,0)
                    #print(str(a) + f" Action Performed = {env.action_space.get_action(a)}")
                    next_s, r, done, _ = env.step(a)
                    s = next_s
                    reward += r
                    s_reward += r
                #print("------------------------"+str(s_reward)+"-------------------------------------\n")
                list_result.append([s_reward,a_counter])
            result.append([str(file),reward,list_result])
    with open("re_Q3_long.txt", "a") as f:
        print(result, file=f)