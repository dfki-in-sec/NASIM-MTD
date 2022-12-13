"""
$ python penbox_dqn_agent.py
$ tensorboard --logdir runs/

"""


import random
import os
import numpy as np
from gym import error
from pprint import pprint
from multiprocessing import Pool

import nasim

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    raise error.DependencyNotInstalled(
        f"{e}. (HINT: you can install dqn_agent dependencies by running "
        "'pip install nasim[dqn]'.)"
    )


class ReplayMemory:

    def __init__(self, capacity, s_dims, device="cuda"):  # CBP was cpu
        self.capacity = capacity
        self.device = device
        self.s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.a_buf = np.zeros((capacity, 1), dtype=np.int64)
        self.next_s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.r_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def store(self, s, a, next_s, r, done):
        self.s_buf[self.ptr] = s
        self.a_buf[self.ptr] = a
        self.next_s_buf[self.ptr] = next_s
        self.r_buf[self.ptr] = r
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size):
        sample_idxs = np.random.choice(self.size, batch_size)
        batch = [self.s_buf[sample_idxs],
                 self.a_buf[sample_idxs],
                 self.next_s_buf[sample_idxs],
                 self.r_buf[sample_idxs],
                 self.done_buf[sample_idxs]]
        return [torch.from_numpy(buf).to(self.device) for buf in batch]


class DQN(nn.Module):
    """A simple Deep Q-Network """

    def __init__(self, input_dim, layers, num_actions):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim[0], layers[0])])
        for l in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[l - 1], layers[l]))
        self.out = nn.Linear(layers[-1], num_actions)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.out(x)
        return x

    def save_DQN(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_DQN(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def get_action(self, x):
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.view(1, -1)
            return self.forward(x).max(1)[1]


class DQNAgent:
    """A simple Deep Q-Network Agent """

    def __init__(self,
                 env_list,
                 seed=None,
                 lr=0.001,
                 training_steps=20000,
                 batch_size=32,
                 replay_size=10000,
                 final_epsilon=0.05,
                 exploration_steps=10000,
                 gamma=0.99,
                 hidden_sizes=[64, 64],
                 target_update_freq=1000,
                 verbose=True,
                 **kwargs):

        # This DQN implementation only works for flat actions
        for env in env_list:
            assert env.flat_actions
        self.verbose = verbose
        if self.verbose:
            print(f"\nRunning DQN with config:")
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
        self.logger = SummaryWriter(comment=str(lr) + "_" + str(batch_size) + "_" + str(replay_size) + "_" + str(final_epsilon) + "_" + str(exploration_steps) + "_" + str(gamma) + "_" + str(hidden_sizes) + "_" + str(training_steps))

        # Training related attributes
        self.lr = lr
        self.exploration_steps = exploration_steps
        self.final_epsilon = final_epsilon
        self.epsilon_schedule = np.linspace(1.0,
                                            self.final_epsilon,
                                            self.exploration_steps)
        self.batch_size = batch_size
        self.discount = gamma
        self.training_steps = training_steps
        self.steps_done = 0

        # Neural Network related attributes
        self.device = torch.device("cuda"
                                   if torch.cuda.is_available()
                                   else "cpu")
        self.dqn = DQN(self.obs_dim,
                       hidden_sizes,
                       self.num_actions).to(self.device)
        if self.verbose:
            print(f"\nUsing Neural Network running on device={self.device}:")
            print(self.dqn)

        self.target_dqn = DQN(self.obs_dim,
                              hidden_sizes,
                              self.num_actions).to(self.device)
        self.target_update_freq = target_update_freq

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

        # replay setup
        self.replay = ReplayMemory(replay_size,
                                   self.obs_dim,
                                   self.device)

    def save(self, save_path):
        self.dqn.save_DQN(save_path)

    def load(self, load_path):
        self.dqn.load_DQN(load_path)

    def get_epsilon(self):
        if self.steps_done < self.exploration_steps:
            return self.epsilon_schedule[self.steps_done]
        return self.final_epsilon

    def get_egreedy_action(self, o, epsilon):
        if random.random() > epsilon:
            o = torch.from_numpy(o).float().to(self.device)
            return self.dqn.get_action(o).cpu().item()
        return random.randint(0, self.num_actions - 1)

    def optimize(self):
        batch = self.replay.sample_batch(self.batch_size)
        s_batch, a_batch, next_s_batch, r_batch, d_batch = batch

        # get q_vals for each state and the action performed in that state
        q_vals_raw = self.dqn(s_batch)
        q_vals = q_vals_raw.gather(1, a_batch).squeeze()

        # get target q val = max val of next state
        with torch.no_grad():
            target_q_val_raw = self.target_dqn(next_s_batch)
            target_q_val = target_q_val_raw.max(1)[0]
            target = r_batch + self.discount * (1 - d_batch) * target_q_val

        # calculate loss
        loss = self.loss_fn(q_vals, target)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        q_vals_max = q_vals_raw.max(1)[0]
        mean_v = q_vals_max.mean().item()
        return loss.item(), mean_v

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
        self.env = self.env_list[self.env_counter % len(self.env_list)]
        self.env.reset()
        o = self.env.reset()

        self.env_counter += 1
        done = False

        steps = 0
        episode_return = 0

        while not done and steps < step_limit:
            a = self.get_egreedy_action(o, self.get_epsilon())

            next_o, r, done, _ = self.env.step(a)
            self.replay.store(o, a, next_o, r, done)
            self.steps_done += 1
            loss, mean_v = self.optimize()
            self.logger.add_scalar("loss", loss, self.steps_done)
            self.logger.add_scalar("mean_v", mean_v, self.steps_done)

            o = next_o
            episode_return += r
            self.logger.add_scalar("reward", episode_return, self.steps_done)
            steps += 1

        return episode_return, steps, self.env.goal_reached()

    def run_eval_episode(self,
                         env=None,
                         render=True,  # CBP was false
                         eval_epsilon=0.05,
                         render_mode="readable"):
        if env is None:
            env = self.env
        o = env.reset()
        done = False

        steps = 0
        episode_return = 0
        print(render)
        line_break = "=" * 60
        if render:
            print("\n" + line_break)
            print(f"Running EVALUATION using epsilon = {eval_epsilon:.4f}")
            print(line_break)
            env.render(render_mode)
            input("Initial state. Press enter to continue..")

        while not done:
            a = self.get_egreedy_action(o, eval_epsilon)
            next_o, r, done, _ = env.step(a)
            o = next_o
            episode_return += r
            steps += 1
            if render:
                print("\n" + line_break)
                print(f"Step {steps}")
                print(line_break)
                print(f"Action Performed = {env.action_space.get_action(a)}")
                # env.render(render_mode)
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


def run(list):
    ### change paramter
    envs = ["s3_r_a_0.yaml", "s3_r_a_1.yaml", "s3_r_a_2.yaml", "s3_r_a_3.yaml", "s3_r_a_4.yaml", "s3_r_a_5.yaml",
            "s3_r_a_6.yaml", "s3_r_a_7.yaml", "s3_r_a_8.yaml", "s3_r_a_9.yaml", "s3_r_a_10.yaml", "s3_r_a_11.yaml",
            "s3_r_a_12.yaml", "s3_r_a_13.yaml", "s3_r_a_14.yaml", "s3_r_a_15.yaml", "s3_r_a_16.yaml", "s3_r_a_17.yaml",
            "s3_r_a_18.yaml", "s3_r_a_19.yaml", "s3_r_a_20.yaml", "s3_r_a_21.yaml", "s3_r_a_22.yaml", "s3_r_a_23.yaml"]
    # Load and set Envirnoments
    env_list = []
    for env in envs:
        print(env)
        env_list.append(nasim.load("../scenarios/benchmark/s3_rotation/" + env, flat_actions=True, flat_obs=True))

    dqn_agent = DQNAgent( env_list, seed=None,
                 lr=list[0], #default
                 training_steps=2000000, #default 20000 -> ideal 200000
                 batch_size=list[1],
                 replay_size=list[2],
                 final_epsilon=list[3],
                 exploration_steps=list[4],
                 gamma=list[5],
                 hidden_sizes=list[6],
                 target_update_freq=list[7],
                 verbose=False,
    )

    dqn_agent.train()
    print("Finished Training with")
    dqn_agent.save("runs/" +  "_" + str(list[0]) +  "_" + str(list[1]) +  "_" + str(list[2]) +  "_" + str(list[3]) +  "_" + str(list[4]) +  "_" + str(list[5]) +  "_" + str(list[6]) +  "_" + str(list[7]))



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
    counter = 0
    for file in os.listdir(directory):
            if file[0] == "_":
                counter += 1
                print(counter)
                list = file.split("_")
                dqn_agent = DQNAgent(env_list, seed=None,
                                     lr=float(list[1]),  # default
                                     training_steps=20000,  # default 20000 -> ideal 200000
                                     batch_size=int(list[2]),
                                     replay_size=int(list[3]),
                                     final_epsilon=float(list[4]),
                                     exploration_steps=int(list[5]),
                                     gamma=float(list[6]),
                                     hidden_sizes=[int(list[7].split(",")[0][1:]), int(list[7].split(",")[1][1:-1])],
                                     target_update_freq=int(list[8]),
                                     verbose=False,
                                     )
                dqn_agent.load("data/"+str(file))
                reward = 0
                list_result = []
                for env in env_list:
                    s = env.reset()
                    #print("-------------------------["+env.name+"]------------------------------------\n")
                    done = False
                    s_reward = 0
                    a_counter = 0
                    while not done:
                        a = dqn_agent.get_egreedy_action(s,0)
                        a_counter += 1
                        #print(str(a) + f" Action Performed = {env.action_space.get_action(a)}")
                        next_s, r, done, _ = env.step(a)
                        s = next_s
                        reward += r
                        s_reward += r
                    #print("------------------------["+str(s_reward)+"]-------------------------------------\n")
                    #print("------------------------[" + str(reward) + "]-------------------------------------\n")
                    list_result.append([s_reward, a_counter])
                result.append([str(file), reward, list_result])
    with open("re_DQN3_long.txt", "a") as f:
        print(result, file=f)