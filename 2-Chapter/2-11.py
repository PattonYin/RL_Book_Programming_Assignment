import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tqdm import tqdm


class NonStationaryTestbed:
    def __init__(self, num_arms=10, true_value_walks=None):
        self.num_arms = num_arms
        self.true_value_means = [0.0] * num_arms
        self.true_value_stds = [0.0] * num_arms
        self.true_value_walks = true_value_walks
        self.step = 0

    def action(self, action):
        reward = self.true_value_means[action]
        if self.true_value_walks is not None:
            walk = self.true_value_walks[self.step]
        else:
            walk = np.random.normal(0, 0.01, size=self.num_arms)

        self.true_value_means = [m + w for m, w in zip(self.true_value_means, walk)]
        self.step += 1
        return reward, self.true_value_means

    def reset(self):
        self.true_value_means = [0.0] * self.num_arms
        self.step = 0


class SampleAverageStepSizeMethod_NonStationaryExperiment:
    def __init__(self, testbed1: NonStationaryTestbed, testbed2: NonStationaryTestbed):
        self.testbed1 = testbed1
        self.testbed2 = testbed2

        self.hyper_param_values = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1]
        self.runs_sample_average = {}
        for hyper_param in self.hyper_param_values:
            self.runs_sample_average[hyper_param] = []
        self.reward_histories_sample_average = {h: [] for h in self.hyper_param_values}

        self.runs_step_size = {}
        for hyper_param in self.hyper_param_values:
            self.runs_step_size[hyper_param] = []
        self.reward_histories_step_size = {h: [] for h in self.hyper_param_values}

        self.n_actions = testbed1.num_arms
        self.counts_sample_average = [0] * self.n_actions
        self.q_values_sample_average = [0] * testbed1.num_arms
        self.reward_history_sample_average = []
        self.true_value_history_sample_average = []
        self.total_reward_sample_average = 0
        self.action_history_sample_average = []

        self.step_size = 0.1
        self.q_values_step_size = [0] * testbed1.num_arms
        self.reward_history_step_size = []
        self.true_value_history_step_size = []
        self.total_reward_step_size = 0
        self.action_history_step_size = []


    def get_action(self, epsilon=0.1):
        if random.random() < epsilon:
            action = random.randint(0, self.n_actions - 1)
            return action, action 
        else:
            return np.argmax(self.q_values_sample_average), np.argmax(self.q_values_step_size)
        
    def get_reward_and_true_value_1(self, action):
        return self.testbed1.action(action)
    
    def get_reward_and_true_value_2(self, action):
        return self.testbed2.action(action)
    
    def update_q_value_sample_average(self, action, reward):
        self.counts_sample_average[action] += 1
        n = self.counts_sample_average[action]
        error = reward - self.q_values_sample_average[action]
        # true sample‐average step‐size
        alpha = 1.0 / n
        self.q_values_sample_average[action] += alpha * error

    def update_q_value_step_size(self, action, reward):
        error = reward - self.q_values_step_size[action]
        self.q_values_step_size[action] += error * self.step_size


    def run(self, num_steps=200_000, epsilon=0.1, num_runs=1):
        for hyper_param in self.hyper_param_values:
            rewards_sa_all_runs = []
            rewards_ss_all_runs = []

            epsilon_step_size = 0.1
            epsilon_sample_average = hyper_param
            step_size = hyper_param

            for _ in range(num_runs):
                # Reset per-run state
                self.counts_sample_average = [0] * self.n_actions
                q_sa = [0] * self.n_actions
                q_ss = [0] * self.n_actions
                rewards_sa = []
                rewards_ss = []

                self.testbed1.reset()  # make sure your testbed supports reset
                self.testbed2.reset()  # make sure your testbed supports reset

                for _ in range(num_steps):
                    if random.random() < epsilon_sample_average:
                        a_sa = random.randint(0, self.n_actions - 1)
                    else:
                        a_sa = np.argmax(q_sa)

                    if random.random() < epsilon_step_size:
                        a_ss = random.randint(0, self.n_actions - 1)
                    else:
                        a_ss = np.argmax(q_ss)

                    r_sa, _ = self.get_reward_and_true_value_1(a_sa)
                    r_ss, _ = self.get_reward_and_true_value_2(a_ss)

                    self.counts_sample_average[a_sa] += 1
                    alpha_sa = 1.0 / self.counts_sample_average[a_sa]
                    q_sa[a_sa] += alpha_sa * (r_sa - q_sa[a_sa])

                    q_ss[a_ss] += step_size * (r_ss - q_ss[a_ss])

                    rewards_sa.append(r_sa)
                    rewards_ss.append(r_ss)

                rewards_sa_all_runs.append(rewards_sa)
                rewards_ss_all_runs.append(rewards_ss)

            # Average across runs
            mean_sa = np.mean(rewards_sa_all_runs, axis=0)
            mean_ss = np.mean(rewards_ss_all_runs, axis=0)

            self.reward_histories_sample_average[hyper_param] = mean_sa
            self.reward_histories_step_size[hyper_param] = mean_ss

    def visualize_experiment(self):
        plt.figure(figsize=(12, 6))
        for h in self.hyper_param_values:
            plt.plot(self.reward_histories_sample_average[h], label=f"Sample Avg (α={1/int(1/h)})")
            plt.plot(self.reward_histories_step_size[h], linestyle='--', label=f"Const Step (α={h})")
        plt.xlabel("Steps")
        plt.ylabel("Average Reward")
        plt.title("Nonstationary Bandit: Sample Average vs Constant Step Size")
        # plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_average_reward_vs_hyperparam(self):
        avg_rewards_sa = []
        avg_rewards_ss = []

        for h in self.hyper_param_values:
            avg_rewards_sa.append(np.mean(self.reward_histories_sample_average[h]))
            avg_rewards_ss.append(np.mean(self.reward_histories_step_size[h]))

        x = range(len(self.hyper_param_values))  # uniform spacing

        plt.figure(figsize=(10, 5))
        plt.plot(x, avg_rewards_sa, marker='o', label='Sample Average')
        plt.plot(x, avg_rewards_ss, marker='s', label='Constant Step Size')
        plt.xticks(x, [f"{h:.5f}" for h in self.hyper_param_values], rotation=45)
        plt.xlabel("Hyperparameter (step size α)")
        plt.ylabel("Average Reward")
        plt.title("Average Reward vs Step Size")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    num_steps = 200_000
    num_arms = 10

    shared_walk = np.random.normal(0, 0.01, size=(num_steps + 1, num_arms))
    walk1 = shared_walk.copy()
    walk2 = shared_walk.copy()

    testbed1 = NonStationaryTestbed(num_arms=num_arms, true_value_walks=walk1)
    testbed2 = NonStationaryTestbed(num_arms=num_arms, true_value_walks=walk2)

    experiment = SampleAverageStepSizeMethod_NonStationaryExperiment(testbed1, testbed2)
    experiment.run(num_steps=num_steps, num_runs=20)
    experiment.visualize_average_reward_vs_hyperparam()

