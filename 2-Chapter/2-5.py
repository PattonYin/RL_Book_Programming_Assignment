''' 
Exercise 2.5
Design and conduct an experiment to demonstrate the difficulties that sample-average methods have for nonstationary problems. Use a modified version of the 10-armed testbed in which all the q*(a) start out equal and then take independent random walks (say by adding a normally distributed increment with mean 0 and standard deviation 0.01 to all the q*(a) on each step). Prepare plots like Figure 2.2 for an action-value method using sample averages, incrementally computed, and another action-value method using a constant step-size parameter, alpha = 0.1. Use epsilon = 0.1 and longer runs, say of 10,000 steps.
'''

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tqdm import tqdm


### Step 1: Implement the StationaryTestbed as the basis
class StationaryTestbed:
    def __init__(self, num_arms=10):
        self.num_arms = num_arms
        self.true_value_means = [-0.16595599, 0.44064899, -0.99977125, 0.39533485, -0.70648822, -0.81532281, -0.62747958, -0.30887855, -0.20646505, 0.07763347]
        self.true_value_stds = [0.4] * num_arms

    def action(self, action):
        return np.random.normal(self.true_value_means[action], self.true_value_stds[action])

    def print_true_values(self):
        print(f"True values: {self.true_value_means}")
        print(f"True values std: {self.true_value_stds}")
    
    def visualize_true_values_distribution(self, num_samples=1000):
        # Sample rewards for each arm
        data = [
            np.random.normal(mu, sigma, size=num_samples)
            for mu, sigma in zip(self.true_value_means, self.true_value_stds)
        ]

        # Plot
        fig, ax = plt.subplots()
        parts = ax.violinplot(data, showmeans=True)
        ax.set_xticks(range(1, self.num_arms + 1))
        ax.set_xlabel('Arm')
        ax.set_ylabel('Reward')
        ax.set_title('True Value Distribution per Arm')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()

class SampleAverageMethod_Stationary:
    def __init__(self, testbed: StationaryTestbed):
        self.testbed = testbed
        self.n_actions = testbed.num_arms
        self.counts_sample_average = [0] * self.n_actions
        self.q_values = [0] * testbed.num_arms
        self.reward_history = []
        self.total_reward = 0
        self.action_history = []

    def get_estimated_q_values(self):
        return self.q_values
    
    def update_q_value(self, action, reward):
        self.counts_sample_average[action] += 1
        self.q_values[action] = self.q_values[action] + (reward - self.q_values[action]) / self.counts_sample_average[action]

    def get_action(self, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_values)
        
    def get_reward(self, action):
        return self.testbed.action(action)
    
    def run(self, num_steps=10000, epsilon=0.1):
        for _ in tqdm(range(num_steps)):
            action = self.get_action(epsilon)
            reward = self.get_reward(action)
            self.update_q_value(action, reward)
            self.total_reward += reward
            self.reward_history.append(reward)
            self.action_history.append(action)
        return self.total_reward
    
    def visualize_total_reward(self, smooth_window=5):
        rewards = np.array(self.reward_history)
        
        # Raw rewards
        plt.plot(rewards, alpha=0.3, label='Raw reward')

        # Moving‐average smoothing
        if len(rewards) >= smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            smooth = np.convolve(rewards, kernel, mode='valid')
            # align the x‐axis so the smooth line is centered on each window
            x_smooth = np.arange(smooth_window//2, smooth_window//2 + len(smooth))
            plt.plot(x_smooth, smooth, linewidth=2, label=f'{smooth_window}-step MA')
        
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title('Reward per Step with Smoothing')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def visualize_action_history(self):
        # Count how many times each action was taken
        counts = np.bincount(self.action_history, minlength=self.n_actions)

        # Bar chart
        plt.bar(range(self.n_actions), counts)
        plt.xlabel('Arm')
        plt.ylabel('Selection Count')
        plt.title('Action Selection Frequency')
        plt.xticks(range(self.n_actions))
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()


class NonStationaryTestbed:
    def __init__(self, num_arms=10):
        self.num_arms = num_arms
        self.true_value_means = [0] * num_arms
        self.true_value_stds = [0] * num_arms

    def action(self, action):
        reward = self.true_value_means[action]
        random_walks = np.random.normal(0, 0.01, size=self.num_arms)
        self.true_value_means = [mean + walk for mean, walk in zip(self.true_value_means, random_walks)]
        return reward, self.true_value_means

class SampleAverageMethod_NonStationary:
    def __init__(self, testbed: NonStationaryTestbed):
        self.testbed = testbed
        self.n_actions = testbed.num_arms
        self.counts_sample_average = [0] * self.n_actions
        self.q_values = [0] * testbed.num_arms
        self.reward_history = []
        self.true_value_history = []
        self.total_reward = 0
        self.action_history = []

    def get_action(self, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_values)
        
    def get_reward_and_true_value(self, action):
        return self.testbed.action(action)
    
    def update_q_value(self, action, reward):
        self.counts_sample_average[action] += 1
        self.q_values[action] = self.q_values[action] + (reward - self.q_values[action]) / self.counts_sample_average[action]

    def run(self, num_steps=1000, epsilon=0.1):
        for _ in tqdm(range(num_steps)):
            action = self.get_action(epsilon)
            reward, true_values = self.get_reward_and_true_value(action)
            self.update_q_value(action, reward)
            self.total_reward += reward
            self.reward_history.append(reward)
            self.action_history.append(action)
            self.true_value_history.append(list(true_values))

    def visualize_experiment(self, smooth_window=10):
        steps = len(self.reward_history)
        rewards = np.array(self.reward_history)

        # Prepare smoothed reward curve
        if steps >= smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            smooth = np.convolve(rewards, kernel, mode='valid')
            x_smooth = np.arange(smooth_window//2, smooth_window//2 + len(smooth))
        else:
            smooth = None

        # Count action selections
        counts = np.bincount(self.action_history, minlength=self.n_actions)

        # Gather true‐value history
        tv = np.array(self.true_value_history)  # shape (steps, num_arms)

        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        # 1) Reward + smoothing
        axes[0].plot(rewards, alpha=0.3, label='Raw reward')
        if smooth is not None:
            axes[0].plot(x_smooth, smooth, lw=2, label=f'{smooth_window}-step MA')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Reward per Step')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # 2) Action frequency
        axes[1].bar(np.arange(self.n_actions), counts)
        axes[1].set_xlabel('Arm')
        axes[1].set_ylabel('Selection Count')
        axes[1].set_title('Action Selection Frequency')
        axes[1].grid(axis='y', linestyle='--', alpha=0.5)

        # 3) True‐value trajectories + dot markers for selections
        # First plot each arm’s drifting value and grab its line color
        line_objs = []
        for arm in range(self.n_actions):
            (line,) = axes[2].plot(tv[:, arm], alpha=0.6)
            line_objs.append(line)

        # Now scatter a small dot at each timestep where that arm was chosen
        for t, arm in enumerate(self.action_history):
            y = tv[t, arm]
            axes[2].scatter(
                t, y,
                s=10,              # size of the dot
                color=line_objs[arm].get_color(),
                marker='o',
                alpha=0.7
            )

        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('True Value')
        axes[2].set_title('True‐Value Means Over Time  (dots mark chosen arm)')
        axes[2].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

class SampleAverageStepSizeMethod_NonStationary:
    def __init__(self, testbed: NonStationaryTestbed):
        self.testbed = testbed

        self.n_actions = testbed.num_arms
        self.counts_sample_average = [0] * self.n_actions
        self.q_values_sample_average = [0] * testbed.num_arms
        self.reward_history_sample_average = []
        self.true_value_history_sample_average = []
        self.total_reward_sample_average = 0
        self.action_history_sample_average = []

        self.step_size = 0.1
        self.q_values_step_size = [0] * testbed.num_arms
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
        
    def get_reward_and_true_value(self, action):
        return self.testbed.action(action)
    
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


    def run(self, num_steps=10_000, epsilon=0.1):
        for _ in tqdm(range(num_steps)):
            action_sample_average, action_step_size = self.get_action(epsilon)
            
            reward_sample_average, true_values = self.get_reward_and_true_value(action_sample_average)
            self.update_q_value_sample_average(action_sample_average, reward_sample_average)
            self.total_reward_sample_average += reward_sample_average  
            self.reward_history_sample_average.append(reward_sample_average)
            self.action_history_sample_average.append(action_sample_average)
            self.true_value_history_sample_average.append(list(true_values))

            reward_step_size, true_values = self.get_reward_and_true_value(action_step_size)
            self.update_q_value_step_size(action_step_size, reward_step_size)
            self.total_reward_step_size += reward_step_size
            self.reward_history_step_size.append(reward_step_size)
            self.action_history_step_size.append(action_step_size)
            self.true_value_history_step_size.append(list(true_values))

    def visualize_experiment(self, smooth_window=100, marker_offset=0.03):
        steps = len(self.reward_history_sample_average)

        # 1) rewards
        r_sa = np.array(self.reward_history_sample_average)
        r_ss = np.array(self.reward_history_step_size)

        # moving‐average smoothing
        if steps >= smooth_window:
            k = np.ones(smooth_window) / smooth_window
            smooth_sa = np.convolve(r_sa, k, mode='valid')
            smooth_ss = np.convolve(r_ss, k, mode='valid')
            x_sm = np.arange(smooth_window//2, smooth_window//2 + len(smooth_sa))
        else:
            smooth_sa = smooth_ss = None

        # 2) action‐count
        counts_sa = np.bincount(self.action_history_sample_average, minlength=self.n_actions)
        counts_ss = np.bincount(self.action_history_step_size,       minlength=self.n_actions)
        x = np.arange(self.n_actions)
        width = 0.4

        # 3) true values
        fig, axes = plt.subplots(3, 1, figsize=(10, 14))

        # — plot 1: rewards
        axes[0].plot(r_sa, alpha=0.2, label='Raw SA')
        axes[0].plot(r_ss, alpha=0.2, label='Raw SS')
        if smooth_sa is not None:
            axes[0].plot(x_sm, smooth_sa, lw=2, label=f'SA MA({smooth_window})')
            axes[0].plot(x_sm, smooth_ss, lw=2, ls='--', label=f'SS MA({smooth_window})')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Reward per Step')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.5)
        print("plot 1 completed")

        # — plot 2: action frequencies
        axes[1].bar(x - width/2, counts_sa, width, label='SA')
        axes[1].bar(x + width/2, counts_ss, width, label='SS')
        axes[1].set_xlabel('Arm')
        axes[1].set_ylabel('Selection Count')
        axes[1].set_title('Action Selection Frequency')
        axes[1].legend()
        axes[1].grid(axis='y', linestyle='--', alpha=0.5)
        print("plot 2 completed")

        # — plot 3: true‐value evolution + selection dots
        tv_sa = np.array(self.true_value_history_sample_average)
        tv_ss = np.array(self.true_value_history_step_size)

        for arm in range(self.n_actions):
            color = f"C{arm}"
            # plot lines
            axes[2].plot(tv_sa[:, arm], color=color, alpha=0.6)
            axes[2].plot(tv_ss[:, arm], color=color, ls='--', alpha=0.6)

        # now scatter with a tiny vertical shift so dots don’t sit exactly on the line
        for t, arm in enumerate(self.action_history_sample_average):
            y = tv_sa[t, arm] + marker_offset
            axes[2].scatter(t, y,
                            s=10,
                            color="red",
                            marker='o',
                            alpha=0.7)
        for t, arm in enumerate(self.action_history_step_size):
            y = tv_ss[t, arm] - marker_offset
            axes[2].scatter(t, y,
                            s=10,
                            color="blue",
                            marker='x',
                            alpha=0.7)

        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('True Value')
        axes[2].set_title('True‐Value Means Over Time\n(o = SA picks, x = SS picks)')
        axes[2].grid(True, linestyle='--', alpha=0.5)

        # custom legend for the selection markers
        custom = [
            Line2D([0], [0], color='k', marker='o', lw=0, label='Picked by SA'),
            Line2D([0], [0], color='k', marker='x', lw=0, label='Picked by SS')
        ]
        axes[2].legend(handles=custom, loc='upper left')
        print("plot 3 completed")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # testbed = StationaryTestbed()
    # testbed.print_true_values()
    # print(testbed.action(0))
    # testbed.visualize_true_values_distribution()
    # sample_average_method = SampleAverageMethod_Stationary(testbed)
    # sample_average_method.run(num_steps=1000, epsilon=0.15)
    # sample_average_method.visualize_total_reward()
    # sample_average_method.visualize_action_history()

    # testbed = NonStationaryTestbed()
    # sample_average_method = SampleAverageMethod_NonStationary(testbed)
    # sample_average_method.run(num_steps=1000, epsilon=0.15)
    # sample_average_method.visualize_experiment()

    testbed = NonStationaryTestbed()
    sample_average_step_size_method = SampleAverageStepSizeMethod_NonStationary(testbed)
    sample_average_step_size_method.run(num_steps=1_000, epsilon=0.1)
    sample_average_step_size_method.visualize_experiment()