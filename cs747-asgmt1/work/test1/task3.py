# Task 3
# Using inspiration from code in task1.py and simulator.py write the appropriate functions to create the plot required.

import numpy as np
import matplotlib.pyplot as plt
from bernoulli_bandit import *
from task1 import Algorithm
from multiprocessing import Pool

# DEFINE your algorithm class here

# DEFINE single_sim_task3() HERE

# DEFINE simulate_task3() HERE

# DEFINE task3() HERE

# Call task3() to generate the plots

def epsilon_greedy(num_arms, horizon, epsilon):
    counts = np.zeros(num_arms)
    values = np.zeros(num_arms)
    for t in range(horizon):
        if np.random.rand() < epsilon:
            arm = np.random.randint(num_arms)
        else:
            arm = np.argmax(values)
        reward = np.random.binomial(1, [0.7, 0.6, 0.5, 0.4, 0.3][arm])
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
    return values

def plot_epsilon_greedy():
    epsilons = np.arange(0, 1.01, 0.01)
    regrets = []
    for eps in epsilons:
        regrets.append(np.mean([sum(epsilon_greedy(5, 30000, eps)) for _ in range(50)]))
    import matplotlib.pyplot as plt
    plt.plot(epsilons, regrets)
    plt.xlabel("Epsilon")
    plt.ylabel("Regret")
    plt.title("Epsilon-Greedy Analysis")
    plt.savefig("epsilon_greedy_plot.png")
    plt.show()

plot_epsilon_greedy()