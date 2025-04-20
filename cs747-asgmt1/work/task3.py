import numpy as np
import matplotlib.pyplot as plt
from bernoulli_bandit import BernoulliBandit
from multiprocessing import Pool
from task1 import Algorithm

class EpsilonGreedy(Algorithm):
    def __init__(self, num_arms, horizon, epsilon):
        super().__init__(num_arms, horizon)
        self.epsilon = epsilon
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.t = 0

    def give_pull(self):
        if self.t < self.num_arms:
            return self.t % self.num_arms
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)

    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        self.values[arm_index] += (reward - self.values[arm_index]) / self.counts[arm_index]
        self.t += 1

def single_sim_task3(seed, epsilon):
    np.random.seed(seed)
    probs = [0.7, 0.6, 0.5, 0.4, 0.3]
    bandit = BernoulliBandit(probs=probs)
    algo = EpsilonGreedy(5, 30000, epsilon)
    for _ in range(30000):
        arm = algo.give_pull()
        reward = bandit.pull(arm)
        algo.get_reward(arm, reward)
    return bandit.regret()

def simulate_task3(epsilon, num_sims=50):
    with Pool() as pool:
        regrets = pool.starmap(single_sim_task3, [(i, epsilon) for i in range(num_sims)])
    return np.mean(regrets)

def task3():
    epsilons = np.arange(0, 1.01, 0.01)
    regrets = []
    for eps in epsilons:
        avg_regret = simulate_task3(eps)
        regrets.append(avg_regret)
        print(f"Epsilon: {eps:.2f}, Average Regret: {avg_regret:.2f}")
    plt.plot(epsilons, regrets)
    plt.xlabel('Epsilon')
    plt.ylabel('Average Regret')
    plt.title('Effect of Epsilon on Regret in Epsilon-Greedy Algorithm')
    plt.savefig('task1-egreedy.png')
    plt.close()

if __name__ == '__main__':
    task3()