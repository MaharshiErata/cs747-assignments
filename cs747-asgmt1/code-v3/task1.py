"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

def kl_div(p, q):
    eps = 1e-15
    p = max(eps, min(1-eps, p))
    q = max(eps, min(1-eps, q))
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def compute_kl_ucb(values, counts, t, arm):
    if counts[arm] == 0:
        return 1.0
    p = values[arm]
    if p >= 1 - 1e-6:
        return 1.0
    low, high = p, 1.0
    threshold = math.log(t) / counts[arm]
    for _ in range(100):
        mid = (low + high) / 2
        if kl_div(p, mid) <= threshold:
            low = mid
        else:
            high = mid
    return low

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.t = 0
    
    def give_pull(self):
        if self.t < self.num_arms:
            return self.t
        ucb = self.values + np.sqrt(2 * np.log(self.t) / (self.counts + 1e-6))
        return np.argmax(ucb)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        self.values[arm_index] += (reward - self.values[arm_index]) / self.counts[arm_index]
        self.t += 1

class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.t = 1
    
    def give_pull(self):
        if self.t <= self.num_arms:
            return self.t - 1
        indices = [compute_kl_ucb(self.values, self.counts, self.t, a) for a in range(self.num_arms)]
        return np.argmax(indices)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        self.values[arm_index] += (reward - self.values[arm_index]) / self.counts[arm_index]
        self.t += 1

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.successes = np.ones(num_arms)
        self.failures = np.ones(num_arms)
    
    def give_pull(self):
        samples = np.random.beta(self.successes + 1, self.failures + 1)
        return np.argmax(samples)
    
    def get_reward(self, arm_index, reward):
        if reward == 1:
            self.successes[arm_index] += 1
        else:
            self.failures[arm_index] += 1