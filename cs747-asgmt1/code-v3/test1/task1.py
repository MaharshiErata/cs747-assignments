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
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
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


# START EDITING HERE
# You can use this space to define any helper functions that you need
# def kl_div(self, p, q):
#     return p * np.log((p + 1e-6) / (q + 1e-6)) + (1 - p) * np.log((1 - p + 1e-6) / (1 - q + 1e-6))

# # def compute_kl_ucb(self, arm):
# #     upper_bound = minimize_scalar(lambda q: -self.kl_div(self.values[arm], q) + np.log(self.t) / (self.counts[arm] + 1e-6), bounds=(self.values[arm], 1), method='bounded').x
# #     return upper_bound
# def compute_kl_ucb(self, arm):
#     """ Computes the KL-UCB index for the given arm using binary search instead of minimize_scalar """
#     lower, upper = self.values[arm], 1  # Search space for q
#     threshold = np.log(self.t) / (self.counts[arm] + 1e-6)
    
#     while upper - lower > 1e-6:  # Binary search precision
#         mid = (lower + upper) / 2
#         if self.kl_div(self.values[arm], mid) <= threshold:
#             lower = mid  # Move right
#         else:
#             upper = mid  # Move left
#     return lower
def kl_div(p, q):
    """Computes the KL divergence between two Bernoulli distributions with safeguards."""
    eps = 1e-15  # Small epsilon to avoid log(0)
    
    # Ensure p and q are within valid range
    p = max(eps, min(1-eps, p))
    q = max(eps, min(1-eps, q))
    
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def compute_kl_ucb(values, counts, t, arm):
    """Computes the KL-UCB index for the given arm using binary search."""
    if counts[arm] == 0:
        return 1.0  # Return maximum for unexplored arms
        
    value = values[arm]
    lower, upper = value, 1.0
    threshold = np.log(t) / counts[arm]
    
    # Safety check - if value is 1, return 1
    if value >= 1.0 - 1e-6:
        return 1.0
    
    # Binary search with maximum iterations to prevent infinite loops
    max_iterations = 100
    iterations = 0
    
    while upper - lower > 1e-6 and iterations < max_iterations:
        mid = (lower + upper) / 2
        if kl_div(value, mid) <= threshold:
            lower = mid  # Move right
        else:
            upper = mid  # Move left
        iterations += 1
    
    return lower  # Best q found within precision

# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.num_arms = num_arms
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.t = 0
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        if self.t < self.num_arms:
            return self.t
        ucb_values = self.values + np.sqrt((2 * np.log(self.t)) / (self.counts + 1e-6))
        return np.argmax(ucb_values)
        # END EDITING HERE  
        
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1
        self.values[arm_index] += (reward - self.values[arm_index]) / self.counts[arm_index]
        self.t += 1
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.num_arms = num_arms
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.t = 1
        # END EDITING HERE

    def give_pull(self):
        # START EDITING HERE
        if self.t <= self.num_arms:
            return self.t - 1
            
        kl_ucb_values = [compute_kl_ucb(self.values, self.counts, self.t, a) for a in range(self.num_arms)]
        return np.argmax(kl_ucb_values)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1
        self.values[arm_index] += (reward - self.values[arm_index]) / self.counts[arm_index]
        self.t += 1
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.num_arms = num_arms
        self.successes = np.zeros(num_arms)
        self.failures = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        samples = [np.random.beta(self.successes[a] + 1, self.failures[a] + 1) for a in range(self.num_arms)]
        return np.argmax(samples)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward == 1:
            self.successes[arm_index] += 1
        else:
            self.failures[arm_index] += 1
        # END EDITING HERE

