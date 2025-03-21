"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the CostlySetBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_query_set(self): This method is called when the algorithm needs to
        provide a query set to the oracle. The method should return an array of 
        arm indices that specifies the query set.
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_query_set method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
"""

import numpy as np
from task1 import Algorithm

class CostlySetBanditsAlgo(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.alpha = np.ones(num_arms)
        self.beta = np.ones(num_arms)
        self.t = 0

    def give_query_set(self):
        samples = np.random.beta(self.alpha, self.beta)
        sorted_arms = np.argsort(samples)[::-1]
        best_k = 1
        best_value = (samples[sorted_arms[0]] - 1) / 1
        sum_p = samples[sorted_arms[0]]
        for k in range(2, self.num_arms + 1):
            sum_p += samples[sorted_arms[k-1]]
            current_value = (sum_p - 1) / k
            if current_value > best_value:
                best_value = current_value
                best_k = k
        return sorted_arms[:best_k].tolist()

    def get_reward(self, arm_index, reward):
        if reward == 1:
            self.alpha[arm_index] += 1
        else:
            self.beta[arm_index] += 1
        self.t += 1