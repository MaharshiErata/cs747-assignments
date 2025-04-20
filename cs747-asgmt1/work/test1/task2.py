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
# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

class CostlySetBanditsAlgo(Algorithm):
    def __init__(self, num_arms, horizon):
        # You can add any other variables you need here
        self.num_arms = num_arms
        self.horizon = horizon
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.t = 0

        self.rewards = np.zeros(num_arms)
        # Beta parameters for Thompson sampling
        self.alpha = np.ones(num_arms)
        self.beta = np.ones(num_arms)
        
        # Adaptive cost estimation - start conservative
        self.estimated_costs = np.full(num_arms, 0.2)
        self.cost_counts = np.zeros(num_arms)
        
        # Exploration parameters
        self.initial_pulls = 2  # Pull each arm at least this many times
        self.epsilon = 0.1      # Probability of random exploration
        # END EDITING HERE
    
    def give_query_set(self):
        # START EDITING HERE
        unexplored_arms = np.where(self.counts < self.initial_pulls)[0]
        if len(unexplored_arms) > 0:
            return [unexplored_arms[0]]
        
        # Use Thompson sampling to estimate arm values
        samples = np.random.beta(self.alpha, self.beta)
        
        # With probability epsilon, explore randomly
        if np.random.random() < self.epsilon:
            # Pick between 1-3 arms randomly for exploration
            num_explore = min(1 + np.random.randint(3), self.num_arms)
            return np.random.choice(self.num_arms, size=num_explore, replace=False).tolist()
        
        # Calculate net expected value (value - cost) for each arm
        net_values = samples - self.estimated_costs
        
        # Start with empty query set
        query_set = []
        
        # Always include the arm with highest net value if it's positive
        if np.max(net_values) > 0:
            best_arm = np.argmax(net_values)
            query_set.append(best_arm)
        else:
            # If all net values are negative, just pull the arm with highest expected value
            # This should rarely happen after initial exploration
            best_arm = np.argmax(samples)
            return [best_arm]
        
        # Decide which additional arms to include
        remaining_arms = list(range(self.num_arms))
        remaining_arms.remove(best_arm)
        
        # Use diminishing returns model - each additional arm should have 
        # significant value compared to the best arm
        best_value = samples[best_arm]
        for arm in sorted(remaining_arms, key=lambda a: -samples[a]):
            expected_gain = samples[arm] - best_value * 0.8  # Require at least 20% improvement
            if expected_gain > self.estimated_costs[arm]:
                query_set.append(arm)
            else:
                # Stop once we find an arm not worth including
                break
        
        # Safety check: limit query set size based on t
        if self.t < self.horizon // 4:
            max_size = 3  # Early phases: be more exploratory
        else:
            max_size = 1  # Later phases: be more conservative
        
        if len(query_set) > max_size:
            query_set = query_set[:max_size]
        
        return query_set
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        # self.counts[arm_index] += 1
        # self.values[arm_index] += (reward - self.values[arm_index]) / self.counts[arm_index]
        # self.t += 1
        # Update counts and running averages
        self.counts[arm_index] += 1
        self.rewards[arm_index] += reward
        self.values[arm_index] = self.rewards[arm_index] / self.counts[arm_index]
        
        # Update beta distribution parameters for Thompson sampling
        if reward == 1:
            self.alpha[arm_index] += 1
        else:
            self.beta[arm_index] += 1
        
        # Adaptive cost estimation (this is a proxy since we don't know the actual cost)
        # The idea is to be more conservative as we gain more information
        if self.t > 0:
            # Reduce cost estimate slightly for arms that perform well
            if self.values[arm_index] > np.mean(self.values):
                self.estimated_costs[arm_index] *= 0.99
            else:
                self.estimated_costs[arm_index] *= 1.01
            
            # Ensure cost estimates stay within reasonable bounds
            self.estimated_costs[arm_index] = max(0.05, min(0.3, self.estimated_costs[arm_index]))
        
        # Decrease exploration probability over time
        self.epsilon = max(0.01, self.epsilon * 0.999)
        
        self.t += 1
        #END EDITING HERE

