# do not modify!

import numpy as np
import matplotlib.pyplot as plt
from bernoulli_bandit import BernoulliArm

class CostlySetBandit:
  def __init__(self, probs=[0.3, 0.5, 0.7]):
    self.__arms = [BernoulliArm(p) for p in probs]
    self.__net_reward = 0

  def pull(self, arm_set):
    actual_set = list(set(arm_set))
    index = np.random.choice(actual_set)
    reward = self.__arms[index].pull()
    self.__net_reward += reward - 1/len(actual_set)
    return reward, index

  def net_reward(self):
    return self.__net_reward
  
  def num_arms(self):
    return len(self.__arms)
