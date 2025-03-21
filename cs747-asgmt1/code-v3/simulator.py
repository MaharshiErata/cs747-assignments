# simulator
# do not modify! (except final few lines)

from bernoulli_bandit import *
from set_bandit import *
from task1 import Algorithm, Eps_Greedy, UCB, KL_UCB, Thompson_Sampling
from task2 import CostlySetBanditsAlgo
from multiprocessing import Pool
import time

def single_sim(seed=0, ALGO=Algorithm, PROBS=[0.3, 0.5, 0.7], HORIZON=1000):
  np.random.seed(seed)
  shuffled_probs = np.random.permutation(PROBS)
  bandit = BernoulliBandit(probs=shuffled_probs)
  algo_inst = ALGO(num_arms=len(shuffled_probs), horizon=HORIZON)
  for t in range(HORIZON):
    arm_to_be_pulled = algo_inst.give_pull()
    reward = bandit.pull(arm_to_be_pulled)
    algo_inst.get_reward(arm_index=arm_to_be_pulled, reward=reward)
  return bandit.regret()

def single_sim_costly_set(seed=0, ALGO=CostlySetBanditsAlgo, PROBS=[0.3, 0.5, 0.7], HORIZON=1000):
  np.random.seed(seed)
  shuffled_probs = np.random.permutation(PROBS)
  bandit = CostlySetBandit(probs=shuffled_probs)
  algo_inst = ALGO(num_arms=len(shuffled_probs), horizon=HORIZON)
  for t in range(HORIZON):
    arm_set_to_be_pulled = algo_inst.give_query_set()
    reward, pulled_arm = bandit.pull(arm_set_to_be_pulled)
    algo_inst.get_reward(arm_index=pulled_arm, reward=reward)
  return bandit.net_reward()

def simulate(algorithm, probs, horizon, num_sims=50):
  """simulates algorithm of class Algorithm
  for BernoulliBandit bandit, with horizon=horizon
  """
  
  def multiple_sims(num_sims=50):
    with Pool(10) as pool:
      sim_out = pool.starmap(single_sim,
        [(i, algorithm, probs, horizon) for i in range(num_sims)])
    return sim_out 

  sim_out = multiple_sims(num_sims)
  regrets = np.mean(sim_out)

  return regrets

def simulate_costly_set(algorithm, probs, horizon, num_sims=50):
  """simulates algorithm of class Algorithm
  for BernoulliBandit bandit, with horizon=horizon
  """
  
  def multiple_sims(num_sims=50):
    with Pool(10) as pool:
      sim_out = pool.starmap(single_sim_costly_set,
        [(i, algorithm, probs, horizon) for i in range(num_sims)])
    return sim_out 

  sim_out = multiple_sims(num_sims)
  net_rewards = np.mean(sim_out)

  return net_rewards 

def task1(algorithm, probs, num_sims=50):
  """generates the plots and regrets for task1
  """
  horizons = [2**i for i in range(10, 19)]
  regrets = []
  for horizon in horizons:
    regrets.append(simulate(algorithm, probs, horizon, num_sims))

  print(regrets)
  plt.plot(horizons, regrets)
  plt.title("Regret vs Horizon")
  plt.savefig("task1-{}-{}.png".format(algorithm.__name__, time.strftime("%Y%m%d-%H%M%S")))
  plt.clf()

def task2(algorithm, probs, num_sims=50):
  """generates the plots and rewards for task2
  """

  horizons = [2**i for i in range(10, 19)]
  net_rewards = []
  for horizon in horizons:
    net_rewards.append(simulate_costly_set(algorithm, probs, horizon, num_sims))

  print(net_rewards)

if __name__ == '__main__':
  ### EDIT only the following code ###

  # TASK 1 STARTS HERE
  # Note - all the plots generated for task 1 will be for the following 
  # bandit instance:
  # 20 arms with uniformly distributed means

  task1probs = [i/20 for i in range(20)]
  # task1(Eps_Greedy, task1probs, 1)
  # task1(UCB, task1probs)
  # task1(KL_UCB, task1probs)
  # task1(Thompson_Sampling, task1probs)
  # TASK 1 ENDS HERE

  # TASK 2 STARTS HERE

  task2probs = [i/20 for i in range(20)]
  task2(CostlySetBanditsAlgo, task2probs)
  # TASK 2 ENDS HERE


