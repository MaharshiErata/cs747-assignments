import argparse, time
from simulator import simulate, simulate_costly_set
from task1 import Algorithm, Eps_Greedy, UCB, KL_UCB, Thompson_Sampling
from task2 import CostlySetBanditsAlgo

class Testcase:
    def __init__(self, task, probs, horizon):
        self.task = task
        self.probs = probs
        self.horizon = horizon
        self.ucb = 0
        self.kl_ucb = 0
        self.thompson = 0
        self.set_algo = 0

def read_tc(path):
    tc = None
    with open(path, 'r') as f:
        lines = f.readlines()
        task = int(lines[0].strip())
        horizon = int(lines[1].strip())
        if task == 1:
            probs = [float(p) for p in lines[2].strip().split()]
            ucb, kl_ucb, thompson = [float(x) for x in lines[3].strip().split()]
            tc = Testcase(task, probs, horizon)
            tc.ucb = ucb
            tc.kl_ucb = kl_ucb
            tc.thompson = thompson
        elif task == 2:
            probs = [float(p) for p in lines[2].strip().split()]
            reference = [float(x) for x in lines[3].strip().split()]
            tc = Testcase(task, probs, horizon)
            tc.set_algo = reference[0]
            
    return tc

def grade_task1(tc_path, algo):
    algo = algo.lower()
    tc = read_tc(tc_path)
    regrets = {}
    scores = {}
    if algo == 'ucb' or algo == 'all':
        regrets['UCB'] = simulate(UCB, tc.probs, tc.horizon)
        scores['UCB'] = 1 if regrets['UCB'] <= tc.ucb else 0
    if algo == 'kl_ucb' or algo == 'all':
        regrets['KL-UCB'] = simulate(KL_UCB, tc.probs, tc.horizon)
        scores['KL-UCB'] = 1 if regrets['KL-UCB'] <= tc.kl_ucb else 0
    if algo == 'thompson' or algo == 'all':
        regrets['Thompson Sampling'] = simulate(Thompson_Sampling, tc.probs, tc.horizon)
        scores['Thompson Sampling'] = 1 if regrets['Thompson Sampling'] <= tc.thompson else 0
    
    return scores, regrets

def grade_task2(tc_path):
    tc = read_tc(tc_path)
    reward = simulate_costly_set(CostlySetBanditsAlgo, tc.probs, tc.horizon)
    ref = tc.set_algo
    score = 0
    if reward >= ref:
        score = 1
    return score, reward 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='The task to run. Valid values are: 1, 2, all')
    parser.add_argument('--algo', type=str, required=False, help='The algo to run (for task 1 only). Valid values are: ucb, kl_ucb, thompson, all')
    args = parser.parse_args()
    pass_fail = ['FAILED', 'PASSED']

    start = time.time()
    if args.task == '1' or args.task == 'all':
        if args.task == 'all':
            args.algo = 'all'
        if args.algo is None:
            print('Please specify an algorithm for task 1')
            exit(1)
        if args.algo.lower() not in ['ucb', 'kl_ucb', 'thompson', 'all']:
            print('Invalid algorithm')
            exit(1)

        print("="*18+" Task 1 "+"="*18)
        for i in range(1, 4):
            print(f"Testcase {i}")
            scores, regrets = grade_task1(f'testcases/task1-{i}.txt', args.algo)
            for algo, score in scores.items():
                print("{:18}: {}. Regret: {:.2f}".format(algo, pass_fail[score], regrets[algo]))
            print("")
    
    if args.task == '2' or args.task == 'all':
        print("="*18+" Task 2 "+"="*18)
        for i in range(1, 9):
            print(f"Testcase {i}")
            score, reward = grade_task2(f'testcases/task2-{i}.txt')
            print("Costly Set Bandit Algorithm: {}. Net Reward: {:.2f}".format(pass_fail[score], reward))
            print("")

    end = time.time()

    print("Time elapsed: {:.2f} seconds".format(end-start))
