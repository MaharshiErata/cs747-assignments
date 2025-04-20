import argparse
import numpy as np
from pulp import *

## Parsing mdp.txt files and returning 
def parse_mdp(file_path):
    transitions = {}
    end_states = set()
    gamma = 0.0
    mdptype = ''
    num_states = 0
    num_actions = 0
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == 'numStates':
                num_states = int(parts[1])
            elif parts[0] == 'numActions':
                num_actions = int(parts[1])
                transitions = {s: {a: [] for a in range(num_actions)} for s in range(num_states)}
            elif parts[0] == 'end':
                end_states = set(map(int, parts[1:]))
            elif parts[0] == 'transition':
                s1 = int(parts[1])
                ac = int(parts[2])
                s2 = int(parts[3])
                r = float(parts[4])
                p = float(parts[5])
                transitions[s1][ac].append((s2, r, p))
            elif parts[0] == 'mdptype':
                mdptype = parts[1]
            elif parts[0] == 'discount':
                gamma = float(parts[1])
    return transitions, num_states, num_actions, end_states, mdptype, gamma

## Policy Evaluation
def evaluate_policy(policy, transitions, num_states, num_actions, end_states, gamma):
    non_terminal = [s for s in range(num_states) if s not in end_states]
    if not non_terminal:
        return {s: 0.0 for s in range(num_states)}
    n = len(non_terminal)
    A = np.zeros((n, n))
    B = np.zeros(n)
    s_to_idx = {s: i for i, s in enumerate(non_terminal)}

    for i, s in enumerate(non_terminal):
        a = policy[s]
        A[i][i] = 1.0
        rhs = 0.0
        for (s_next, r, p_trans) in transitions[s][a]:
            rhs += p_trans * r
            if s_next in end_states:
                continue
            if s_next in s_to_idx:
                j = s_to_idx[s_next]
                A[i][j] -= gamma * p_trans
        B[i] = rhs

    try:
        solution = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        solution = np.zeros(n)
    
    V = {s: 0.0 for s in end_states}
    for i, s in enumerate(non_terminal):
        V[s] = solution[i]
    return V


## Howard's Policy Iteration
def howard_policy_iteration(transitions, num_states, num_actions, end_states, gamma):
    policy = [0] * num_states
    while True:
        V = evaluate_policy(policy, transitions, num_states, num_actions, end_states, gamma)
        new_policy = policy.copy()
        policy_changed = False
        for s in range(num_states):
            if s in end_states:
                continue
            max_q = -float('inf')
            best_a = 0
            for a in range(num_actions):
                q = 0.0
                for (s_next, r, p) in transitions[s][a]:
                    if s_next in end_states:
                        q += p * r
                    else:
                        q += p * (r + gamma * V[s_next])
                if q > max_q or (q == max_q and a < best_a):
                    max_q = q
                    best_a = a
            if new_policy[s] != best_a:
                policy_changed = True
                new_policy[s] = best_a
        if not policy_changed:
            break
        policy = new_policy
    for s in end_states:
        policy[s] = 0
    return V, policy

## Linear Programming
def linear_programming(transitions, num_states, num_actions, end_states, gamma):
    prob = LpProblem("MDP_LP", LpMinimize)
    V_vars = {}
    for s in range(num_states):
        if s in end_states:
            V_vars[s] = 0.0
        else:
            V_vars[s] = LpVariable(f"V_{s}", cat='Continuous')

    prob += lpSum([V_vars[s] for s in range(num_states) if s not in end_states])

    for s in range(num_states):
        if s in end_states:
            continue
        for a in range(num_actions):
            sum_expr = 0
            for (s_next, r, p) in transitions[s][a]:
                if s_next in end_states:
                    sum_expr += p * r
                else:
                    sum_expr += p * (r + gamma * V_vars[s_next])
            prob += (V_vars[s] >= sum_expr)

    prob.solve(PULP_CBC_CMD(msg=False))

    V_values = {}
    for s in range(num_states):
        if s in end_states:
            V_values[s] = 0.0
        else:
            V_values[s] = value(V_vars[s])

    policy = [0] * num_states
    for s in range(num_states):
        if s in end_states:
            policy[s] = 0
            continue
        max_q = -float('inf')
        best_a = 0
        for a in range(num_actions):
            q = 0.0
            for (s_next, r, p) in transitions[s][a]:
                if s_next in end_states:
                    q += p * r
                else:
                    q += p * (r + gamma * V_values[s_next])
            if q > max_q or (q == max_q and a < best_a):
                max_q = q
                best_a = a
        policy[s] = best_a
    return V_values, policy

## Final main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdp', required=True)
    parser.add_argument('--algorithm', choices=['hpi', 'lp'], default='hpi')
    parser.add_argument('--policy', default=None)
    args = parser.parse_args()

    transitions, num_states, num_actions, end_states, mdptype, gamma = parse_mdp(args.mdp)

    if args.policy:
        with open(args.policy, 'r') as f:
            policy = list(map(int, f.read().split()))
        V = evaluate_policy(policy, transitions, num_states, num_actions, end_states, gamma)
        optimal_policy = policy
    else:
        if args.algorithm == 'hpi':
            V, optimal_policy = howard_policy_iteration(transitions, num_states, num_actions, end_states, gamma)
        else:
            V, optimal_policy = linear_programming(transitions, num_states, num_actions, end_states, gamma)

    for s in range(num_states):
        if s in end_states:
            print(f"{0.0:.6f} 0")
        else:
            print(f"{V[s]:.6f} {optimal_policy[s]}")

if __name__ == "__main__":
    main()
