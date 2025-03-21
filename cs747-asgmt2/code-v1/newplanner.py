import argparse
import numpy as np
import pulp

def load_mdp(mdp_file):
    with open(mdp_file, 'r') as f:
        lines = f.readlines()
    
    num_states = int(lines[0].split()[1])
    num_actions = int(lines[1].split()[1])
    transitions = {s: {a: [] for a in range(num_actions)} for s in range(num_states)}
    end_states = set()
    gamma = 1.0
    mdptype = "continuing"

    for line in lines[2:]:
        parts = line.strip().split()
        if parts[0] == "end":
            end_states = set(map(int, parts[1:]))
        elif parts[0] == "transition":
            s1, a, s2, r, p = int(parts[1]), int(parts[2]), int(parts[3]), float(parts[4]), float(parts[5])
            transitions[s1][a].append((s2, r, p))
        elif parts[0] == "mdptype":
            mdptype = parts[1]
        elif parts[0] == "discount":
            gamma = float(parts[1])
    
    return num_states, num_actions, transitions, end_states, gamma, mdptype

def policy_iteration(num_states, num_actions, transitions, end_states, gamma):
    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    stable = False
    
    while not stable:
        stable = True
        for s in range(num_states):
            if s in end_states:
                continue
            a = policy[s]
            V[s] = sum(p * (r + gamma * V[s2]) for s2, r, p in transitions[s][a])
        
        for s in range(num_states):
            if s in end_states:
                continue
            best_action = max(range(num_actions), key=lambda a: sum(p * (r + gamma * V[s2]) for s2, r, p in transitions[s][a]))
            if best_action != policy[s]:
                policy[s] = best_action
                stable = False
    return V, policy

def linear_programming(num_states, num_actions, transitions, end_states, gamma):
    prob = pulp.LpProblem("MDP", pulp.LpMinimize)
    V_vars = [pulp.LpVariable(f"V_{s}") for s in range(num_states)]
    prob += pulp.lpSum(V_vars)

    for s in range(num_states):
        if s in end_states:
            prob += V_vars[s] == 0
        else:
            for a in range(num_actions):
                prob += V_vars[s] >= pulp.lpSum(p * (r + gamma * V_vars[s2]) for s2, r, p in transitions[s][a])
    prob.solve()
    
    V = np.array([V_vars[s].varValue for s in range(num_states)])
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        if s not in end_states:
            policy[s] = max(range(num_actions), key=lambda a: sum(p * (r + gamma * V[s2]) for s2, r, p in transitions[s][a]))
    return V, policy

def run_algorithm(mdp_file, algorithm):
    num_states, num_actions, transitions, end_states, gamma, mdptype = load_mdp(mdp_file)
    if algorithm == "hpi":
        return policy_iteration(num_states, num_actions, transitions, end_states, gamma)
    elif algorithm == "lp":
        return linear_programming(num_states, num_actions, transitions, end_states, gamma)
    return policy_iteration(num_states, num_actions, transitions, end_states, gamma)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp", required=True, help="Path to the MDP file")
    parser.add_argument("--algorithm", choices=["hpi", "lp"], default="hpi", help="Algorithm to use")
    args = parser.parse_args()
    
    V, policy = run_algorithm(args.mdp, args.algorithm)
    
    for s in range(len(V)):
        print(f"{V[s]:.6f} {policy[s]}")
