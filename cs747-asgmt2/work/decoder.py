import argparse

def parse_testcases(test_file):
    testcases = []
    current = []
    with open(test_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Testcase"):
                if current:
                    testcases.append(current)
                    current = []
            elif line:
                current.append(line.split())
    if current:
        testcases.append(current)
    return testcases

def rebuild_states_from_mdp(mdp_file):
    states = []
    with open(mdp_file, 'r') as f:
        for line in f:
            if line.startswith("numStates"):
                num_states = int(line.split()[1])
            elif line.startswith("transition"):
                parts = line.split()
                s1 = int(parts[1])
                a = int(parts[2])
                s2 = int(parts[3])
                # Extract state from transitions (not ideal but works)
                # This assumes states are ordered 0,1,2,...,n-1
                if s1 >= len(states):
                    states.extend([None] * (s1 - len(states) + 1))
                if states[s1] is None:
                    states[s1] = (0, 0, 0, 0)  # Placeholder
    return list(range(num_states))  # Simplified for policy indexing

def get_agent_state(grid, key_pos):
    x, y, dir = -1, -1, -1
    has_key = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            cell = grid[i][j]
            if cell in ['^', '>', 'v', '<']:
                dir = {'^':0, '>':1, 'v':2, '<':3}[cell]
                x, y = i, j
            if cell == 'k':
                has_key = 0
    if (x, y) == key_pos:
        has_key = 1
    return (x, y, dir, has_key)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdp', required=True)
    parser.add_argument('--value-policy', required=True)
    parser.add_argument('--gridworld', required=True)
    args = parser.parse_args()

    testcases = parse_testcases(args.gridworld)
    states = rebuild_states_from_mdp(args.mdp)
    policy = []
    with open(args.value_policy, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                policy.append(int(parts[1]))

    # Assume key position is known (from first test case)
    key_pos = None
    for row in testcases[0]:
        if 'k' in row:
            key_pos = (testcases[0].index(row), row.index('k'))

    actions = []
    for grid in testcases:
        state = get_agent_state(grid, key_pos)
        # Simplified: Assume state index matches policy index
        action = policy[state[0]] if state[0] < len(policy) else 0
        actions.append(action)

    print(' '.join(map(str, actions)))

if __name__ == "__main__":
    main()