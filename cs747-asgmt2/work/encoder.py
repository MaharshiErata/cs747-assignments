import argparse

def load_gridworld(file_path):
    grid = []
    with open(file_path, 'r') as f:
        for line in f:
            row = line.strip().split()
            if row:
                grid.append(row)
    return grid

def encode_gridworld(gridworld_file):
    grid = load_gridworld(gridworld_file)
    if not grid:
        raise ValueError("Empty gridworld file")
    
    state_map = {}
    terminal_states = set()
    state_id = 0

    # Create state mappings
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            cell = grid[i][j]
            if cell == 'W':
                continue
            if cell == 'g':  # Terminal states
                for direction in range(4):
                    for has_key in [0, 1]:
                        state_map[(i, j, direction, has_key)] = state_id
                        terminal_states.add(state_id)
                        state_id += 1
                continue
            for direction in range(4):
                for has_key in [0, 1]:
                    state_map[(i, j, direction, has_key)] = state_id
                    state_id += 1

    num_states = len(state_map)
    num_actions = 4
    transitions = []

    # Generate transitions
    for (i, j, d, k) in state_map:
        s = state_map[(i, j, d, k)]
        cell = grid[i][j]
        
        if s in terminal_states:
            continue

        for action in range(num_actions):
            if action == 0:  # Move forward
                # Block move through closed door
                if cell == 'd' and k == 0:
                    continue

                directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]
                di, dj = directions[d]
                
                possible_steps = []
                current_i, current_j = i, j
                for _ in range(3):
                    current_i += di
                    current_j += dj
                    if (0 <= current_i < len(grid)) and (0 <= current_j < len(grid[current_i])) and (grid[current_i][current_j] != 'W'):
                        possible_steps.append((current_i, current_j))
                    else:
                        break

                if not possible_steps:
                    continue

                # Normalize probabilities
                probs = [0.5, 0.3, 0.2][:len(possible_steps)]
                total = sum(probs)
                probs = [p/total for p in probs]

                for idx, (ni, nj) in enumerate(possible_steps):
                    new_k = 1 if grid[ni][nj] == 'k' else k
                    s_next = state_map[(ni, nj, d, new_k)]
                    transitions.append(f"transition {s} {action} {s_next} -1 {probs[idx]:.6f}")

            else:  # Turn actions
                new_dirs = []
                probs = []
                if action == 1:  # Turn left
                    new_dirs = [(d-1)%4]
                    probs = [1.0]
                elif action == 2:  # Turn right
                    new_dirs = [(d+1)%4]
                    probs = [1.0]
                else:  # Turn around
                    new_dirs = [(d+2)%4]
                    probs = [1.0]

                for nd, p in zip(new_dirs, probs):
                    s_next = state_map[(i, j, nd, k)]
                    transitions.append(f"transition {s} {action} {s_next} -1 {p:.6f}")

    print(f"numStates {num_states}")
    print(f"numActions {num_actions}")
    print(f"end {' '.join(map(str, terminal_states))}")
    print('\n'.join(transitions))
    print("mdptype episodic")
    print("discount 1.0")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gridworld", required=True)
    args = parser.parse_args()
    encode_gridworld(args.gridworld)

