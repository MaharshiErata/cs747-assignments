# CS747 Programming Assignments

This repository contains my submissions for the CS747 Spring 2025 Programming Assignments at IIT Bombay. The assignments cover core concepts in reinforcement learning, including multi-armed bandits, Markov Decision Processes (MDPs), and optimal control in dynamic environments.

## Table of Contents
- [Assignment 1: Multi-Armed Bandits](#assignment-1-multi-armed-bandits)
- [Assignment 2: MDP Planning and Modeling](#assignment-2-mdp-planning-and-modeling)
- [Assignment 3: Optimal Driving Control](#assignment-3-optimal-driving-control)
- [Environment Setup](#environment-setup)
- [How to Run](#how-to-run)
- [Reports](#reports)
- [References](#references)

---

## Assignment 1: Multi-Armed Bandits

This assignment involved implementing and analyzing various algorithms for the multi-armed bandit problem:

- **Task 1**: Implemented UCB, KL-UCB, and Thompson Sampling algorithms.
- **Task 2**: Designed a cost-sensitive bandit algorithm for settings with query costs.
- **Task 3**: Investigated epsilon values in epsilon-greedy policy and analyzed regret.

**Code Files**:
- `task1.py`, `task2.py`, `task3.py`

**Evaluation**:
- Autograded via `autograder.py`
- Regret plots and analysis included in `report.pdf`

---

## Assignment 2: MDP Planning and Modeling

This assignment focused on solving MDPs using:

- **Task 1**: Howard’s Policy Iteration and Linear Programming.
- **Task 2**: Formulating a gridworld problem as an MDP and solving it using the implemented planner.

**Code Files**:
- `planner.py`: Solves MDPs using specified algorithm
- `encoder.py`: Encodes gridworld into an MDP
- `decoder.py`: Decodes the planner's output to grid actions
- `autograder.py`: Verifies outputs for correctness

**Evaluation**:
- Tested on provided and hidden MDP/gridworld instances
- Report describes design decisions and MDP modeling

---

## Assignment 3: Optimal Driving Control

Designed a reinforcement learning agent to drive a car on challenging tracks using the `highway-env` simulator.

- Policy designed to map local road observations to acceleration and steering.
- Optional use of **CMA-ES** for optimizing policy parameters.
- Agent evaluated on 6 public and hidden tracks.

**Code Files**:
- `main.py`: Core logic with policy and CMA-ES integration
- `cmaes_params.json`: Stores optimized policy parameters (if CMA-ES used)
- `videos/`: Directory for rendered simulations

**Evaluation**:
- Performance compared against baseline agents on unseen tracks
- Report discusses policy design, parameter optimization, and fitness strategy

---

## Environment Setup

Each assignment includes a `requirements.txt` or virtual environment instructions. Make sure to use the appropriate Python version:
- Assignment 1 & 2: Python 3.9.6
- Assignment 3: Python 3.12.0

Set up virtual environments using:
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
