import highway_env
import gymnasium
from gymnasium.wrappers import RecordVideo
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cma
from cma.optimization_tools import EvalParallel2
import argparse
import warnings
warnings.filterwarnings('ignore')

env = gymnasium.make('racetrack-v0', render_mode='rgb_array')

def policy(state, info, eval_mode = False, params = []):

    # The next 3 lines are used for reading policy parameters learned by training CMA-ES. Do not change them even if you don't use CMA-ES.
    # if eval_mode:
    #     param_df = pd.read_json("cmaes_params.json")
    #     params = np.array(param_df.iloc[0]["Params"])

    """Replace the default policy given below by your policy"""

    # acceleration = 0.0
    # steering = 0.0
    # return [acceleration, steering]
    default_params = [0.8, 0.05, 0.1, 15.0, 6.0, 0.2]
    if eval_mode:
        param_df = pd.read_json("cmaes_params.json")
        params = np.array(param_df.iloc[0]["Params"]).astype(float)
    else:
        params = default_params
    params = list(params[:6]) + default_params[len(params):]
    params = [
        max(0.1, min(2.0, float(params[0]))),   # Kp
        max(0.0, min(10, float(params[1]))),   # Ki
        max(0.0, min(10, float(params[2]))),   # Kd
        max(10.0, min(20.0, float(params[3]))), # target_speed
        int(max(3, min(9, round(float(params[4]))))), # lookahead
        max(0.0, min(0.5, float(params[5])))    # speed_factor
    ]
    Kp, Ki, Kd, target_speed, lookahead, speed_factor = params
    if not hasattr(policy, 'prev_error'):
        policy.prev_error = 0
        policy.integral = 0
    def get_weighted_midpoint(state, lookahead_rows):
        midpoints = []
        weights = []
        for row in lookahead_rows:
            if row < 0 or row >= 13:
                continue
            lane_centers = np.where(state[row] == 1)[0]
            if len(lane_centers) < 1:
                continue
            left = lane_centers[0]
            right = lane_centers[-1]
            width = right - left
            weight = (1.0/(13 - row)) * (width/12.0)
            if weight < 0.1:
                continue
            midpoints.append((left + right) / 2)
            weights.append(weight)
        if not midpoints:
            return 6.0, False
        weights_sum = sum(weights)
        if weights_sum == 0:
            return 6.0, False 
        return np.average(midpoints, weights=weights), True
    try:
        adaptive_lookahead = sorted(list(range(13))[-lookahead:], reverse=True)
    except:
        adaptive_lookahead = [12, 11, 10, 9, 8, 7]
    target_x, valid = get_weighted_midpoint(state, adaptive_lookahead)
    error = (target_x - 6) / 6.0
    policy.integral = 0.95 * policy.integral + error
    policy.integral = np.clip(policy.integral, -1.0, 1.0)
    derivative = error - policy.prev_error
    steering = Kp * error + Ki * policy.integral + Kd * derivative
    steering = np.clip(steering, -1.0, 1.0)
    policy.prev_error = error
    current_speed = info["speed"]
    speed_target = target_speed
    if valid:
        nearest_row = adaptive_lookahead[0]
        lane_centers = np.where(state[nearest_row] == 1)[0]
        if len(lane_centers) >= 2:
            road_width = lane_centers[-1] - lane_centers[0]
            curvature_penalty = np.clip((6 - road_width) / 6.0, 0, 1)
            speed_target *= (1 - curvature_penalty * speed_factor)
            lane_offset = abs(target_x - 6) / 6
            speed_target *= (1 - lane_offset * 0.6)
    speed_target = max(7.0, speed_target)
    speed_error = (speed_target - current_speed) / speed_target
    acceleration = np.clip(0.6 * speed_error + 0.4 * np.tanh(2*speed_error), -1.0, 1.0)
    if not info["on_road"] or current_speed > 1.1 * target_speed:
        acceleration = -1.0  # Full brake
        steering = np.clip(steering * 0.5, -0.5, 0.5)
    return [acceleration, steering]


def fitness(params):

    """This is the fitness function which is optimised by CMA-ES.
    Note that the cma library minimises the fitness function by default.
    You should make suitable adjustments to make sure fitness is maximised"""

    # Write your fitness function below. You have to write the code to interact with the environment and
    # use the information provided by the environment to formulate the fitness function in terms of CMA-ES params
    # which are provided as an argument to this function. You can refer the code provided in evaluation section of
    # the main function to see how to interact with the environment. You should invoke your policy using the following:
    # policy(state, info, False, params)

    # fitness_value = 0.0
    # return fitness_value
    total_score = 0
    num_tracks = 6
    min_score = -100
    for track in range(num_tracks):
        env.unwrapped.config["track"] = track
        obs, info = env.reset()
        state = obs[0]
        done = False
        track_score = 0
        steps = 0
        max_steps = 350
        has_been_on_road = False
        while not done and steps < max_steps:
            action = policy(state, info, False, params)
            obs, _, term, trunc, info = env.step(action)
            state = obs[0]
            done = term or trunc
            steps += 1
            if info["on_road"]:
                has_been_on_road = True
                track_score += info["distance_covered"] / max_steps
                track_score += info["speed"] / 20.0
            else:
                if has_been_on_road:
                    track_score -= 2.0
                else:
                    track_score = min_score
                    break  
        if info["distance_covered"] > 250:
            track_score *= 1.5
        if steps < max_steps and info["distance_covered"] < 100:
            track_score -= (max_steps - steps) * 0.1
        total_score += max(track_score, min_score)
    return -total_score / num_tracks

def call_cma(num_gen=10, pop_size=3, num_policy_params = 6):
    sigma0 = 0.5
    x0 = [0.8, 0.05, 0.1, 15.0, 6.0, 0.2]
    opts = {
        'maxiter': num_gen,
        'popsize': pop_size,
        'bounds': [
            [0.1, 0.0, 0.0, 10.0, 3.0, 0.0],
            [2.0, 0.3, 0.5, 20.0, 9.0, 0.5]
        ],
        'integer_variables': [4],
        'verbose': -1
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    with EvalParallel2(fitness, es.popsize + 1) as eval_all:
        while not es.stop():
            X = es.ask()
            es.tell(X, eval_all(X))
            es.logger.add()  # write data to disc for plotting
            es.disp()
    es.result_pretty()
    return es.result

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true')  # For training using CMA-ES
    parser.add_argument("--eval", action='store_true')  # For evaluating a trained CMA-ES policy
    parser.add_argument("--numTracks", type=int, default=6, required=False)  # Number of tracks for evaluation
    parser.add_argument("--seed", type=int, default=2025, required=False)  # Seed for evaluation
    parser.add_argument("--render", action='store_true')  # For rendering the evaluations
    args = parser.parse_args()

    train_mode = args.train
    eval_mode = args.eval
    num_tracks = args.numTracks
    seed = args.seed
    rendering = args.render

    """CMA-ES code begins"""
    # You can skip this part if you don't intend to use CMA-ES

    if train_mode:
        num_gen = 100
        pop_size = 6
        num_policy_params = 6
        X = call_cma(num_gen, pop_size, num_policy_params)
        cmaes_params = X[0]  # Parameters returned by CMA-ES after training
        cmaes_params_df = pd.DataFrame({
            'Params': [cmaes_params]
        })
        cmaes_params_df.to_json("cmaes_params.json")  # Storing parameters for evaluation purpose

    """CMA-ES code ends"""

    """Evaluation code begins"""
    # Do not modify this part.

    if rendering:
        env = RecordVideo(env, video_folder="videos", name_prefix="eval", episode_trigger=lambda x: True)

    if not train_mode:
        track_score_list = []  # This list stores the scores for different tracks

        for t in range(num_tracks):
            env.unwrapped.config["track"] = t  # Configuring the environment to provide track associated with index t. There are 6 tracks indexed 0 to 5.
            (obs, info) = env.reset(seed=seed)  # Getting initial state information from the environment
            state = obs[0]
            done = False

            while not done:  # While the episode is not done
                action = policy(state, info, eval_mode)  # Call policy to produce action
                (obs, _, term, trunc, info) = env.step(action)  # Take action in the environment
                state = obs[0]
                done = term or trunc  # If episode has terminated or truncated, set boolean variable done to True

            track_score = np.round(info["distance_covered"], 4).item()  # .item() converts numpy float to python float
            print("Track " + str(t) + " score:", track_score)
            track_score_list.append(track_score)

        env.close()

        # The next 4 lines of code generate a performance file which is used by autograder for evaluation. Don't change anything here.
        perf_df = pd.DataFrame()
        perf_df["Track_number"] = [n for n in range(num_tracks)]
        perf_df["Score"] = track_score_list
        perf_df.to_json("Performance_" + str(seed) + ".json")

        # A scatter plot is generated for you to visualise the performance of your agent across different tracks
        plt.scatter(np.arange(len(track_score_list)), track_score_list)
        plt.xlabel("Track index")
        plt.ylabel("Scores")
        plt.title("Scores across various tracks")
        plt.savefig('Evaluation.jpg')
        plt.close()

    """Code to generate learning curve and logs of CMA-ES"""
    # To be used only if your policy has parameters which are optimised using CMA-ES
    if train_mode:
        datContent = [i.strip().split() for i in open("outcmaes/fit.dat").readlines()]

        generations = []
        evaluations = []
        bestever = []
        best = []
        median = []
        worst = []

        for i in range(1, len(datContent)):
            generations.append(int(datContent[i][0]))
            evaluations.append(int(datContent[i][1]))
            bestever.append(-float(datContent[i][4]))
            best.append(-float(datContent[i][5]))
            median.append(-float(datContent[i][6]))
            worst.append(-float(datContent[i][7]))

        logs_df = pd.DataFrame()
        logs_df['Generations'] = generations
        logs_df['Evaluations'] = evaluations
        logs_df['BestEver'] = bestever
        logs_df['Best'] = best
        logs_df['Median'] = median
        logs_df['Worst'] = worst

        logs_df.to_csv('logs.csv')

        plt.plot(generations, best, color='green')
        plt.plot(generations, median, color='blue')
        plt.xlabel("Number of generations")
        plt.ylabel("Fitness")
        plt.legend(["Best", "Median"])
        plt.title('Evolution of fitness across generations')
        plt.savefig('LearningCurve.jpg')
        plt.close()

