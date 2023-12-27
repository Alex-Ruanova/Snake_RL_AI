import sys
import os
from tensorflow.keras.callbacks import TensorBoard
from time import time
from os.path import dirname, abspath
import tensorflow as tf
import matplotlib.pyplot as plt

from agents.DQN_agents import DDQN
from agents.DQN_agents.Dueling_DDQN import Dueling_DDQN
from agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from agents.policy_gradient_agents.REINFORCE import REINFORCE
from agents.policy_gradient_agents.PPO import PPO
from custom_environments.SnakeGame.SnakeEnvironment import SnakeGameEnv


sys.path.append(dirname(dirname(abspath(__file__))))
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
from agents.DQN_agents.DQN import DQN
from agents.DQN_agents.DDQN import DDQN
import pandas as pd


results_graph_dir = "results/data_and_graphs/"
if not os.path.exists(results_graph_dir):
    os.makedirs(results_graph_dir)


config = Config()
config.seed = 1
config.environment = SnakeGameEnv()
config.test_environment = SnakeGameEnv()
config.num_episodes_to_run = 200
config.random_episodes_to_run = 20
config.eval_every_n_steps = 100
config.fixed_action_frm_existing_policy = 3
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = False
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = True
config.save_model_location = os.path.dirname(__file__) + '/TrainedModels/'
config.turn_off_exploration = False

# Set the location to save the model and ensure the directory exists
model_save_path = os.path.join(dirname(__file__), 'TrainedModels')
config.save_model = True
config.save_model_location = model_save_path
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# Initialize TensorBoard
tensorboard_log_dir = f"logs/{config.environment.envName}_{int(time())}"
tensorboard = TensorBoard(log_dir=tensorboard_log_dir)

config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.01,
        "batch_size": 512,
        "buffer_size": 40000,
        "epsilon": 1.0,
        "epsilon_decay_rate_denominator": 1,
        "discount_rate": 0.99,
        "tau": 0.01,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.1,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 50,
        "update_learning_rate_based_on_score": False,
        "linear_hidden_units": [200, 200],
        "final_layer_activation": "None",
        "batch_norm": False,
        "gradient_clipping_norm": 0.7,
        "learning_iterations": 2,
        "clip_rewards": False
    },
    "Stochastic_Policy_Search_Agents": {
        "policy_network_type": "Linear",
        "noise_scale_start": 1e-2,
        "noise_scale_min": 1e-3,
        "noise_scale_max": 2.0,
        "noise_scale_growth_factor": 2.0,
        "stochastic_action_decision": False,
        "num_policies": 10,
        "episodes_per_policy": 1,
        "num_policies_to_keep": 5,
        "clip_rewards": False
    },
    "Policy_Gradient_Agents": {
        "learning_rate": 0.05,
        "linear_hidden_units": [20, 20],
        "final_layer_activation": "SOFTMAX",
        "learning_iterations_per_round": 5,
        "discount_rate": 0.99,
        "batch_norm": False,
        "clip_epsilon": 0.1,
        "episodes_per_learning_round": 4,
        "normalise_rewards": True,
        "gradient_clipping_norm": 7.0,
        "mu": 0.0,
        "theta": 0.0,
        "sigma": 0.0,
        "epsilon_decay_rate_denominator": 1.0,
        "clip_rewards": False
    },

    "Actor_Critic_Agents":  {

        "learning_rate": 0.005,
        "linear_hidden_units": [20, 10],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": True,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,

        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": "Softmax",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 400,
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0, #for O-H noise
        "theta": 0.15, #for O-H noise
        "sigma": 0.25, #for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    }
}
if __name__ == "__main__":
    agent_objs = [DQN]  # Agents List
    # agent_objs = [SAC_Discrete, DDQN, Dueling_DDQN, DQN, DQN_With_Fixed_Q_Targets,
                    # DDQN_With_Prioritised_Experience_Replay, A2C, PPO, A3C ]
    trainer = Trainer(config, agent_objs)
    results = trainer.run_games_for_agents()

    for agent_obj in agent_objs:
        agent_name = agent_obj.agent_name
        episode_numbers, rewards, rolling_scores, steps, _ = results[agent_name][0]

        # Preparing data for graphs
        scores = rewards
        num_steps = steps

        # Registering in TensorBoard and preparing data for Matplotlib
        score_data_for_plot = []
        steps_data_for_plot = []
        for episode_index in range(len(episode_numbers)):
            score = rolling_scores[episode_index]
            step = steps[episode_index]

            # TensorBoard registration
            tensorboard_log_dir = f"logs/{agent_name}_{int(time())}"
            tensorboard = TensorBoard(log_dir=tensorboard_log_dir)
            with tf.summary.create_file_writer(tensorboard_log_dir).as_default():
                tf.summary.scalar('Score', score, step=episode_index)
                tf.summary.scalar('Steps', step, step=episode_index)
            tf.summary.flush()

            # Adding data to Matplotlib
            score_data_for_plot.append(score)
            steps_data_for_plot.append(step)

        # Display chart using Matplotlib
        plt.figure(figsize=(10, 6))
        plt.scatter(episode_numbers, score_data_for_plot, label='Score', color='g')
        plt.scatter(episode_numbers, steps_data_for_plot, label='No of Steps', color='r')
        plt.title(f'{agent_name} Agent - Performance')
        plt.xlabel('Episode No')
        plt.ylabel('Score/Steps')
        plt.legend()
        # Save the chart in the specified directory
        graph_save_path = os.path.join(results_graph_dir, f"{agent_name.replace(' ', '_')}_Performance_Graph.png")
        plt.savefig(graph_save_path)
        print(f"Graph saved at {graph_save_path}")  # Optional: prints the path to the file where the graph was saved

        plt.show()

        # Save results in CSV
        data = pd.DataFrame({
            'Episode No': episode_numbers,
            'Score': rewards,
            'Steps': steps,
            'Rolling Score': rolling_scores
        })
        data.to_csv(f'output_{agent_name.replace(" ", "")}_{config.num_episodes_to_run}episodes.csv', encoding='utf-8-sig')

    print('TensorBoard registration completed.')