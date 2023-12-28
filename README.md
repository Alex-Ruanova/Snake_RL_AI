# Snake_RL_AI

## Description
Snake_RL_AI is an innovative project where we've built a robust Snake game environment tailored for Reinforcement Learning (RL), specifically designed to interface with various Deep Q-Network (DQN) algorithms. This project provides a playground to train, evaluate, and visualize the performance of different RL agents as they learn to master the game of Snake.

### What's Inside:
- **Custom Snake Game Environment:** A tailored environment for training RL agents.
- **Multiple DQN Algorithms:** Supports a range of state-of-the-art DQN algorithms for comprehensive testing and learning.
- **Visualization Tools:** Integrated TensorBoard support for in-depth analysis and performance tracking.

## Available Algorithms
The environment supports the following RL algorithms:
- SAC_Discrete
- DDQN
- Dueling_DDQN
- DQN
- DQN_With_Fixed_Q_Targets
- DDQN_With_Prioritised_Experience_Replay
- A2C
- PPO
- A3C

## Installation

### Prerequisites
- **Python Version:** Ensure you have Python 3.9 installed on your system.

### Setting Up the Environment
Clone the repository and set up a virtual environment:


git clone https://github.com/Alex-Ruanova/Snake_RL_AI.git
cd Snake_RL_AI
python -m venv venv
source venv/bin/activate  # On Windows use `.\venv\Scripts\activate`

Install the required dependencies:
pip install -r requirements.txt

[!TIP] Always activate your virtual environment before running the project to ensure you're using the correct dependencies.

## Usage
To start the Snake RL agent, run:

python SnakeAgent.py

## Customizing Agents
To switch between different DQN algorithms, edit the 'agent_objs' array in SnakeAgent.py with the desired class names from the available algorithms list.

## TensorBoard Integration
For an in-depth analysis and to visualize the training performance, run:

tensorboard --logdir=logs/

Navigate to the provided link in your browser to access the TensorBoard dashboard.

[!TIP] TensorBoard provides valuable insights into the training process, helping you understand and optimize your models.

## Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Fork the Project
Create your Feature Branch (git checkout -b feature/AmazingFeature)
Commit your Changes (git commit -m 'Add some AmazingFeature')
Push to the Branch (git push origin feature/AmazingFeature)
Open a Pull Request
