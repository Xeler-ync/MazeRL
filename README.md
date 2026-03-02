
# MazeRL - Reinforcement Learning in Maze Environment

## Installation

Install dependencies using pip:
```bash
pip install -r requirements.txt
```

## Usage

### 0. Generate a Maze
```bash
python generate_maze.py
```
This will launch an interactive menu where you can:
- Generate a new random maze
- Set custom dimensions
- Save/load maze configurations

You can actually ignore this step; the creation of maze will be attempted when ```config.py``` is imported.

### 1. Human Play
```bash
python human.py
```
Controls:
- W/A/S/D: Move up/left/down/right
- Q: Quit the game

### 2. Train the Agent
```bash
python train.py
```
This will:
- Train a Q-learning agent for 200 episodes
- Save the trained Q-table to `trained_q_table.npy`

### 3. Demo with Trained Agent
```bash
python demo.py
```
This will:
- Load the pre-trained Q-table
- Show the agent's learned behavior in the maze

## Configuration (config.py)

The `config.py` file contains the main maze configuration:

```python
MAZE_ENV = Maze(...)
```

The Maze class accepts:
- `maze_array`: 2D list where 0 = walkable cell, 1 = wall
- `start`: tuple (row, col) for starting position
- `goal`: tuple (row, col) for goal position

## Reward System

The environment provides the following rewards:
- +10: Reaching the goal
- -1: Hitting a wall (agent stays in place)
- -0.1: Each normal step (encourages shortest path)

## Agent Parameters

The Q-learning agent uses these default parameters:
- Learning rate: 0.1
- Discount factor: 0.95
- Epsilon (exploration rate): 0.3

## File Structure

```
MazeRL/
├── config.py          # Environment configuration
├── generate_maze.py   # Maze generation utilities
├── human.py          # Human player interface
├── train.py          # Training script
├── demo.py           # Demo with trained agent
└── utils/
    ├── Maze.py       # Environment implementation
    ├── QLearningAgent.py  # Q-learning agent
    └── PygameRenderer.py # Visualization
```