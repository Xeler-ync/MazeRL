import os
import json

from utils.Maze import Maze

os.makedirs("training_results", exist_ok=True)

TRAINING_RESULTS_DIR = "./training_results"
MODEL_NAME = "trained_q_table.npy"
MODEL_PATH = os.path.join(TRAINING_RESULTS_DIR, MODEL_NAME)
MAZE_NAME = "maze.json"
MAZE_PATH = os.path.join(TRAINING_RESULTS_DIR, MAZE_NAME)
MAZE_DEFAULT_WIDTH = 21
MAZE_DEFAULT_HEIGHT = 21


def load_or_generate_maze():
    """Load maze from file or generate a new one if file doesn't exist."""
    if os.path.exists(MAZE_PATH):
        with open(MAZE_PATH, "r") as f:
            data = json.load(f)
        return Maze(data["maze"], tuple(data["start"]), tuple(data["goal"]))
    else:
        from generate_maze import generate_maze, save_maze

        maze, start, goal = generate_maze(MAZE_DEFAULT_WIDTH, MAZE_DEFAULT_HEIGHT)
        save_maze(maze, start, goal, MAZE_PATH)
        return Maze(maze, start, goal)


MAZE_ENV = load_or_generate_maze()
