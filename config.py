import os

from utils.Maze import Maze

os.mkdir("training_results")

TRAINING_RESULTS_DIR = "./training_results"
MODEL_NAME = "trained_q_table.npy"
MODEL_PATH = os.path.join(TRAINING_RESULTS_DIR, MODEL_NAME)

MAZE_ENV = Maze(
    [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0],
    ],
    start=(0, 0),
    goal=(4, 4),
)
