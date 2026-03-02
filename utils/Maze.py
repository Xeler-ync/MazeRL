import numpy as np


class Maze:
    """
    A simple grid maze.
    0 = walkable cell, 1 = wall.
    Actions: 0=up, 1=down, 2=left, 3=right.
    """

    def __init__(self, maze_array, start, goal):
        """
        maze_array : 2D list or array of 0/1
        start      : tuple (row, col)
        goal       : tuple (row, col)
        """
        self.maze = np.array(maze_array)
        self.start = start
        self.goal = goal
        self.rows, self.cols = self.maze.shape
        self.state = start

        self.actions = [0, 1, 2, 3]
        self.action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

    def reset(self):
        """Reset to start state and return it."""
        self.state = self.start
        return self.state

    def step(self, action):
        """
        Execute action, return (next_state, reward, done).
        Reward design:
            +10  for reaching goal
            -1   for hitting a wall (stay in place)
            -0.1 for each normal step (encourage shortest path)
        """
        row, col = self.state
        # Compute new position
        if action == 0:  # up
            new_row, new_col = row - 1, col
        elif action == 1:  # down
            new_row, new_col = row + 1, col
        elif action == 2:  # left
            new_row, new_col = row, col - 1
        elif action == 3:  # right
            new_row, new_col = row, col + 1
        else:
            raise ValueError(f"Invalid action {action}")

        # Check boundaries and walls
        if (
            new_row < 0
            or new_row >= self.rows
            or new_col < 0
            or new_col >= self.cols
            or self.maze[new_row, new_col] == 1
        ):
            # Hit wall: stay in place, negative reward, episode not ended
            next_state = self.state
            reward = -1.0
            done = False
        else:
            next_state = (new_row, new_col)
            if next_state == self.goal:
                reward = 10.0
                done = True
            else:
                reward = -0.1
                done = False

        self.state = next_state
        return next_state, reward, done
