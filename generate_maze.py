import json
import os
import random
from typing import List, Tuple

from config import MAZE_PATH


def generate_maze(
    width: int, height: int
) -> Tuple[List[List[int]], Tuple[int, int], Tuple[int, int]]:
    """
    Generate a random maze using modified recursive backtracking algorithm.
    First creates a path from start to goal, then adds branches.
    Returns the maze, start position, and goal position.
    """
    # Initialize maze with all walls
    maze = [[1 for _ in range(width)] for _ in range(height)]

    # Randomly select start and end positions on opposite edges
    if random.random() < 0.5:
        # Horizontal layout
        start = (random.randint(0, height - 1), 0)
        goal = (random.randint(0, height - 1), width - 1)
    else:
        # Vertical layout
        start = (0, random.randint(0, width - 1))
        goal = (height - 1, random.randint(0, width - 1))

    # Start generating maze from start position
    visited = set()
    _carve_passages(start[1], start[0], visited, width, height, maze, goal[1], goal[0])

    # Create branches along the main path
    path = [(x, y) for y in range(height) for x in range(width) if maze[y][x] == 0]
    _create_branches(
        path, width, height, maze, visited, branch_probability=0.3, max_branch_length=3
    )

    return maze, start, goal


def _carve_passages(
    cx: int,
    cy: int,
    visited: set,
    width: int,
    height: int,
    maze: List[List[int]],
    goal_x: int,
    goal_y: int,
):
    """
    Recursively carve passages through the maze using a modified algorithm
    that prioritizes unvisited directions while maintaining some randomness.
    First creates a path to goal, then generates branches.
    """
    maze[cy][cx] = 0  # Mark current position as a path
    visited.add((cx, cy))

    # Calculate direction priority based on distance to goal and exploration status
    directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
    directions.sort(
        key=lambda d: (
            random.random(),
            sum(
                0 <= cx + dx < width
                and 0 <= cy + dy < height
                and (cx + dx, cy + dy) not in visited
                for dx, dy in [(d[0] * 2, d[1] * 2)]
            ),
            -abs((cx + d[0]) - goal_x) - abs((cy + d[1]) - goal_y),
        )
    )

    for dx, dy in directions:
        nx, ny = cx + dx, cy + dy
        if (
            0 <= nx < width
            and 0 <= ny < height
            and maze[ny][nx] == 1
            and (nx, ny) not in visited
        ):
            # Carve passage between current and next position
            maze[cy + dy // 2][cx + dx // 2] = 0
            _carve_passages(nx, ny, visited, width, height, maze, goal_x, goal_y)


def _create_branches(
    path: List[Tuple[int, int]],
    width: int,
    height: int,
    maze: List[List[int]],
    visited: set,
    branch_probability: float = 0.3,
    max_branch_length: int = 3,
):
    """
    Create random branches along the main path with controlled probability and length.
    Ensures branches don't overlap with existing paths.
    """
    for x, y in path:
        if random.random() < branch_probability:
            # Try to create a branch in a random direction
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 1:
                    # Create a branch with limited length
                    current_x, current_y = x, y
                    for _ in range(max_branch_length):
                        next_x = current_x + dx
                        next_y = current_y + dy

                        if not (0 <= next_x < width and 0 <= next_y < height):
                            break
                        if maze[next_y][next_x] == 0:
                            break

                        maze[current_y + dy // 2][current_x + dx // 2] = 0
                        maze[next_y][next_x] = 0
                        visited.add((next_x, next_y))

                        current_x, current_y = next_x, next_y
                    break


def print_maze(maze: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]):
    """
    Print the maze to the console with S (start) and G (goal) markers,
    surrounded by a border.

    Args:
        maze: 2D array representing the maze
        start: Start position (row, col)
        goal: Goal position (row, col)
    """
    height = len(maze)
    width = len(maze[0])

    # Top border
    print("█" * (width + 2))

    for y in range(height):
        # Left border
        print("█", end="")

        for x in range(width):
            if (x, y) == start:  # Start position
                print("S", end="")
            elif (x, y) == goal:  # Goal position
                print("G", end="")
            elif maze[y][x] == 1:  # Wall
                print("█", end="")
            else:  # Path
                print(" ", end="")

        # Right border
        print("█")

    # Bottom border
    print("█" * (width + 2))


def get_user_input():
    """Get maze dimensions from user input."""
    while True:
        try:
            width = int(input("Enter maze width (odd number recommended): "))
            height = int(input("Enter maze height (odd number recommended): "))
            if width > 0 and height > 0:
                return width, height
            print("Please enter positive numbers.")
        except ValueError:
            print("Please enter valid numbers.")


def save_maze(maze, start, goal, filename):
    """Save maze configuration to a JSON file."""
    data = {"maze": maze, "start": start, "goal": goal}
    with open(filename, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"Maze saved to {filename}")


def load_maze(filename):
    """Load maze configuration from a JSON file."""
    if not os.path.exists(filename):
        return None
    with open(filename, "r") as f:
        data = json.load(f)
    return data["maze"], data["start"], data["goal"]


def main():
    """Main function with user menu."""
    while True:
        print("\nMaze Generator Menu:")
        print("1. Generate new maze")
        print("2. Save current maze")
        print("3. Show saved maze")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            width, height = get_user_input()
            maze, start, goal = generate_maze(width, height)
            print("\nGenerated Maze:")
            print(f"Start: {start}")
            print(f"Goal: {goal}")
            print_maze(maze, start, goal)

        elif choice == "2":
            if "maze" in locals():
                filename = (
                    input(f"Enter filename (default: {MAZE_PATH}): ") or MAZE_PATH
                )
                save_maze(maze, start, goal, filename)
            else:
                print("No maze to save. Please generate one first.")

        elif choice == "3":
            filename = input(f"Enter filename (default: {MAZE_PATH}): ") or MAZE_PATH
            loaded = load_maze(filename)
            if loaded:
                maze, start, goal = loaded
                print("\nLoaded Maze:")
                print(f"Start: {start}")
                print(f"Goal: {goal}")
                print_maze(maze, start, goal)
            else:
                print("File not found.")

        elif choice == "4":
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
