import os
import sys
import time
import pygame

from config import MODEL_PATH, MAZE_ENV
from utils.PygameRenderer import PygameRenderer
from utils.QLearningAgent import QLearningAgent


def test_policy_pygame(env, agent):
    """Run the agent greedily and show the path."""
    renderer = PygameRenderer(env)
    state = env.reset()
    path = [state]
    done = False
    steps = 0

    while not done and steps < 100:
        action = agent.get_action(state, greedy=True)
        next_state, _, done = env.step(action)
        path.append(next_state)
        state = next_state
        steps += 1

        renderer.draw_maze(agent_pos=state, path=path)
        renderer.check_quit()
        time.sleep(0.3)

    if done:
        print(f"Successfully reached the goal in {steps} steps.")
    else:
        print("Failed to reach the goal (too many steps).")
    time.sleep(2)
    pygame.quit()


if __name__ == "__main__":
    agent = QLearningAgent(MAZE_ENV)
    if not os.path.exists(MODEL_PATH):
        print("No saved Q-table found. Please run training (option 1) first.")
        pygame.quit()
        sys.exit(1)
    agent.load_q_table(MODEL_PATH)
    print("Testing loaded policy:")
    test_policy_pygame(MAZE_ENV, agent)
