import pygame

from utils.Maze import Maze
from utils.PygameRenderer import PygameRenderer


def human_play_pygame(env):
    """Let a human player control the agent with keyboard."""
    renderer = PygameRenderer(env)
    state = env.reset()
    path = [state]
    done = False
    print("Press 'w', 'a', 's', 'd' to move the agent. Press 'q' to quit.")

    final_reward = 0
    while not done:
        renderer.draw_maze(agent_pos=state, path=path)
        action = renderer.wait_key()
        if action == "quit":
            break
        next_state, reward, done = env.step(action)
        path.append(next_state)
        state = next_state
        final_reward += reward
        print(f"Reward: {reward}")

        if done:
            renderer.draw_maze(agent_pos=state, path=path)
            print("=" * 30)
            print("You reached the goal.")
            print(f"Your final reward is: {final_reward:.1f}")

    pygame.quit()
    input("Press Enter to exit")


if __name__ == "__main__":
    human_play_pygame(
        Maze(
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
    )
