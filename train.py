import os.path
import time
import numpy as np
import pygame

from config import MODEL_PATH, MAZE_ENV
from utils.PygameRenderer import PygameRenderer
from utils.QLearningAgent import QLearningAgent


def train_with_pygame(
    env,
    agent,
    episodes=200,
    max_steps=100,
    render_interval=10,
    convergence_window=100,
    convergence_threshold=1e-3,
):
    """
    Train the agent with early stopping based on convergence.
    """
    renderer = PygameRenderer(env) if render_interval > 0 else None
    print_interval = (
        render_interval if render_interval > 0 else min(100, int(episodes / 10))
    )
    episode_rewards = []
    episode_lengths = []
    start_time = time.time()
    episode_times = []

    for ep in range(episodes):
        ep_start_time = time.time()
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        path = [state]  # for display

        while not done and steps < max_steps:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            path.append(state)
            total_reward += reward
            steps += 1

            # Render if this episode is selected
            if renderer and ep % render_interval == 0:
                renderer.draw_maze(agent_pos=state, path=path)
                renderer.check_quit()  # allow window closing
                time.sleep(0.05)  # slow down for visibility

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        ep_time = time.time() - ep_start_time
        episode_times.append(ep_time)

        # Check for convergence
        if len(episode_rewards) >= 2 * convergence_window:
            recent_avg = np.mean(episode_rewards[-convergence_window:])
            prev_avg = np.mean(
                episode_rewards[-2 * convergence_window : -convergence_window]
            )
            if abs(recent_avg - prev_avg) < convergence_threshold:
                print(f"\nConverged after {ep+1} episodes!")
                print(
                    f"Average reward over last {convergence_window} episodes: {recent_avg:.2f}"
                )
                break

        if (ep + 1) % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            avg_length = np.mean(episode_lengths[-print_interval:])
            avg_time = np.mean(episode_times[-print_interval:])
            total_time = time.time() - start_time
            print(
                f"Episode {ep+1}/{episodes} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Avg Length: {avg_length:.2f} | "
                f"Avg Episode Time: {avg_time:.2f}s | "
                f"Total Time: {total_time:.2f}s"
            )

    if renderer:
        pygame.quit()

    total_time = time.time() - start_time
    avg_episode_time = total_time / (ep + 1)
    print("\nTraining Summary:")
    print(f"Total Training Time: {total_time:.2f}s")
    print(f"Average Episode Time: {avg_episode_time:.2f}s")
    print(f"Episodes per Second: {(ep+1)/total_time:}")

    return episode_rewards, episode_lengths


if __name__ == "__main__":
    agent = QLearningAgent(
        MAZE_ENV, learning_rate=0.1, discount_factor=0.95, epsilon=0.3
    )
    print("Training started")
    rewards, lengths = train_with_pygame(
        MAZE_ENV,
        agent,
        # Although the episodes is large, it will automatically terminate upon convergence.
        episodes=MAZE_ENV.maze.size**3,
        render_interval=0,
        convergence_window=1000,
        convergence_threshold=1e-8,
    )
    # Save the trained Q-tablef
    agent.save_q_table(MODEL_PATH)
