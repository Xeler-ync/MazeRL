import sys
import pygame

pygame.init()


class PygameRenderer:
    """Handles all drawing and user input via Pygame."""

    COLORS = {
        "wall": (40, 40, 40),  # dark gray
        "path": (255, 255, 255),  # white
        "start": (0, 255, 0),  # green
        "goal": (255, 0, 0),  # red
        "agent": (0, 0, 255),  # blue
        "trail": (200, 200, 100),  # light yellow
        "grid_line": (100, 100, 100),  # gray
    }

    def __init__(self, maze, cell_size=60):
        self.maze = maze
        self.cell_size = cell_size
        self.width = maze.cols * cell_size
        self.height = maze.rows * cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Maze Reinforcement Learning")

        # Initialize font safely
        pygame.font.init()
        self.font = pygame.font.Font(None, 18)

        self.clock = pygame.time.Clock()

    def draw_maze(self, agent_pos=None, path=None):
        """
        Draw the maze, optionally with a trail (path) and the agent.
        agent_pos : current position of the agent (tuple)
        path      : list of positions visited so far in the current episode
        """
        self.screen.fill(self.COLORS["wall"])

        # Draw each cell
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                rect = pygame.Rect(
                    c * self.cell_size,
                    r * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                if self.maze.maze[r, c] == 0:
                    pygame.draw.rect(self.screen, self.COLORS["path"], rect)
                pygame.draw.rect(self.screen, self.COLORS["grid_line"], rect, 1)

        # Draw trail (if provided)
        if path:
            for pos in path:
                if pos != self.maze.start and pos != self.maze.goal:
                    r, c = pos
                    center = (
                        c * self.cell_size + self.cell_size // 2,
                        r * self.cell_size + self.cell_size // 2,
                    )
                    pygame.draw.circle(
                        self.screen, self.COLORS["trail"], center, self.cell_size // 4
                    )

        # Draw start cell (with 'S')
        sr, sc = self.maze.start
        start_rect = pygame.Rect(
            sc * self.cell_size, sr * self.cell_size, self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLORS["start"], start_rect)
        s_text = self.font.render("S", True, (0, 0, 0))
        s_rect = s_text.get_rect(center=start_rect.center)
        self.screen.blit(s_text, s_rect)

        # Draw goal cell (with 'G')
        gr, gc = self.maze.goal
        goal_rect = pygame.Rect(
            gc * self.cell_size, gr * self.cell_size, self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLORS["goal"], goal_rect)
        g_text = self.font.render("G", True, (255, 255, 255))
        g_rect = g_text.get_rect(center=goal_rect.center)
        self.screen.blit(g_text, g_rect)

        # Draw agent (blue circle)
        if agent_pos:
            ar, ac = agent_pos
            agent_center = (
                ac * self.cell_size + self.cell_size // 2,
                ar * self.cell_size + self.cell_size // 2,
            )
            pygame.draw.circle(
                self.screen, self.COLORS["agent"], agent_center, self.cell_size // 3
            )

        pygame.display.flip()

    def check_quit(self):
        """Check only for QUIT events; do not consume other events."""
        for event in pygame.event.get(pygame.QUIT):
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def wait_key(self):
        """
        Wait for a key press and return corresponding action.
        Also handles QUIT events.
        Returns:
            0-3 for movement,
            'quit' if Q pressed or window closed.
        """
        while True:
            self.check_quit()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_UP, pygame.K_w):
                        return 0  # up
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        return 1  # down
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        return 2  # left
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        return 3  # right
                    elif event.key == pygame.K_q:
                        return "quit"
            self.clock.tick(30)
