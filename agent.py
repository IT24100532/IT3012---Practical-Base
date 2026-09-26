# agent.py
import heapq
import random
from collections import deque

MOVES = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}
LEFT_OF = {'Up': 'Left', 'Left': 'Down', 'Down': 'Right', 'Right': 'Up'}
RIGHT_OF = {v: k for k, v in LEFT_OF.items()}
BEHIND = {'Up': 'Down', 'Down': 'Up', 'Left': 'Right', 'Right': 'Left'}


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    """Acts only on the current percept using Condition-Action rules. No memory."""

    def sense_and_act(self, percept: dict) -> str:
        facing = percept.get('facing', 'Up')

        # Condition-Action rules
        if percept['food_here']:
            return 'Suck'
        if percept['wall_ahead']:
            return LEFT_OF[facing]  # turn left
        return facing  # move forward


class ModelBasedAgent:
    """Keeps an internal model (relative position, heading, visited cells, walls) to escape loops."""

    def __init__(self):
        # Internal state, relative to where the agent started
        self.pos = (0, 0)
        self.heading = 'Up'
        self.visit_counts = {self.pos: 1}
        self.known_walls = set()
        self.known_traps = set()
        self.last_action = None

    def cell_towards(self, direction):
        dx, dy = MOVES[direction]
        return (self.pos[0] + dx, self.pos[1] + dy)

    def update_state(self, percept: dict):
        # Sensor model: what my last action did to me in the world
        if self.last_action in MOVES:
            if percept.get('bump', False):
                self.known_walls.add(self.cell_towards(self.last_action))
            else:
                self.pos = self.cell_towards(self.last_action)
                self.visit_counts[self.pos] = self.visit_counts.get(self.pos, 0) + 1
            self.heading = self.last_action

        # Transition model: walls and traps are static, so anything seen stays true
        if percept['wall_ahead']:
            self.known_walls.add(self.cell_towards(self.heading))
        if percept.get('smells_toxin', False):
            self.known_traps.add(self.pos)

    def is_unvisited(self, direction):
        cell = self.cell_towards(direction)
        return cell not in self.visit_counts and cell not in self.known_walls and cell not in self.known_traps

    def sense_and_act(self, percept: dict) -> str:
        self.update_state(percept)
        action = self.choose_action(percept)
        self.last_action = action
        return action

    def choose_action(self, percept: dict) -> str:
        ahead, left, right, back = self.heading, LEFT_OF[self.heading], RIGHT_OF[self.heading], BEHIND[self.heading]

        # Condition-Action rules that query the internal model
        if percept['food_here']:
            return 'Suck'
        if not percept['wall_ahead'] and self.is_unvisited(ahead):
            return ahead
        if self.is_unvisited(left):
            return left
        if self.is_unvisited(right):
            return right
        if self.is_unvisited(back):
            return back

        # Everything nearby is explored: go to the least-visited open neighbour to break loops
        def cost(direction):
            cell = self.cell_towards(direction)
            return self.visit_counts.get(cell, 0) + (100 if cell in self.known_traps else 0)

        options = [d for d in (ahead, left, right, back) if self.cell_towards(d) not in self.known_walls]
        return min(options, key=cost) if options else left


class SearchAgent:
    """Goal-based agent that plans a full path to the closest food (BFS, DFS or UCS) before moving."""

    def __init__(self, active_algo='BFS'):
        self.plan = []
        self.active_algo = active_algo

    def successors(self, pos, walls, grid_size):
        width, height = grid_size
        for action, (dx, dy) in MOVES.items():
            nxt = (pos[0] + dx, pos[1] + dy)
            if 0 <= nxt[0] < width and 0 <= nxt[1] < height and nxt not in walls:
                yield action, nxt

    def bfs_search(self, start, goal, walls, grid_size):
        start, goal, walls = tuple(start), tuple(goal), set(map(tuple, walls))
        frontier = deque([(start, [])])  # FIFO queue: shallowest node first
        reached = {start}
        while frontier:
            pos, path = frontier.popleft()
            if pos == goal:
                return path
            for action, nxt in self.successors(pos, walls, grid_size):
                if nxt not in reached:
                    reached.add(nxt)
                    frontier.append((nxt, path + [action]))
        return None

    def dfs_search(self, start, goal, walls, grid_size):
        start, goal, walls = tuple(start), tuple(goal), set(map(tuple, walls))
        frontier = [(start, [])]  # LIFO stack: deepest node first
        reached = set()
        while frontier:
            pos, path = frontier.pop()
            if pos == goal:
                return path
            if pos in reached:
                continue
            reached.add(pos)
            for action, nxt in self.successors(pos, walls, grid_size):
                if nxt not in reached:
                    frontier.append((nxt, path + [action]))
        return None

    def ucs_search(self, start, goal, walls, grid_size):
        start, goal, walls = tuple(start), tuple(goal), set(map(tuple, walls))
        frontier = [(0, start, [])]  # Priority queue ordered by path cost g(n)
        reached = {start: 0}
        while frontier:
            cost, pos, path = heapq.heappop(frontier)
            if pos == goal:
                return path
            if cost > reached[pos]:
                continue
            for action, nxt in self.successors(pos, walls, grid_size):
                new_cost = cost + 1  # every step costs 1
                if nxt not in reached or new_cost < reached[nxt]:
                    reached[nxt] = new_cost
                    heapq.heappush(frontier, (new_cost, nxt, path + [action]))
        return None

    def sense_and_act(self, percept: dict) -> str:
        if percept['food_here']:
            self.plan = []
            return 'Suck'

        if not self.plan:
            start = tuple(percept['agent_pos'])
            search = {'BFS': self.bfs_search, 'DFS': self.dfs_search, 'UCS': self.ucs_search}[self.active_algo]
            # Closest food first (Manhattan distance); skip any that are unreachable
            for food in sorted(percept['all_food'], key=lambda f: abs(f[0] - start[0]) + abs(f[1] - start[1])):
                path = search(start, food, percept['walls'], percept['grid_size'])
                if path:
                    self.plan = path
                    break

        return self.plan.pop(0) if self.plan else 'Stay'
