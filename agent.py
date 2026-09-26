# agent.py
import random

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
