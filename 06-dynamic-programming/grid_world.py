import numpy as np

class Grid:
    def __init__(self, start, blocks, rewards, terminal_states):
        """
        start: tuple (i, j) start position
        blocks: numpy logical array of blocks
        rewards: numpy float array of rewards
        terminal_states: numpy logical array of terminal states
        """
        self.height = terminal_states.shape[0] # i
        self.width = terminal_states.shape[1] # j
        self.i = start[0]
        self.j = start[1]
        self.previous_i = np.nan
        self.previous_j = np.nan

        self.rewards = rewards
        self.terminal_states = terminal_states
        self.blocks = blocks

        # Create actions
        self.actions = {}
        for i in range(self.height):
            for j in range(self.width):
                if not (blocks[i, j] or terminal_states[i, j]):
                    key = (i, j)
                    a = np.array(["D", "U", "L", "R"])
                    # Check for walls
                    if(j == 0):
                        a = a[a != "L"]
                    if(j == (self.width - 1)):
                        a = a[a != "R"]
                    if(i == 0):
                        a = a[a != "U"]
                    if(i == (self.height - 1)):
                        a = a[a != "D"]
                    # Now check for blocks in possible action spaces
                    if ("D" in a) and blocks[i + 1, j]:
                        a = a[a != "D"]
                    if ("U" in a) and blocks[i - 1, j]:
                        a = a[a != "U"]
                    if ("L" in a) and blocks[i, j - 1]:
                        a = a[a != "L"]
                    if ("R" in a) and blocks[i, j + 1]:
                        a = a[a != "R"]
                    self.actions[key] = a
                    

    def set_state(self, state):
        # state is a tuple of i,j coords (e.g. (0, 1))
        self.i = state[0]
        self.j = state[1]

    def current_state(self):
        return((self.i, self.j))

    def is_terminal(self, state):
        """ State is (i, j) tuple"""
        return(self.terminal_states[state[0], state[1]])

    def move(self, action):
        self.previous_i = self.i
        self.previous_j = self.j
        """
        action: either "U", "D", "L", "R"
        """
        if (action in self.actions[(self.i, self.j)]):
            # Do action
            if action == "D":
                self.i = self.i + 1
            elif action == "L":
                self.j = self.j - 1
            elif action == "R":
                self.j = self.j + 1
            elif action == "U":
                self.i = self.i - 1
            # return reward
            return(self.rewards[self.i, self.j])
        else:
            raise ValueError("Invalid action")

    def undo_move(self):
        """Undoes the most recent move"""
        self.i = self.previous_i
        self.j = self.previous_j

    def game_over(self):
        return(self.is_terminal((self.i, self.j)))

    def all_states(self, include_terminal = False):
        """returns a list of all states except blocks"""
        out = list(self.actions.keys())
        # append the terminal states
        for i in range(self.terminal_states.shape[0]):
            for j in range(self.terminal_states.shape[1]):
                if self.terminal_states[i,j]:
                    out.append((i, j))

        return(out)

def standard_grid():
    "Returns standard grid game"
    width = 4
    height = 3
    blocks = np.zeros(shape = (height, width), dtype = bool)
    blocks[1,1] = True

    rewards = np.zeros(shape = (height, width))
    rewards[0, 3] = 1
    rewards[1, 3] = -1

    terminal_states = np.zeros(shape = (height, width), dtype = bool)
    terminal_states[0:2, 3] = True

    return Grid(
        start=(0, 0),
        blocks=blocks,
        rewards=rewards,
        terminal_states=terminal_states,
    )

def negative_grid(step_reward=-0.1):
    "Returns grid game with negative step rewards"
    g = standard_grid()
    g.rewards = g.rewards + step_reward
    return(g)
