import numpy as np
import pdb
import warnings

class Grid:
    def __init__(self, start, blocks, rewards, terminal_states, windy = 0):
        """
        start: tuple (i, j) start position
        blocks: numpy logical array of blocks
        rewards: numpy float array of rewards
        terminal_states: numpy logical array of terminal states
        windy: probability of taking a random action
        """
        self.height = terminal_states.shape[0] # i
        self.width = terminal_states.shape[1] # j
        self.start = start
        self.i = start[0]
        self.j = start[1]
        self.previous_i = np.nan
        self.previous_j = np.nan
        
        self.windy = windy

        self.rewards = rewards
        self.terminal_states = terminal_states
        self.blocks = blocks
        self.actions_array = np.array(["U", "D", "L", "R"])

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

    def move(self, action, force = False):
        self.previous_i = self.i
        self.previous_j = self.j
        """
        action: either "U", "D", "L", "R"
        force: if True, then ignore the wind and don't take a random action
        """
        if (action in self.actions[(self.i, self.j)]):
            p = np.random.sample()
            if (p < self.windy) and not force:
                # Do random action
                action = np.random.choice(self.actions[(self.i, self.j)])

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
        if(include_terminal):
            # append the terminal states
            for i in range(self.terminal_states.shape[0]):
                for j in range(self.terminal_states.shape[1]):
                    if self.terminal_states[i,j]:
                        out.append((i, j))

        return(out)

    def play_game(self, policy, gamma = 0.9, max_iter = 1000, init_policy = None, init_state = None):
        # Plays an episode of the grid game using the given policy
        # Returns visited states and observed future discounted rewards for each state
        # initial_policy: initial policy for explorting starts (array of probabilities)
        # init_state: initial state for exploring starts
        
        # Set up initial states
        state_log = []
        reward_log = []
        state_action_log = []

        i = 0

        # initial reward
        reward = 0

        # Exploring starts (if relevant)
        if (init_policy is not None) and (init_state is not None):
            i += 1
            # Choose action based on initial_policy
            self.set_state(init_state)
            action = np.random.choice(self.actions_array, p = init_policy)

            # log state and action, and most recent reward
            state_log.append(self.current_state())
            state_action_log.append((self.current_state(), action))
            reward_log.append(reward)
            
            # Advance
            reward = self.move(action)


        # Play game
        while (not self.game_over()) and (i < max_iter):
            i += 1
            # Choose action based on policy
            action = np.random.choice(self.actions_array, p = policy[self.current_state()])

            # log state and action, and most recent reward
            state_log.append(self.current_state())
            state_action_log.append((self.current_state(), action))
            reward_log.append(reward)
            
            # Advance
            reward = self.move(action)
        
        # Log final state and reward (but no final action)
        state_log.append(self.current_state())
        reward_log.append(reward)
        state_action_log.append((self.current_state(), None))

        if i == max_iter:
            warnings.warn("Game did not finish within max_iter. Bad policy")

        # Calculate return (G)
        # G(t) = r(t+1) + gamma*G(t+1)
        # Calculate in reverse order

        G = np.zeros(len(state_log))
        # Loop in reverse order, but skip last G (leave as 0)
        for i in range(len(reward_log)- 2, -1, -1):
            G[i] = reward_log[i+1] + gamma*G[i+1]
        
        return((state_log, state_action_log, G, reward_log))

    def reset(self):
        self.set_state(self.start)

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
        start=(2, 0),
        blocks=blocks,
        rewards=rewards,
        terminal_states=terminal_states,
    )

def negative_grid(step_reward=-0.1):
    "Returns grid game with negative step rewards"
    g = standard_grid()
    g.rewards = g.rewards + step_reward
    return(g)
    
def windy_grid(step_reward = -0.1, windy = 0.5):
    g = standard_grid()
    g.rewards = g.rewards + step_reward
    g.windy = 0.5
    return(g)

def big_grid():
    "Returns big grid game"
    width = 5
    height = 5
    blocks = np.zeros(shape = (height, width), dtype = bool)
    blocks[1,1] = True
    blocks[3,3] = True

    rewards = np.zeros(shape = (height, width))
    rewards[0, 4] = 1
    rewards[1, 3] = -1

    terminal_states = np.zeros(shape = (height, width), dtype = bool)
    terminal_states[rewards != 0] = True

    return Grid(
        start=(4, 0),
        blocks=blocks,
        rewards=rewards,
        terminal_states=terminal_states,
    )

def big_grid_negative(step_reward=-0.1):
    # Returns big grid game with negative step rewards
    g = big_grid()
    g.rewards = g.rewards + step_reward
    return(g)
