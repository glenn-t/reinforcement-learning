import numpy as np

## Approximation methods

def generate_features(state):
    # Generate an X vector of features for a input state

    # Generate using taylor polynomial
    out = np.array([1.0, state[0], state[1], state[0]*state[1], state[0]**2, state[1]**2, state[0]*state[1]**2, state[0]**2*state[1], state[0]**3, state[1]**3, state[0]**2*state[1]**2])
    # Just use 4 variables (reducing from 9 states to learn, to 4 parameters to learn).
    out = out[0:4]

    # If wanting to use all states - perfect model - one feature for each state
    # out = np.zeros(12)
    # cell_id = 0
    # for i in range(3):
    #     for j in range(4):
    #         out[cell_id] = state == (i,j)
    #         cell_id = cell_id + 1
    return(out)

def mc_predict(g, policy, alpha_function, N = 1000, gamma = 0.9):
    # Gets the value function using monte carlo (using simulation)
    # Tracks state space using linear model
    
    # Initialise theta (linear weights)
    theta = np.zeros(len(generate_features(g.current_state())))
    # theta = np.random.normal(size = len(generate_features(g.current_state())))

    possible_starting_states = g.all_states(include_terminal=False)

    for n in range(1, N + 1):
        g.reset()

        # Set random start - exploring starts
        starting_state_ind = np.random.choice(len(possible_starting_states))
        g.set_state(possible_starting_states[starting_state_ind])

        state_log, state_action_log, G, reward_log = g.play_game(policy, gamma = gamma)
        # Use first visit MC
        seen_states = set()
        for i in range(len(state_log)):
            s = state_log[i]
            if s not in seen_states:
                seen_states.add(s)
                # Create features
                X = generate_features(s)
                # Update model
                theta_old = theta
                # Effectively this is just online training of the model
                # Alpha should really be selected use cross-validation etc.
                # Could retrain using all the data.
                theta = theta_old + alpha_function(n)*(G[i] - theta_old.dot(X))*X

    # Generate value function
    V = {}
    for s in possible_starting_states:
        X = generate_features(s)
        V[s] = theta.dot(X)

    print("Theta: " + str(theta))

    return(V)
