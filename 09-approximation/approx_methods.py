import numpy as np
import td
import Model
import dynamic_programming_functions as dp

## Approximation methods

def mc_predict(g, policy, alpha_function, N = 1000, gamma = 0.9):
    # Gets the value function using monte carlo (using simulation)
    # Tracks state space using linear model
    
    # Initialise theta (linear weights)
    m = Model.Model()
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
        for i in range(len(state_log) - 1):
            s = state_log[i]
            if s not in seen_states:
                seen_states.add(s)
                # Effectively this is just online training of the model
                # Alpha should really be selected use cross-validation etc.
                # Could retrain using all the data to get the weights right (i.e. minimise RMSE)
                m.theta = m.theta + alpha_function(n)*(G[i] - m.predict(s))*m.grad(s)

    # Generate value function
    V = {}
    for s in possible_starting_states:
        V[s] = m.predict(s)

    print("Theta: " + str(m.theta))

    return(V)

### TD(0) with approximation (semi-gradient method)
# It is semi gradient because the target uses the model, so not a true gradient.

def td0_predict(g, policy, alpha_function, N = 10, gamma = 0.9, epsilon = 0.01):

    # Use epsilon-soft
    policy = td.create_epsilon_soft_policy(policy, g, eps = epsilon)

    # Initialise theta (linear weights)
    m = Model.Model()

    for n in range(1, N + 1):
        # play game
        g.reset()
        state_log, state_action_log, G, reward_log = g.play_game(policy, gamma = gamma)

        for t in range(len(state_log) - 1):
            s = state_log[t]
            s2 = state_log[t+1]
            r = reward_log[t+1]
            
            if (t + 1) == (len(state_log) - 1):
                # The next state is the last state
                target = r
            else:
                # Generate target in normal way
                target = r + gamma*m.predict(s2)

            # grad(V_hat) = x (final multiplier). Refers to state s.
            m.theta = m.theta + alpha_function(n)*(target - m.predict(s))*m.grad(s)

    # Generate value function
    V = {}
    for s in g.all_states(include_terminal = False):
        V[s] = m.predict(s)

    print("Theta: " + str(m.theta))

    return(V)
