import numpy as np
import td
import Models
import dynamic_programming_functions as dp

## Approximation methods

def mc_predict(g, policy, alpha_function, N = 1000, gamma = 0.9):
    # Gets the value function using monte carlo (using simulation)
    # Tracks state space using linear model
    
    # Initialise theta (linear weights)
    m = Models.Model_V()
    # theta = np.random.normal(size = len(generate_features(g.current_state())))

    possible_starting_states = g.all_states(include_terminal=False)

    for n in range(1, N + 1):
        g.reset()

        # Set random start - exploring starts
        starting_state_ind = np.random.choice(len(possible_starting_states))
        g.set_state(possible_starting_states[starting_state_ind])

        state_log, _, G, _ = g.play_game(policy, gamma = gamma)
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
    m = Models.Model_V()

    for n in range(1, N + 1):
        # play game
        g.reset()
        state_log, _, _, reward_log = g.play_game(policy, gamma = gamma)

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


## SARSA
def sarsa(g, epsilon_function, alpha_function, N = 10, gamma = 0.9, max_game_length = 100):
    # N = number of episodes
    # max_game_length = stops game after so many moves. Helps prevent bad policies slowing down the programme.

    # Initialise Q (arbitrarily) (model)
    m = Models.Model_Q(g)

    # Play game N times
    for n in range(1, (N+1)):
        g.reset()
        s1 = g.current_state()
        a1 = epsilon_greedy_action(g=g, model=m, s=s1, epsilon=epsilon_function(n))
        
        game_over = False
        i = 0
        while (not game_over) and (i < max_game_length):
            i += 1
            r = g.move(a1)
            s2 = g.current_state()

            game_over = g.game_over()
            if game_over:
                a2 = None
                # s2 is the last state, so don't use model
                target = r
                
            else:
                a2 = epsilon_greedy_action(g=g, model=m, s=s2, epsilon=epsilon_function(n))
                # Generate target in normal way
                target = r + gamma*m.predict(s2, a2)

            # grad(Q_hat(s,a)) = x (final multiplier).
            m.theta = m.theta + alpha_function(n)*(target - m.predict(s1, a1))*m.grad(s1, a1)

            # Update
            s1 = s2
            a1 = a2 
        
        # if n % 1 == 0:
        #     policy, value_f = get_policy_and_value_function(m, g)
        #     print(value_f[g.start])
        #     td.print_determinisitic_policy(policy, g)
        
    # Get policy from Q approximation
    print(m.theta)
    policy, value_f = get_policy_and_value_function(m, g)

    return(policy, value_f)

def epsilon_greedy_action(g, model, s, epsilon):
    p = np.random.sample()
    if p < epsilon:
        # random action
        action = np.random.choice(g.actions[s])
    else:
        # greedy action
        value = -np.Inf
        for a in g.actions[s]:
            Q_s_a = model.predict(s, a)
            if Q_s_a > value:
                value = Q_s_a
                action = a
    return(action)

def get_policy_and_value_function(model, g):
    policy = {}
    value_f = {}
    for s in g.all_states(include_terminal=False):
        value = -np.Inf
        for a in g.actions[s]:
            Q_s_a = model.predict(s, a)
            if Q_s_a > value:
                value = Q_s_a
                action = a
        policy[s] = action
        value_f[s] = value
    return(policy, value_f)
