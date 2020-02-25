# Creates an intelligent agent
new_agent_01 = function(
  all_states,
  symbol,
  win_reward = 1,
  draw_reward = 0,
  loss_reward = -1,
  initial = 0,
  eps = 1,
  alpha = 0.5,
  eps_decay = TRUE) {
  
  
  # all_states has a column winner.
  # 0 = draw
  # 1 = cross wins
  # 2 = circle wins
  
  # Need to create a mapping between winner column and reward
  possible_symbols = c(1, 2)
  win_symbol = symbol
  loss_symbol = possible_symbols[possible_symbols != symbol]
  draw = 0
  
  rewards = tibble(
    winner = c(win_symbol, loss_symbol, draw, NA),
    value = c(win_reward, loss_reward, draw_reward, initial)
  )
  
  # Create value function
  value_function = all_states %>% 
    select(unique_state_id, winner) %>%
    left_join(rewards, by = "winner") %>%
    select(-winner)
  
  # Create choose action function
  choose_action_temp = function(self, board) {
    available_actions = which(board == 0)
    
    if(self$eps_decay) {
      eps = self$eps/(self$n_updates+1)
    } else {
      eps = self$eps
    }
    
    # Use epsilon-greedy
    p = runif(1)
    if(p < eps) {
      # Explore
      print("Random action")
      action_ind = sample(length(available_actions), size = 1)
      chosen_action = available_actions[action_ind]
    } else {
      # Exploit
      
      # Playing each of these actions will create a new state
      # Calculate the states
      possible_states = vapply(
        X = available_actions, 
        FUN = function(action, board) {
          board[action] = symbol
          unique_state_id = paste0(board, collapse = "")
          return(unique_state_id)
        },
        FUN.VALUE = "",
        board = board)
      # Get value of states
      ind = match(possible_states, self$value_function$unique_state_id)
      assertthat::assert_that(all(!is.na(ind)), msg = "Placing symbol would result in impossible game state, make sure that the agent has the appropriate symbol")
      possible_values = self$value_function$value[ind]
      # Choose action with highest value
      action_ind = which.max(possible_values)[1]
      chosen_action = available_actions[action_ind]
    }
    return(chosen_action)
  }
  
  # Update function
  update_function = function(self, state_history) {
    # Get value function values for observed states
    ind = match(state_history, self$value_function$unique_state_id)
    values = self$value_function$value[ind]
    
    # Update values
    n_states = length(state_history)
    for(i in (n_states - 1):1) {
      values[i] = values[i] + alpha*(values[i+1] - values[i])
    }
    
    # Load back into self
    self$value_function$value[ind] = values
    self$n_updates = self$n_updates + 1
    return(self)
  }
  
  # Prepare output
  out = list(
    "choose_action" = choose_action_temp,
    "update_function" = update_function,
    "value_function" = value_function,
    "eps" = eps,
    "eps_decay" = eps_decay,
    "n_updates" = 0)
  class(out) = c("updatable_agent", "agent")
  return(out)
}