# Function to generate all tic-tac-toe states, and if it is a terminal state

get_all_states_and_winner = function() {
  
  # Create all states. 9 cells each with 3 options
  # 0 for empty
  # 1 for cross
  # 2 for circle
  
  all_states = expand.grid(
    x1 = 0:2,
    x2 = 0:2,
    x3 = 0:2,
    x4 = 0:2,
    x5 = 0:2,
    x6 = 0:2,
    x7 = 0:2,
    x8 = 0:2,
    x9 = 0:2) 
  
  all_states$winner = NA
  # 0 = draw
  # 1 = player 1 win (cross)
  # 2 = player 2 win (nought)
  
  
  all_states$is_ended = FALSE
  
  # Set all states with full as a draw - winning states will be overwritten
  all_states_tmp = all_states %>%
    mutate_all(function(x) x == 0)
  draw_ind = with(all_states_tmp, {
    !(x1 | x2 | x3 | x4 | x5 | x6 | x7 | x8 | x9)
  })
  
  all_states[draw_ind, "winner"] = 0
  all_states[draw_ind, "is_ended"] = TRUE
  
  # Work out winners
  for(symbol in 1:2) {
    all_states_tmp = all_states %>%
      mutate_all(function(x) x == symbol)
    
    player_won_ind = with(all_states_tmp, {
      (x1 & x2 & x3) |
        (x4 & x5 & x6) |
        (x7 & x8 & x9) |
        (x1 & x4 & x7) |
        (x2 & x5 & x8) |
        (x3 & x6 & x9) |
        (x1 & x5 & x9) |
        (x3 & x5 & x7)
    })
    
    all_states[player_won_ind, "is_ended"] = TRUE
    all_states[player_won_ind, "winner"] = symbol
    
    # Calculate unique state id for easy joining
    # unique_state_id = rep(0, nrow(all_states))
    # for(i in 1:9) {
    #   unique_state_id = unique_state_id + 3^(i-1)*all_states[[i]]
    # }
    # all_states$unique_state_id = unique_state_id
    all_states = all_states %>% mutate(
      unique_state_id = paste0(x1, x2, x3, x4, x5, x6, x7, x8, x9)
    )
    
  }
  
  # Remove impossible states
  all_states_mat = as.matrix(select(all_states, x1:x9))
  # Replace circle with -1
  all_states_mat[all_states_mat == 2] = -1
  row_sums = rowSums(all_states_mat)
  # Sum will be zero if equal number of circle and cross
  # Sum will be 1 if cross has played another move
  # Since cross always starts, circle cannot be ahead 1 move.
  possible = row_sums %in% c(0, 1)
  all_states = all_states[possible, ]
  
  return(all_states)
}
