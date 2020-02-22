play_game = function(agent_list, draw = FALSE, all_states) {
  # Function to play tic tac toe
  # agent_list is a list containing two agents
  # draw is a boolean - TRUE to draw board, false otherwise
  # all_states - data.frame created by get_all_states_and_winner function
  
  # Initialise game
  board = rep(0, 9)
  winner = NA
  
  symbols = c("cross" = 1, "nought" = 2)
  
  # Play game
  current_player_index = c(TRUE, FALSE)
  while(is.na(winner)) {
    if(draw) {
      draw_board(board)
    }
    a = choose_action(agent_list[current_player_index][[1]], board)
    # Update state
    board[a] = symbols[current_player_index]
    # Check if game has ended
    winner = get_winner(board, all_states)
    
    # Switch current player
    current_player_index = !current_player_index
  }
  # Final drawing of board
  if(draw) {
    draw_board(board)
  }
  
  # Return winner
  if(draw) {
    if(winner == 0) {
      cat("Draw\n")
    } else if(winner == 1) {
      cat("Player 1 (cross) wins\n")
    } else {
      cat("Player 2 (nought) wins\n")
    }
  }
  
  return(winner)
  
}

# Helpers
get_winner = function(board, all_states) {
  unique_state_id = paste0(board, collapse = "")
  return(all_states[all_states$unique_state_id == unique_state_id,][["winner"]])
}