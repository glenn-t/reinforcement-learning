# Training functons
train_agents = function(N, p1, p2, draw = FALSE) {
  winner = integer(length = N)
  for(i in 1:N) {
    res = play_game(list(p1, p2), draw = draw, all_states)
    p1 = update_agent(p1, res$state_history)
    p2 = update_agent(p2, res$state_history)
    winner[i] = res$winner
    if(i %% 200 == 0) {
      cat(i, mean(winner[1:i] == 1), mean(winner[(i-199):i] == 1), "\n")
    }
  }
  return(list("p1" = p1, "p2" = p2, winner = winner))
}